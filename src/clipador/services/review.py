"""Review queue (/revisao): status transitions, decisions with a persisted undo stack, the global
queue with filters, range validation and edge snapping.

Pure helpers (`can_transition`, `validate_range`, `snap_edge`, `order_key`) have no DB access;
the rest take a Session and never commit (the caller does).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

from sqlalchemy import func
from sqlmodel import Session, col, select

from clipador.config import Settings
from clipador.errors import ClipadorError, ValidationError
from clipador.models import (
    PENDING_STATUSES,
    RENDERABLE_STATUSES,
    Candidate,
    CandidateEnrichment,
    CandidateFeedback,
    CandidateStatus,
    Decision,
    DecisionAction,
    EnrichStatus,
    Job,
    JobStatus,
    Render,
    RenderStatus,
    Source,
    Verdict,
)
from clipador.utils import format_hms, utcnow

REJECT_REASONS = {
    "sem_graca": "Sem graça",
    "sem_contexto": "Sem contexto",
    "audio_ruim": "Áudio ruim",
    "risco_copyright_musica": "Risco de música com direitos",
    "repetido": "Repetido",
    "conteudo_sensivel": "Conteúdo sensível",
    "outro": "Outro",
}

ACTION_TARGET = {
    DecisionAction.APPROVE: CandidateStatus.APPROVED,
    DecisionAction.REJECT: CandidateStatus.REJECTED,
    DecisionAction.SKIP: CandidateStatus.SKIPPED,
    DecisionAction.REOPEN: CandidateStatus.SUGGESTED,
}
# Allowed status changes made by a review decision. 'rendered' is only ever set by the render
# worker; from it you can still reject or reopen.
TRANSITIONS: dict[str, set[str]] = {
    CandidateStatus.SUGGESTED: {
        CandidateStatus.APPROVED, CandidateStatus.REJECTED, CandidateStatus.SKIPPED,
    },
    CandidateStatus.APPROVED: {CandidateStatus.REJECTED, CandidateStatus.SUGGESTED},
    CandidateStatus.REJECTED: {CandidateStatus.APPROVED, CandidateStatus.SUGGESTED},
    CandidateStatus.SKIPPED: {
        CandidateStatus.APPROVED, CandidateStatus.REJECTED, CandidateStatus.SKIPPED,
        CandidateStatus.SUGGESTED,
    },
    CandidateStatus.RENDERED: {CandidateStatus.REJECTED, CandidateStatus.SUGGESTED},
}  # fmt: skip
STATUS_LABELS = {
    CandidateStatus.SUGGESTED: "sugerido",
    CandidateStatus.APPROVED: "aprovado",
    CandidateStatus.REJECTED: "rejeitado",
    CandidateStatus.SKIPPED: "pulado",
    CandidateStatus.RENDERED: "renderizado",
}
ACTION_LABELS = {
    DecisionAction.APPROVE: "aprovou",
    DecisionAction.REJECT: "rejeitou",
    DecisionAction.SKIP: "pulou",
    DecisionAction.UNDO: "desfez",
    DecisionAction.EDIT_RANGE: "ajustou o corte de",
    DecisionAction.REOPEN: "reabriu",
    DecisionAction.EDIT_CHECKS: "mudou as verificações de",
}
UNDOABLE = {a for a in DecisionAction if a != DecisionAction.UNDO}
EDIT_COALESCE_SECONDS = 90  # consecutive edits of the same clip become one undo step
PREFETCH_PRIORITY = -10


def can_transition(current: str, target: str) -> bool:
    return target in TRANSITIONS.get(current, set())


# --- pre-publish checks (5B) -------------------------------------------------------------------

CHECK_KEYS = ("musica_conferida", "alertas_reconhecidos")


def has_alerts(e: CandidateEnrichment | None) -> bool:
    return bool(e and e.status == EnrichStatus.OK and e.alertas)


def ready_to_publish(c: Candidate, has_alerts: bool) -> bool:
    """Approved (or already rendered) AND background music checked AND — only when the AI
    raised alerts — the alerts read. Approving is never blocked by this; only the badge and
    the 'prontos' filter depend on it."""
    ck = c.checks or {}
    return (
        c.status in RENDERABLE_STATUSES
        and bool(ck.get("musica_conferida"))
        and (not has_alerts or bool(ck.get("alertas_reconhecidos")))
    )


def alerted_ids(session: Session, candidate_ids: list[int]) -> set[int]:
    """Candidates whose (successful) AI enrichment raised alerts."""
    if not candidate_ids:
        return set()
    rows = session.exec(
        select(CandidateEnrichment).where(col(CandidateEnrichment.candidate_id).in_(candidate_ids))
    ).all()
    return {e.candidate_id for e in rows if has_alerts(e)}


def ready_ids(session: Session, cands: list[Candidate]) -> set[int]:
    alerted = alerted_ids(session, [c.id for c in cands if c.status in RENDERABLE_STATUSES])
    return {c.id for c in cands if ready_to_publish(c, c.id in alerted)}


def set_checks(session: Session, candidate_id: int, **values: bool) -> Candidate:
    """Tick/untick pre-publish checks (keys in CHECK_KEYS). Each real change is recorded as an
    'edit_checks' decision with the previous state, so Z (undo) restores it like any other."""
    unknown = set(values) - set(CHECK_KEYS)
    if unknown:
        raise ValidationError(f"Verificação desconhecida: {', '.join(sorted(unknown))}.")
    c = _get_candidate(session, candidate_id)
    current = {k: bool((c.checks or {}).get(k)) for k in CHECK_KEYS}
    new = {**current, **{k: bool(v) for k, v in values.items()}}
    if new == current:
        return c
    session.add(
        Decision(candidate_id=c.id, action=DecisionAction.EDIT_CHECKS,
                 prev_state=snapshot(session, c))
    )  # fmt: skip
    c.checks = new
    c.updated_at = utcnow()
    session.add(c)
    session.flush()
    return c


# --- state snapshots ---------------------------------------------------------------------------


def _feedback(session: Session, candidate_id: int) -> CandidateFeedback | None:
    return session.exec(
        select(CandidateFeedback).where(CandidateFeedback.candidate_id == candidate_id)
    ).first()


def snapshot(session: Session, c: Candidate) -> dict[str, Any]:
    fb = _feedback(session, c.id)
    return {
        "status": c.status,
        "reviewed_at": c.reviewed_at.isoformat() if c.reviewed_at else None,
        "user_start": c.user_start,
        "user_end": c.user_end,
        "checks": dict(c.checks) if c.checks else None,
        "feedback": (
            {"verdict": fb.verdict, "note": fb.note, "reason_tag": fb.reason_tag} if fb else None
        ),
    }


def _restore(session: Session, c: Candidate, state: dict[str, Any]) -> None:
    c.status = state.get("status", c.status)
    ra = state.get("reviewed_at")
    c.reviewed_at = datetime.fromisoformat(ra) if ra else None
    c.user_start, c.user_end = state.get("user_start"), state.get("user_end")
    if "checks" in state:  # snapshots made before 5B don't have it: leave the checks alone
        c.checks = dict(state["checks"]) if state["checks"] else None
    c.updated_at = utcnow()
    session.add(c)
    fb = _feedback(session, c.id)
    want = state.get("feedback")
    if want is None:
        if fb is not None:
            session.delete(fb)
        return
    fb = fb or CandidateFeedback(candidate_id=c.id, verdict=want["verdict"])
    fb.verdict, fb.note, fb.reason_tag = (
        want["verdict"],
        want.get("note", ""),
        want.get("reason_tag"),
    )
    fb.updated_at = utcnow()
    session.add(fb)


def _set_feedback(
    session: Session, c: Candidate, verdict: str, reason_tag: str | None, note: str | None
) -> None:
    fb = _feedback(session, c.id) or CandidateFeedback(candidate_id=c.id, verdict=verdict)
    fb.verdict, fb.reason_tag = verdict, reason_tag
    if note is not None:
        fb.note = note[:500]
    fb.updated_at = utcnow()
    session.add(fb)


# --- decisions -----------------------------------------------------------------------------------


@dataclass
class DecisionResult:
    decision: Decision
    candidate: Candidate
    render_job: Job | None = None
    notice: str | None = None


def _get_candidate(session: Session, candidate_id: int) -> Candidate:
    c = session.get(Candidate, candidate_id)
    if c is None:
        raise ValidationError("Candidato não encontrado.")
    return c


def apply_decision(
    session: Session,
    candidate_id: int,
    action: str,
    reason_tag: str | None = None,
    note: str | None = None,
    *,
    settings: Settings | None = None,
) -> DecisionResult:
    """approve | reject | skip | reopen. Records a Decision (with the previous state) and keeps
    CandidateFeedback in sync: approve -> 'bom', reject -> 'ruim' + reason; skip/reopen keep it.
    With `settings.render_on_approve`, approving queues the final render (cache-aware)."""
    try:
        act = DecisionAction(action)
    except ValueError:
        raise ValidationError(f"Ação desconhecida: {action}.") from None
    if act not in ACTION_TARGET:
        raise ValidationError("Use undo_last/edit_range para desfazer ou ajustar o corte.")
    if reason_tag is not None and reason_tag not in REJECT_REASONS:
        raise ValidationError(f"Motivo de rejeição desconhecido: {reason_tag}.")
    c = _get_candidate(session, candidate_id)
    target = ACTION_TARGET[act]
    if not can_transition(c.status, target):
        raise ValidationError(
            f"Não dá para mudar de '{STATUS_LABELS.get(c.status, c.status)}' para "
            f"'{STATUS_LABELS.get(target, target)}'."
        )
    prev = snapshot(session, c)
    c.status = target
    c.reviewed_at = None if act == DecisionAction.REOPEN else utcnow()
    c.updated_at = utcnow()
    session.add(c)
    if act == DecisionAction.APPROVE:
        _set_feedback(session, c, Verdict.GOOD, None, note)
    elif act == DecisionAction.REJECT:
        _set_feedback(session, c, Verdict.BAD, reason_tag, note)

    job, notice = None, None
    if act == DecisionAction.APPROVE and settings is not None and settings.render_on_approve:
        job, notice = _render_on_approve(session, settings, c)
        if job is not None:
            prev["_render_job"] = job.id
    d = Decision(
        candidate_id=c.id, action=act, prev_state=prev,
        reason_tag=reason_tag if act == DecisionAction.REJECT else None,
        note=(note or "")[:500],
    )  # fmt: skip
    session.add(d)
    session.flush()
    return DecisionResult(d, c, job, notice)


def _render_on_approve(session: Session, settings: Settings, c: Candidate):
    from clipador.services import analysis

    try:  # plan_render raises before anything is added to the session
        job, _todo, reused = analysis.enqueue_render(
            session, settings, c.source_id, [c.id], "final"
        )
    except (ValidationError, ClipadorError) as e:
        return None, f"Aprovado, mas o render final não foi para a fila: {e.message}"
    if job is not None:
        return job, f"Aprovado; render final na fila (job #{job.id})."
    if reused:
        return None, "Aprovado; o render final com estes ajustes já existe."
    return None, "Aprovado; o render final já estava na fila."


def _undo_target(session: Session) -> Decision | None:
    """Walk the history newest-first; each 'undo' cancels the next older live decision."""
    pending_undos = 0
    for d in session.exec(select(Decision).order_by(col(Decision.id).desc())):
        if d.action == DecisionAction.UNDO:
            pending_undos += 1
        elif pending_undos:
            pending_undos -= 1
        else:
            return d
    return None


def undo_last(session: Session, settings: Settings | None = None) -> Decision | None:
    """Undo the most recent live decision (any candidate). Call again to keep undoing.
    Returns the decision that was undone (None when there is nothing left)."""
    target = _undo_target(session)
    if target is None:
        return None
    c = _get_candidate(session, target.candidate_id)
    before = snapshot(session, c)
    _restore(session, c, target.prev_state)
    job_id = target.prev_state.get("_render_job")
    if job_id:
        _cancel_queued_render_job(session, job_id)
    session.add(
        Decision(candidate_id=c.id, action=DecisionAction.UNDO, prev_state={"undid": target.id,
                                                                            **before})
    )  # fmt: skip
    session.flush()
    return target


def _cancel_queued_render_job(session: Session, job_id: int) -> None:
    """The final render queued by an approval that was undone: drop it if it hasn't started."""
    job = session.get(Job, job_id)
    if job is None or job.status != JobStatus.QUEUED:
        return
    from clipador.worker import queue

    queue.cancel(session, job_id)
    for r in session.exec(select(Render).where(Render.job_id == job_id)).all():
        if r.status == RenderStatus.QUEUED:
            r.status, r.error, r.updated_at = RenderStatus.CANCELED, "Aprovação desfeita.", utcnow()
            session.add(r)


def history(session: Session, limit: int = 15) -> list[dict[str, Any]]:
    """Latest decisions for the UI, each flagged `undone` when a later undo cancelled it."""
    rows = list(session.exec(select(Decision).order_by(col(Decision.id).desc()).limit(200)))
    undone = {d.prev_state.get("undid") for d in rows if d.action == DecisionAction.UNDO}
    cands = {
        c.id: c
        for c in session.exec(
            select(Candidate).where(col(Candidate.id).in_({d.candidate_id for d in rows}))
        )
    }
    out = []
    for d in rows[:limit]:
        c = cands.get(d.candidate_id)
        out.append({"d": d, "c": c, "undone": d.id in undone,
                    "label": ACTION_LABELS.get(d.action, d.action)})  # fmt: skip
    return out


# --- range adjust ------------------------------------------------------------------------------


@dataclass(frozen=True)
class RangeCheck:
    start: float
    end: float
    warnings: list[str] = field(default_factory=list)


def _fmt_limit(seconds: float) -> str:
    if seconds >= 60 and seconds % 60 == 0:
        return f"{int(seconds // 60)} min"
    return f"{seconds:g} s"


def validate_range(
    start: float,
    end: float,
    *,
    duration: float | None,
    min_s: float,
    max_s: float,
    others: list[tuple[int, float, float]] = (),  # (rank, start, end) of approved clips
) -> RangeCheck:
    """Pure check of a hand-adjusted range. Errors raise ValidationError (PT-BR); overlapping
    another approved clip of the same source is only a warning."""
    limit = duration if duration else float("inf")
    if not (0 <= start < end <= limit):
        raise ValidationError("O fim precisa ser depois do início, dentro do vídeo.")
    if end - start > max_s:
        raise ValidationError(f"Trecho longo demais (máx. {_fmt_limit(max_s)}).")
    if end - start < min_s:
        raise ValidationError(f"Trecho curto demais (mín. {_fmt_limit(min_s)}).")
    warnings = [
        f"Sobrepõe o candidato #{rank} já aprovado ({format_hms(s)}–{format_hms(e)})."
        for rank, s, e in others
        if s < end and e > start
    ]
    return RangeCheck(round(start, 2), round(end, 2), warnings)


def snap_edge(
    t: float,
    words: list[tuple[float, float]],
    side: str,
    *,
    tolerance: float = 0.35,
    min_pause: float = 0.3,
) -> float:
    """Snap a clip edge to the nearest word boundary (starts for 'start', ends for 'end') or the
    middle of a pause between words, if one is within `tolerance` seconds; else keep `t`."""
    ws = sorted(words)
    edges = [s if side == "start" else e for s, e in ws]
    edges += [(a[1] + b[0]) / 2 for a, b in itertools.pairwise(ws) if b[0] - a[1] >= min_pause]
    best = min(edges, key=lambda x: abs(x - t), default=None)
    if best is None or abs(best - t) > tolerance:
        return t
    return round(best, 3)


def approved_neighbours(session: Session, c: Candidate) -> list[tuple[int, float, float]]:
    rows = session.exec(
        select(Candidate).where(
            Candidate.source_id == c.source_id,
            Candidate.id != c.id,
            col(Candidate.status).in_(RENDERABLE_STATUSES),
        )
    ).all()
    return [(o.final_rank or o.rank, o.eff_start, o.eff_end) for o in rows]


def edit_range(
    session: Session,
    settings: Settings,
    candidate_id: int,
    start: float | None,
    end: float | None,
    *,
    reset: bool = False,
    snap_words: list[tuple[float, float]] | None = None,
) -> tuple[Candidate, list[str]]:
    """Validate + save user_start/user_end (or reset to the suggestion) and record an
    'edit_range' decision. Consecutive edits of the same clip within a short time coalesce into
    one undo step. Renders are invalidated by their params hash (start/end are in it)."""
    c = _get_candidate(session, candidate_id)
    warnings: list[str] = []
    if reset:
        new = (None, None)
    else:
        if start is None or end is None:
            raise ValidationError("Início e fim devem ser números (segundos).")
        if snap_words:
            start = snap_edge(start, snap_words, "start")
            end = snap_edge(end, snap_words, "end")
        source = session.get(Source, c.source_id)
        chk = validate_range(
            start, end, duration=source.duration_seconds if source else None,
            min_s=settings.clip_min_seconds, max_s=settings.clip_max_seconds,
            others=approved_neighbours(session, c),
        )  # fmt: skip
        new, warnings = (chk.start, chk.end), chk.warnings
    if (c.user_start, c.user_end) == new:
        return c, warnings
    last = session.exec(select(Decision).order_by(col(Decision.id).desc())).first()
    coalesce = (
        last is not None
        and last.action == DecisionAction.EDIT_RANGE
        and last.candidate_id == c.id
        and utcnow() - _aware(last.created_at) < timedelta(seconds=EDIT_COALESCE_SECONDS)
    )
    if not coalesce:
        prev = snapshot(session, c)
        session.add(Decision(candidate_id=c.id, action=DecisionAction.EDIT_RANGE, prev_state=prev))
    c.user_start, c.user_end = new
    c.updated_at = utcnow()
    session.add(c)
    session.flush()
    return c, warnings


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


# --- the queue ---------------------------------------------------------------------------------

ORDERS = {"rank": "Posição (nota final)", "nota": "Maior nota", "horario": "Horário no vídeo"}
STATUS_FILTERS = {
    "pendentes": "Pendentes (sugeridos e pulados)",
    "todos": "Todos",
    CandidateStatus.SUGGESTED: "Só sugeridos",
    CandidateStatus.SKIPPED: "Só pulados",
    CandidateStatus.APPROVED: "Aprovados",
    CandidateStatus.REJECTED: "Rejeitados",
    CandidateStatus.RENDERED: "Renderizados",
    "prontos": "Prontos para publicar (aprovado + verificações)",
    "ao_vivo": "Ao vivo (pendentes das lives)",
}
PENDING_LIKE = ("pendentes", "ao_vivo")  # filters that walk suggested, then skipped


@dataclass(frozen=True)
class ReviewFilters:
    source_id: int | None = None
    streamer_id: int | None = None
    platform: str | None = None
    min_score: float | None = None
    status: str = "pendentes"
    order: str = "rank"

    @classmethod
    def from_params(cls, params) -> ReviewFilters:
        def num(name, cast):
            raw = str(params.get(name) or "").strip().replace(",", ".")
            try:
                return cast(raw) if raw else None
            except ValueError:
                return None

        status = str(params.get("status") or "pendentes")
        order = str(params.get("ordem") or "rank")
        return cls(
            source_id=num("fonte", int), streamer_id=num("streamer", int),
            platform=str(params.get("plataforma") or "") or None, min_score=num("nota_min", float),
            status=status if status in STATUS_FILTERS else "pendentes",
            order=order if order in ORDERS else "rank",
        )  # fmt: skip

    def params(self) -> dict[str, str]:
        out = {
            "fonte": self.source_id, "streamer": self.streamer_id, "plataforma": self.platform,
            "nota_min": self.min_score,
            "status": None if self.status == "pendentes" else self.status,
            "ordem": None if self.order == "rank" else self.order,
        }  # fmt: skip
        return {k: f"{v:g}" if isinstance(v, float) else str(v) for k, v in out.items()
                if v is not None}  # fmt: skip

    def query(self, **extra) -> str:
        p = {**self.params(), **{k: str(v) for k, v in extra.items() if v is not None}}
        return urlencode(p)

    def matches_status(self, status: str, ready: bool = False) -> bool:
        """`ready` (ready_to_publish) is only needed for the 'prontos' filter."""
        if self.status == "todos":
            return True
        if self.status in PENDING_LIKE:  # ao_vivo: scope() already kept only live candidates
            return status in PENDING_STATUSES
        if self.status == "prontos":
            return ready
        return status == self.status


def _score(c: Candidate) -> float:
    return c.final_score if c.final_score is not None else c.score


def order_key(order: str):
    keys = {
        "rank": lambda c: (c.final_rank if c.final_rank is not None else c.rank, -_score(c),
                           c.source_id, c.id),
        "nota": lambda c: (-_score(c), c.source_id, c.id),
        "horario": lambda c: (c.source_id, c.eff_start, c.id),
    }  # fmt: skip
    return keys.get(order, keys["rank"])


LIVE_TOP_N = 5  # the best N pending clips of every live always show up in the review


def hidden_live_clips(cands: list[Candidate], mins: dict[int, float]) -> set[int]:
    """Pending live clips kept out of the review: under the session's 'nota mínima' AND not
    among the session's best LIVE_TOP_N. So every live shows its best moments, without the
    hundreds of weak candidates a long live produces."""
    pending = [c for c in cands if c.live_session_id
               and c.status in (CandidateStatus.SUGGESTED, CandidateStatus.SKIPPED)]  # fmt: skip
    by_live: dict[int, list[Candidate]] = {}
    for c in pending:
        by_live.setdefault(c.live_session_id, []).append(c)
    hidden: set[int] = set()
    for lid, rows in by_live.items():
        rows.sort(key=lambda c: -_score(c))
        hidden |= {c.id for c in rows[LIVE_TOP_N:] if _score(c) < mins.get(lid, 0.0)}
    return hidden


def scope(session: Session, f: ReviewFilters) -> list[Candidate]:
    """Candidates of the latest analysis run of each source, filtered by everything but status,
    in queue order."""
    latest = select(Candidate.source_id, func.max(Candidate.analysis_run_id).label("run")).group_by(
        Candidate.source_id
    )
    if f.source_id is not None:
        latest = latest.where(Candidate.source_id == f.source_id)
    runs = [r for _sid, r in session.exec(latest).all()]
    if not runs:
        return []
    q = (
        select(Candidate)
        .join(Source, Source.id == Candidate.source_id)
        .where(col(Candidate.analysis_run_id).in_(runs))
    )
    if f.streamer_id is not None:
        q = q.where(Source.streamer_id == f.streamer_id)
    if f.platform:
        q = q.where(Source.platform == f.platform)
    if f.status == "ao_vivo":
        q = q.where(Candidate.live_session_id != None)  # noqa: E711
    cands = list(session.exec(q).all())
    if f.status != "todos":  # live clips below their session's minimum score stay out of the way
        from clipador.models import LiveSession

        ids = {c.live_session_id for c in cands if c.live_session_id}
        mins = (
            {
                ls.id: ls.min_score
                for ls in session.exec(
                    select(LiveSession).where(col(LiveSession.id).in_(ids))
                ).all()
            }
            if ids
            else {}
        )
        hidden = hidden_live_clips(cands, mins)
        cands = [c for c in cands if c.id not in hidden]
    if f.min_score is not None:
        cands = [c for c in cands if _score(c) >= f.min_score]
    key = order_key(f.order)
    if f.order == "horario":
        return sorted(cands, key=key)
    # Live moments that crossed the auto-clip score go first: they are the urgent ones.
    return sorted(cands, key=lambda c: (not c.auto_clipped, key(c)))


def next_pending(
    session: Session, f: ReviewFilters, after_id: int | None = None
) -> Candidate | None:
    """Next candidate to review. With the default 'pendentes' filter, suggested ones come before
    skipped ones; with `after_id`, the search starts right after that candidate (wrapping), so
    skipping moves on instead of showing the same clip again."""
    items = scope(session, f)
    if not items:
        return None
    pos = next((i for i, c in enumerate(items) if c.id == after_id), -1)
    rotated = items[pos + 1 :] + items[: pos + 1] if pos >= 0 else items
    if pos >= 0 and len(rotated) > 1:
        rotated = rotated[:-1]  # the one we just left goes last (only if nothing else)
    tiers = (
        [(CandidateStatus.SUGGESTED,), (CandidateStatus.SKIPPED,)]
        if f.status in PENDING_LIKE
        else [None]
    )
    ready = ready_ids(session, items) if f.status == "prontos" else set()
    for tier in tiers:
        for c in rotated:
            if (c.status in tier) if tier else f.matches_status(c.status, c.id in ready):
                return c
    if pos >= 0:
        last = items[pos]
        if f.matches_status(last.status, last.id in ready):
            return last
    return None


def upcoming(session: Session, f: ReviewFilters, current_id: int | None, n: int) -> list[Candidate]:
    """The next `n` pending candidates after `current_id` (for prefetching previews)."""
    out: list[Candidate] = []
    seen = {current_id}
    after = current_id
    probe = (
        replace(f, status="pendentes")
        if f.status not in PENDING_STATUSES and f.status not in PENDING_LIKE
        else f
    )
    for _ in range(n):
        c = next_pending(session, probe, after)
        if c is None or c.id in seen:
            break
        out.append(c)
        seen.add(c.id)
        after = c.id
    return out


def progress(session: Session, f: ReviewFilters, current_id: int | None = None) -> dict[str, Any]:
    items = scope(session, f)
    by = {s: 0 for s in CandidateStatus}
    for c in items:
        by[c.status] = by.get(c.status, 0) + 1
    ids = [c.id for c in items]
    pos = ids.index(current_id) if current_id in ids else None
    return {
        "total": len(items),
        "position": pos + 1 if pos is not None else None,
        "suggested": by[CandidateStatus.SUGGESTED],
        "skipped": by[CandidateStatus.SKIPPED],
        "approved": by[CandidateStatus.APPROVED] + by[CandidateStatus.RENDERED],
        "rejected": by[CandidateStatus.REJECTED],
        "ready": len(ready_ids(session, items)),
        "reviewed": len(items) - by[CandidateStatus.SUGGESTED] - by[CandidateStatus.SKIPPED],
        "prev_id": ids[pos - 1] if pos else None,
        "next_id": ids[pos + 1] if pos is not None and pos + 1 < len(ids) else None,
    }


def prefetch_previews(
    session: Session, settings: Settings, candidates: list[Candidate]
) -> list[int]:
    """Queue faithful previews for the given candidates at low priority (cache-aware: nothing
    is queued twice). Returns the ids of the jobs created. Errors (e.g. video missing) are
    ignored: prefetching is best-effort."""
    from clipador.services import analysis

    jobs = []
    for c in candidates:  # one at a time: a failing plan never leaves half-created rows
        try:
            job, _todo, _reused = analysis.enqueue_render(
                session, settings, c.source_id, [c.id], "preview", priority=PREFETCH_PRIORITY
            )
        except (ValidationError, ClipadorError):
            continue
        if job is not None:
            jobs.append(job.id)
    return jobs
