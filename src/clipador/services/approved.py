"""/aprovados: every approved clip in one place - render what's missing and post it to the
connected accounts (now, on the account's next grid slot, or at a date)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlmodel import Session, col, select

from clipador.config import Settings
from clipador.errors import ValidationError
from clipador.models import (
    Candidate,
    CandidateStatus,
    PostDraft,
    Publication,
    PublicationStatus,
    Render,
    RenderKind,
    RenderStatus,
    SocialAccount,
    Source,
    Streamer,
)
from clipador.utils import utcnow

APPROVED = (CandidateStatus.APPROVED, CandidateStatus.RENDERED)
FILTERS = {
    "todos": "Todos",
    "sem_render": "Falta renderizar",
    "renderizando": "Renderizando",
    "prontos": "Prontos para postar",
    "postados": "Já postados",
}


@dataclass
class Item:
    c: Candidate
    source: Source | None
    streamer: Streamer | None
    final: Render | None  # finished final render with the file on disk
    rendering: Render | None  # final render queued/running
    progress: float | None
    failed: Render | None  # last final render that failed (and nothing newer)
    draft: PostDraft | None
    pubs: list[Publication] = field(default_factory=list)

    @property
    def state(self) -> str:
        if self.rendering is not None:
            return "renderizando"
        if self.final is None:
            return "sem_render"
        active = [p for p in self.pubs if p.status != PublicationStatus.CANCELED]
        return "postados" if active else "prontos"

    @property
    def title(self) -> str:
        if self.draft and self.draft.title:
            return self.draft.title
        return (self.c.transcript_snippet or f"Clip #{self.c.id}")[:90]


def _latest_final(
    rows: list[Render], settings: Settings
) -> tuple[Render | None, Render | None, Render | None]:
    rows = sorted(rows, key=lambda r: r.id, reverse=True)
    done = next((r for r in rows if r.status == RenderStatus.DONE and r.path
                 and settings.abs_path(r.path).is_file()), None)  # fmt: skip
    running = next(
        (r for r in rows if r.status in (RenderStatus.QUEUED, RenderStatus.RUNNING)), None
    )
    failed = rows[0] if rows and rows[0].status == RenderStatus.FAILED and done is None else None
    return done, running, failed


def items(session: Session, settings: Settings, *, streamer_id: int | None = None,
          state: str = "todos") -> list[Item]:  # fmt: skip
    from clipador.services import rendering

    q = select(Candidate).where(col(Candidate.status).in_(list(APPROVED)))
    cands = list(session.exec(q).all())
    if not cands:
        return []
    ids = [c.id for c in cands]
    sources = {s.id: s for s in session.exec(
        select(Source).where(col(Source.id).in_({c.source_id for c in cands}))).all()}  # fmt: skip
    streamers = {s.id: s for s in session.exec(select(Streamer)).all()}
    renders: dict[int, list[Render]] = {}
    for r in session.exec(
        select(Render).where(
            col(Render.candidate_id).in_(ids),
            Render.kind == RenderKind.FINAL,
            col(Render.account_id).is_(None),
        )
    ).all():
        renders.setdefault(r.candidate_id, []).append(r)
    drafts = {d.candidate_id: d for d in session.exec(select(PostDraft).where(
        col(PostDraft.candidate_id).in_(ids), PostDraft.platform == "youtube")).all()}  # fmt: skip
    pubs: dict[int, list[Publication]] = {}
    for p in session.exec(select(Publication).where(col(Publication.candidate_id).in_(ids))
                          .order_by(col(Publication.id))).all():  # fmt: skip
        pubs.setdefault(p.candidate_id, []).append(p)
    out = []
    for c in cands:
        src = sources.get(c.source_id)
        st = streamers.get(src.streamer_id) if src else None
        if streamer_id and (st is None or st.id != streamer_id):
            continue
        done, running, failed = _latest_final(renders.get(c.id, []), settings)
        prog = rendering.job_progress(session, running) if running else None
        it = Item(c, src, st, done, running, prog, failed, drafts.get(c.id), pubs.get(c.id, []))
        if state != "todos" and it.state != state:
            continue
        out.append(it)
    order = {"sem_render": 0, "renderizando": 1, "prontos": 2, "postados": 3}
    out.sort(key=lambda it: (order[it.state], -(it.c.reviewed_at or it.c.updated_at).timestamp()))
    return out


def counts(session: Session, settings: Settings) -> dict[str, int]:
    all_items = items(session, settings)
    out = {k: 0 for k in FILTERS}
    out["todos"] = len(all_items)
    for it in all_items:
        out[it.state] += 1
    return out


def render_missing(session: Session, settings: Settings, cand_ids: list[int] | None = None) -> int:
    """Queue the final render of approved clips that don't have one. Returns how many."""
    from clipador.services import analysis

    todo = [it for it in items(session, settings) if it.state == "sem_render"
            and (cand_ids is None or it.c.id in cand_ids)]  # fmt: skip
    by_source: dict[int, list[int]] = {}
    for it in todo:
        by_source.setdefault(it.c.source_id, []).append(it.c.id)
    n = 0
    for source_id, cids in by_source.items():
        try:
            _job, queued, _reused = analysis.enqueue_render(session, settings, source_id, cids,
                                                            "final", priority=5)  # fmt: skip
            n += len(queued)
        except Exception as e:  # video gone, etc.
            if cand_ids is not None:
                raise ValidationError(getattr(e, "message", str(e))) from e
    return n


def post(session: Session, settings: Settings, cand_id: int, account_id: int, *, when: str,
         privacy: str, at: datetime | None = None) -> Publication:  # fmt: skip
    """Post an approved clip: when = 'agora' | 'grade' (the account's next free slot) | 'data'.
    Does not commit."""
    from clipador.services import accounts, publishing

    if when == "agora":
        return publishing.request_publication(session, settings, cand_id, account_id,
                                              privacy=privacy)  # fmt: skip
    if when == "data":
        if at is None:
            raise ValidationError("Escolha a data e a hora do agendamento.")
        return publishing.request_publication(session, settings, cand_id, account_id,
                                              privacy=privacy, publish_at=at)  # fmt: skip
    if when != "grade":
        raise ValidationError("Escolha quando postar.")
    c = session.get(Candidate, cand_id)
    acc = session.get(SocialAccount, account_id) if account_id else None
    if c is None or c.status not in APPROVED:
        raise ValidationError("Só clips aprovados podem ser postados.")
    if acc is None or not acc.active:
        raise ValidationError("Escolha uma conta conectada.")
    r, _ready = accounts.account_render(session, settings, acc, c)
    if r is None:
        raise ValidationError("Não foi possível renderizar este clip (o vídeo da fonte sumiu?).")
    if session.exec(select(Publication).where(
            Publication.candidate_id == cand_id, Publication.account_id == acc.id,
            col(Publication.status).in_(list(accounts.ACTIVE_PUB)))).first():  # fmt: skip
        raise ValidationError(f"Este clip já foi (ou vai ser) postado em {acc.name}.")
    pub = accounts.new_publication(session, acc, c, r.id, privacy=privacy)
    cap = settings.youtube_uploads_per_project_day if acc.project_id else None
    pub.status = PublicationStatus.SCHEDULED
    pub.scheduled_for = accounts.next_slot(
        utcnow(),
        acc,
        accounts.taken_times(session, acc.id),
        group_taken=accounts.project_times(session, acc.project_id),
        group_cap=cap,
    )
    session.add(pub)
    session.flush()
    return pub


def account_options(session: Session) -> list[dict[str, Any]]:
    accs = session.exec(select(SocialAccount).where(SocialAccount.active == True)  # noqa: E712
                        .order_by(SocialAccount.platform, SocialAccount.name)).all()  # fmt: skip
    return [{"id": a.id, "label": f"{'TikTok' if a.platform == 'tiktok' else 'YouTube'} · {a.name}",
             "platform": a.platform, "streamer_ids": a.streamer_ids, "privacy": a.privacy,
             "slots": a.slots, "post_now": a.post_now} for a in accs]  # fmt: skip
