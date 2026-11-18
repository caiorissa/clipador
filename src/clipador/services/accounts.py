"""Social accounts + auto-post (step 6B).

The rule that never changes: only clips YOU approved are posted. With auto-post on, an account
picks up the approved clips of its streamers and posts them by itself on its time grid
(e.g. 12:00 / 18:00 / 21:00, at most N per day, a minimum gap between posts). A clip with a
strong AI alert or lots of background music is HELD for a manual "Liberar". "Pausar" stops an
account (or all of them) at once; scheduled clips just wait.

YouTube: uploaded up to `youtube_upload_lead_hours` before its slot, private with publishAt =
the slot, so it goes public on time even if the PC is off by then. TikTok has no scheduled
posting in its API: it is uploaded at the slot (the worker must be running).
"""

from __future__ import annotations

import contextlib
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from typing import Any

from sqlalchemy import Engine
from sqlmodel import Session, col, select

from clipador.config import Settings
from clipador.errors import ValidationError
from clipador.models import (
    Candidate,
    CandidateEnrichment,
    CandidateStatus,
    EnrichStatus,
    PostDraft,
    Publication,
    PublicationStatus,
    SocialAccount,
    Source,
)
from clipador.utils import utcnow

PLATFORMS = ("youtube", "tiktok")
PLATFORM_LABELS = {"youtube": "YouTube", "tiktok": "TikTok"}
SEVERE_ALERTS = {
    "discurso_de_odio_ou_ofensa",
    "violencia_real_ou_autolesao",
    "conteudo_sexual",
    "dados_pessoais_ou_doxxing",
}
MUSIC_HOLD_PCT = 60.0
TIKTOK_MAX_PER_DAY = 15  # Content Posting API limit per creator account (approx.)


def quota_tz() -> tzinfo:
    """YouTube API quotas reset at midnight Pacific time."""
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo("America/Los_Angeles")
    except Exception:  # Windows without tzdata: close enough (off by 1 h in summer)
        return timezone(timedelta(hours=-8))


ACTIVE_PUB = (
    PublicationStatus.SCHEDULED,
    PublicationStatus.QUEUED,
    PublicationStatus.UPLOADING,
    PublicationStatus.PUBLISHED,
)


def aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


# --- credentials ---------------------------------------------------------------------------------


class AccountStore:
    """Token of one account, read/written in its DB row (each call its own session)."""

    def __init__(self, engine: Engine, account_id: int):
        self.engine, self.account_id = engine, account_id

    def load(self) -> dict[str, Any] | None:
        with Session(self.engine) as s:
            a = s.get(SocialAccount, self.account_id)
            return dict(a.credentials) if a and a.credentials else None

    def save(self, token: dict[str, Any]) -> None:
        with Session(self.engine) as s:
            a = s.get(SocialAccount, self.account_id)
            if a is not None:
                a.credentials, a.updated_at = dict(token), utcnow()
                s.add(a)
                s.commit()


def client_for(engine: Engine, settings: Settings, account: SocialAccount):
    from clipador.services import tiktok, youtube

    store = AccountStore(engine, account.id)
    if account.platform == "tiktok":
        return tiktok.TikTok(settings, store=store)
    return youtube.YouTube(settings, store=store)


def upsert_account(
    session: Session, platform: str, external_id: str, name: str, token: dict[str, Any]
) -> SocialAccount:
    """New account, or new token for an account connected before. Does not commit."""
    acc = session.exec(
        select(SocialAccount).where(
            SocialAccount.platform == platform, SocialAccount.external_id == external_id
        )
    ).first()
    if acc is None:
        acc = SocialAccount(platform=platform, external_id=external_id, name=name)
        if platform == "tiktok":
            acc.privacy = "private"  # unaudited apps can only post privately
    acc.name, acc.credentials, acc.active = name or acc.name, token, True
    acc.updated_at = utcnow()
    session.add(acc)
    session.flush()
    return acc


def import_legacy_youtube(session: Session, settings: Settings) -> SocialAccount | None:
    """The first version kept one YouTube token in data/youtube_token.json: make it an account
    (and attach its old publications to it). Does not commit."""
    from clipador.services import youtube

    tok = youtube.load_token(settings)
    if not tok or not tok.get("refresh_token"):
        return None
    ch = tok.get("channel") or {}
    acc = upsert_account(
        session,
        "youtube",
        ch.get("id") or "legacy",
        ch.get("title") or "YouTube",
        {k: v for k, v in tok.items() if k != "channel"},
    )
    for p in session.exec(select(Publication).where(Publication.account_id == None)).all():  # noqa: E711
        p.account_id = acc.id
        session.add(p)
    youtube.token_path(settings).unlink(missing_ok=True)
    return acc


def accounts(session: Session, *, active_only: bool = False) -> list[SocialAccount]:
    q = select(SocialAccount).order_by(SocialAccount.platform, SocialAccount.name)
    if active_only:
        q = q.where(SocialAccount.active == True)  # noqa: E712
    return list(session.exec(q).all())


def parse_slots(text: str) -> list[str]:
    """'12:00, 18h, 21:30' -> ['12:00', '18:00', '21:30'] (sorted, unique)."""
    out = set()
    for raw in text.replace(";", ",").replace(" ", ",").split(","):
        raw = raw.strip().lower().replace("h", ":")
        if not raw:
            continue
        hh, _, mm = raw.partition(":")
        try:
            h, m = int(hh), int(mm or 0)
        except ValueError as e:
            raise ValidationError(f"Horário inválido: '{raw}' (use 12:00, 18:30…).") from e
        if not (0 <= h < 24 and 0 <= m < 60):
            raise ValidationError(f"Horário inválido: '{raw}'.")
        out.add(f"{h:02d}:{m:02d}")
    return sorted(out)


# --- the time grid (pure) ---------------------------------------------------------------------


def _local_tz() -> tzinfo:
    return datetime.now().astimezone().tzinfo


def next_slot(
    now: datetime,
    acc: SocialAccount,
    taken: list[datetime],
    tz: tzinfo | None = None,
    horizon_days: int = 60,
    group_taken: list[datetime] | None = None,
    group_cap: int | None = None,
    group_tz: tzinfo | None = None,
) -> datetime:
    """Earliest time (UTC) >= now that respects the account's grid, daily cap and min gap,
    given the times already `taken` by its other posts. `group_*`: a second daily cap shared
    with other accounts (the YouTube upload quota of a Google Cloud project, whose day is
    Pacific time)."""
    tz = tz or _local_tz()
    gtz = group_tz or quota_tz()
    gtaken = sorted(aware(t) for t in (group_taken or []))

    def group_full(t: datetime) -> bool:
        if not group_cap:
            return False
        d = t.astimezone(gtz).date()
        return sum(1 for x in gtaken if x.astimezone(gtz).date() == d) >= group_cap

    gap = timedelta(minutes=max(0, acc.min_gap_minutes))
    cap = max(1, acc.max_per_day)
    taken = sorted(aware(t) for t in taken)

    def per_day(d: date) -> int:
        return sum(1 for t in taken if t.astimezone(tz).date() == d)

    def free(t: datetime) -> bool:
        return all(abs(t - x) >= gap for x in taken) and not any(t == x for x in taken)

    now = aware(now)
    if acc.post_now or not acc.slots:
        t = now
        for _ in range(horizon_days * 48):
            d = t.astimezone(tz).date()
            if per_day(d) >= cap:  # day full: first minute of the next day
                nxt = datetime.combine(d + timedelta(days=1), datetime.min.time(), tz)
                t = nxt.astimezone(UTC)
                continue
            if group_full(t):  # project quota of that day used: next quota day
                gd = t.astimezone(gtz).date() + timedelta(days=1)
                t = datetime.combine(gd, datetime.min.time(), gtz).astimezone(UTC)
                continue
            clash = [x for x in taken if abs(t - x) < gap]
            if not clash:
                return t
            t = max(clash) + gap
        return t
    day = now.astimezone(tz).date()
    for i in range(horizon_days):
        d = day + timedelta(days=i)
        if per_day(d) >= cap:
            continue
        for hhmm in acc.slots:
            h, m = (int(x) for x in hhmm.split(":"))
            t = datetime(d.year, d.month, d.day, h, m, tzinfo=tz).astimezone(UTC)
            if t >= now and free(t) and not group_full(t):
                return t
    raise ValidationError("A grade de horários desta conta está cheia pelos próximos 60 dias.")


# --- what gets posted -------------------------------------------------------------------------


def hold_reason(session: Session, c: Candidate) -> str | None:
    e = session.exec(
        select(CandidateEnrichment).where(CandidateEnrichment.candidate_id == c.id)
    ).first()
    if e and e.status == EnrichStatus.OK:
        bad = sorted({a.get("tipo") for a in (e.alertas or [])} & SEVERE_ALERTS)
        if bad:
            return "alerta da IA: " + ", ".join(bad)
    music = (c.breakdown or {}).get("music")
    if music is not None and music >= MUSIC_HOLD_PCT:
        return f"música em {music:.0f}% do trecho"
    return None


def draft_for(session: Session, cand_id: int, platform: str) -> PostDraft | None:
    return session.exec(
        select(PostDraft).where(PostDraft.candidate_id == cand_id, PostDraft.platform == platform)
    ).first()


def new_publication(
    session: Session,
    acc: SocialAccount,
    c: Candidate,
    render_id: int,
    *,
    privacy: str,
    title: str | None = None,
    auto: bool = False,
) -> Publication:
    from clipador.services import tiktok

    d = draft_for(session, c.id, acc.platform)
    if acc.platform == "tiktok":
        text = tiktok.caption(d.description if d else (title or ""), list(d.hashtags) if d else [])
        title_ = (title or text)[:100]
        desc, tags = text, []
    else:
        title_ = (title or _title_variant(session, c, d) or "").strip()[:100]
        desc, tags = (d.description if d else ""), (list(d.hashtags) if d else [])
    return Publication(
        candidate_id=c.id,
        render_id=render_id,
        platform=acc.platform,
        account_id=acc.id,
        privacy=privacy,
        title=title_,
        description=desc,
        tags=tags,
        auto=auto,
    )


def _title_variant(session: Session, c: Candidate, d: PostDraft | None) -> str:
    """The same clip on a second/third channel gets another of the AI's titles, so the
    channels don't look like copies of each other."""
    base = d.title if d else ""
    others = session.exec(
        select(Publication).where(
            Publication.candidate_id == c.id,
            Publication.platform == "youtube",
            Publication.status != PublicationStatus.CANCELED,
        )
    ).all()
    if not others:
        return base
    e = session.exec(
        select(CandidateEnrichment).where(CandidateEnrichment.candidate_id == c.id)
    ).first()
    used = {p.title for p in others}
    for t in e.titulos if e and e.titulos else []:
        if t and t not in used and t != base:
            return t
    return base


def taken_times(session: Session, acc_id: int) -> list[datetime]:
    pubs = session.exec(
        select(Publication).where(
            Publication.account_id == acc_id, col(Publication.status).in_(list(ACTIVE_PUB))
        )
    ).all()
    return [aware(p.scheduled_for or p.created_at) for p in pubs]


def _round_robin(cands: list[Candidate], streamer_of: dict[int, int]) -> list[Candidate]:
    """Best first, but alternating streamers: a general profile mixes everyone instead of
    posting 5 clips of whoever had the best day."""
    by: dict[int, list[Candidate]] = {}
    for c in cands:  # already sorted by score
        by.setdefault(streamer_of.get(c.source_id, 0), []).append(c)
    queues = sorted(by.values(), key=lambda q: -_score(q[0]))
    out: list[Candidate] = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def _score(c: Candidate) -> float:
    return c.final_score if c.final_score is not None else c.score


def _candidates_for(
    session: Session, acc: SocialAccount, now: datetime | None = None
) -> list[Candidate]:
    since = aware(acc.autopost_since)
    ready_before = (aware(now) or utcnow()) - timedelta(minutes=max(0, acc.delay_minutes))
    q = (
        select(Candidate)
        .join(Source, Source.id == Candidate.source_id)
        .where(col(Candidate.status).in_([CandidateStatus.APPROVED, CandidateStatus.RENDERED]))
    )
    if acc.streamer_ids:
        q = q.where(col(Source.streamer_id).in_(acc.streamer_ids))
    done = {
        p.candidate_id
        for p in session.exec(select(Publication).where(Publication.account_id == acc.id)).all()
    }
    out = []
    for c in session.exec(q).all():
        reviewed = aware(c.reviewed_at or c.updated_at)
        if c.id in done or (since and reviewed and reviewed < since):
            continue
        if reviewed and reviewed > ready_before:  # the account's delay hasn't passed yet
            continue
        out.append(c)
    out.sort(key=lambda c: -_score(c))
    streamer_of = (
        dict(
            session.exec(
                select(Source.id, Source.streamer_id).where(
                    col(Source.id).in_({c.source_id for c in out})
                )
            ).all()
        )
        if out
        else {}
    )
    return _round_robin(out, streamer_of)


def project_times(session: Session, project_id: str | None) -> list[datetime]:
    """Slots of every YouTube post (any account) on the same Google Cloud project."""
    if not project_id:
        return []
    accs = [a.id for a in accounts(session) if a.project_id == project_id]
    pubs = session.exec(
        select(Publication).where(
            col(Publication.account_id).in_(accs), col(Publication.status).in_(list(ACTIVE_PUB))
        )
    ).all()
    return [aware(p.scheduled_for or p.created_at) for p in pubs]


def project_uploads_today(session: Session, project_id: str | None, now: datetime) -> int:
    """Uploads already made (or on their way) today, in the project's quota day."""
    if not project_id:
        return 0
    qtz = quota_tz()
    day = aware(now).astimezone(qtz).date()
    accs = [a.id for a in accounts(session) if a.project_id == project_id]
    pubs = session.exec(
        select(Publication).where(
            col(Publication.account_id).in_(accs),
            col(Publication.status).in_(
                [PublicationStatus.QUEUED, PublicationStatus.UPLOADING, PublicationStatus.PUBLISHED]
            ),
        )
    ).all()
    return sum(
        1
        for p in pubs
        if p.status != PublicationStatus.PUBLISHED
        or aware(p.updated_at).astimezone(qtz).date() == day
    )


def autopost_tick(
    session: Session, settings: Settings, now: datetime | None = None, tz: tzinfo | None = None
) -> list[str]:
    """Schedule newly approved clips and queue the uploads that are due. Does not commit."""
    from clipador.services import analysis, publishing
    from clipador.worker import queue

    now = aware(now) or utcnow()
    notes: list[str] = []
    for acc in accounts(session, active_only=True):
        if not acc.autopost:
            continue
        taken = taken_times(session, acc.id)
        proj_cap = settings.youtube_uploads_per_project_day if acc.project_id else None
        for c in _candidates_for(session, acc, now):
            r = publishing.final_render(session, settings, c.id)
            if r is None:  # approve -> the final render is (or will be) on its way
                if c.status == CandidateStatus.APPROVED:
                    # video gone etc.: the clip just isn't posted
                    with contextlib.suppress(Exception):
                        analysis.enqueue_render(session, settings, c.source_id, [c.id], "final")
                continue
            pub = new_publication(session, acc, c, r.id, privacy=acc.privacy, auto=True)
            reason = hold_reason(session, c) if acc.hold_on_alerts else None
            if reason:
                pub.status, pub.hold_reason = PublicationStatus.HELD, reason
                notes.append(f"{acc.name}: clip #{c.id} segurado ({reason}).")
            else:
                pub.status = PublicationStatus.SCHEDULED
                pub.scheduled_for = next_slot(
                    now,
                    acc,
                    taken,
                    tz,
                    group_taken=project_times(session, acc.project_id),
                    group_cap=proj_cap,
                )
                taken.append(pub.scheduled_for)
                notes.append(
                    f"{acc.name}: clip #{c.id} agendado para "
                    f"{pub.scheduled_for.astimezone(tz or _local_tz()):%d/%m %H:%M}."
                )
            session.add(pub)
            session.flush()
    # due uploads
    lead = timedelta(hours=max(0.0, settings.youtube_upload_lead_hours))
    due = session.exec(
        select(Publication).where(Publication.status == PublicationStatus.SCHEDULED)
    ).all()
    for p in due:
        acc = session.get(SocialAccount, p.account_id) if p.account_id else None
        if acc is None or not acc.active or acc.paused:
            continue
        when = aware(p.scheduled_for) or now
        ahead = lead if p.platform == "youtube" else timedelta(0)
        if when - ahead > now:
            continue
        if (
            acc.project_id
            and project_uploads_today(session, acc.project_id, now)
            >= settings.youtube_uploads_per_project_day
        ):
            continue  # this project's quota is used up today: tries again after midnight PT
        c = session.get(Candidate, p.candidate_id)
        job = queue.enqueue(
            session,
            "publish",
            c.source_id if c else None,
            {"publication_id": p.id},
            dedupe=False,
            priority=publishing.PUBLISH_PRIORITY,
        )
        p.status, p.job_id, p.updated_at = PublicationStatus.QUEUED, job.id, utcnow()
        session.add(p)
        notes.append(f"{acc.name}: enviando o clip #{p.candidate_id}.")
    return notes


def release(
    session: Session, settings: Settings, pub_id: int, now: datetime | None = None
) -> Publication:
    """'Liberar' a held clip: it gets the next slot of its account. Does not commit."""
    p = session.get(Publication, pub_id)
    if p is None or p.status != PublicationStatus.HELD:
        raise ValidationError("Esta postagem não está segurada.")
    acc = session.get(SocialAccount, p.account_id)
    p.scheduled_for = next_slot(aware(now) or utcnow(), acc, taken_times(session, acc.id))
    p.status, p.hold_reason, p.updated_at = PublicationStatus.SCHEDULED, None, utcnow()
    session.add(p)
    return p


def cancel(session: Session, pub_id: int) -> Publication:
    p = session.get(Publication, pub_id)
    if p is None or p.status not in (
        PublicationStatus.SCHEDULED,
        PublicationStatus.HELD,
        PublicationStatus.FAILED,
    ):
        raise ValidationError("Só dá para cancelar uma postagem agendada, segurada ou que falhou.")
    p.status, p.updated_at = PublicationStatus.CANCELED, utcnow()
    session.add(p)
    return p


def post_now(session: Session, pub_id: int) -> Publication:
    """Move a scheduled/held clip to 'now' (the next tick uploads it). Does not commit."""
    p = session.get(Publication, pub_id)
    if p is None or p.status not in (PublicationStatus.SCHEDULED, PublicationStatus.HELD):
        raise ValidationError("Só dá para antecipar uma postagem agendada ou segurada.")
    p.status, p.hold_reason, p.scheduled_for = PublicationStatus.SCHEDULED, None, utcnow()
    session.add(p)
    return p
