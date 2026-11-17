"""Publishing (step 6): queue the upload of an approved clip's final render to one of the
connected accounts, and pull the numbers back. Only approved clips are ever posted: by hand
here, or by the account's auto-post grid (services/accounts.py)."""

from __future__ import annotations

from datetime import UTC, datetime

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
)
from clipador.services import youtube
from clipador.utils import utcnow

PUBLISH_PRIORITY = 10  # an upload you asked for goes ahead of background work


def final_render(session: Session, settings: Settings, cand_id: int) -> Render | None:
    """Most recent finished final render whose file still exists."""
    rows = session.exec(
        select(Render)
        .where(
            Render.candidate_id == cand_id,
            Render.kind == RenderKind.FINAL,
            Render.status == RenderStatus.DONE,
        )
        .order_by(col(Render.id).desc())
    ).all()
    return next((r for r in rows if r.path and settings.abs_path(r.path).is_file()), None)


def youtube_draft(session: Session, cand_id: int) -> PostDraft | None:
    return session.exec(
        select(PostDraft).where(PostDraft.candidate_id == cand_id, PostDraft.platform == "youtube")
    ).first()


def parse_publish_at(value: str | None) -> datetime | None:
    """'2026-11-20T18:30' (local time from <input type=datetime-local>) -> aware UTC."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as e:
        raise ValidationError("Data de agendamento inválida.") from e
    if dt.tzinfo is None:
        dt = dt.astimezone()  # the user's local time
    dt = dt.astimezone(UTC)
    if dt <= utcnow():
        raise ValidationError("O agendamento precisa ser no futuro.")
    return dt


def request_publication(
    session: Session,
    settings: Settings,
    cand_id: int,
    account_id: int,
    *,
    privacy: str,
    publish_at: datetime | None = None,
    title: str | None = None,
) -> Publication:
    """'Publicar' clicked by hand on an approved clip. YouTube with a date: uploaded now,
    private until then (publishAt). TikTok with a date: waits for it in the schedule.
    Does not commit."""
    from clipador.services import accounts

    c = session.get(Candidate, cand_id)
    if c is None:
        raise ValidationError("Candidato não encontrado.")
    if c.status not in (CandidateStatus.APPROVED, CandidateStatus.RENDERED):
        raise ValidationError("Só clips aprovados podem ser publicados.")
    if privacy not in youtube.PRIVACY:
        raise ValidationError("Visibilidade inválida.")
    acc = session.get(SocialAccount, account_id) if account_id else None
    if acc is None or not acc.active:
        raise ValidationError("Escolha uma conta conectada (Publicar → Contas).")
    r = final_render(session, settings, cand_id)
    if r is None:
        raise ValidationError("Renderize o corte final antes de publicar.")
    busy = session.exec(
        select(Publication).where(
            Publication.candidate_id == cand_id,
            Publication.account_id == acc.id,
            col(Publication.status).in_(list(accounts.ACTIVE_PUB)),
        )
    ).first()
    if busy is not None:
        raise ValidationError(f"Este clip já foi (ou vai ser) postado em {acc.name}.")
    pub = accounts.new_publication(session, acc, c, r.id, privacy=privacy, title=title)
    if acc.platform == "tiktok" and publish_at is not None:
        pub.status, pub.scheduled_for = PublicationStatus.SCHEDULED, publish_at
        session.add(pub)
        session.flush()
        return pub
    pub.publish_at = publish_at
    session.add(pub)
    session.flush()
    from clipador.worker import queue

    job = queue.enqueue(
        session,
        "publish",
        c.source_id,
        {"publication_id": pub.id},
        dedupe=False,
        priority=PUBLISH_PRIORITY,
    )
    pub.job_id = job.id
    session.add(pub)
    return pub


def refresh_metrics(session: Session, settings: Settings, engine=None, clients=None) -> int:
    """Update views/likes/comments of everything published, per account. Returns how many
    videos got numbers. `clients` = {account_id: client} (tests)."""
    from clipador.services import accounts

    engine = engine or session.get_bind()
    n = 0
    for acc in accounts.accounts(session, active_only=True):
        pubs = session.exec(
            select(Publication).where(
                Publication.account_id == acc.id,
                Publication.status == PublicationStatus.PUBLISHED,
                Publication.external_id != None,  # noqa: E711
            )
        ).all()
        if not pubs:
            continue
        client = (clients or {}).get(acc.id) or accounts.client_for(engine, settings, acc)
        stats = client.stats([p.external_id for p in pubs])
        now = utcnow()
        for p in pubs:
            st = stats.get(p.external_id)
            if st is None:
                continue
            p.views, p.likes, p.comments = st["view"], st["like"], st["comment"]
            if st.get("privacy") in youtube.PRIVACY:
                p.privacy = st["privacy"]
            p.metrics_at = p.updated_at = now
            session.add(p)
            n += 1
    return n


def performance_labels(session: Session, min_views_total: int = 200) -> dict[int, str]:
    """{candidate_id: 'bom' | 'ruim'} from real views, relative to the user's own channel:
    top third of the published clips = bom, bottom third = ruim (only public clips older
    than 2 days, and only when there is enough data). Used by `clipador calibrate`."""
    pubs = [
        p
        for p in session.exec(
            select(Publication).where(
                Publication.status == PublicationStatus.PUBLISHED,
                Publication.views != None,  # noqa: E711
            )
        ).all()
        if p.privacy == "public" and (utcnow() - _aware(p.created_at)).days >= 2
    ]
    if len(pubs) < 6 or sum(p.views or 0 for p in pubs) < min_views_total:
        return {}
    ranked = sorted(pubs, key=lambda p: p.views or 0)
    third = len(ranked) // 3
    labels = {p.candidate_id: "ruim" for p in ranked[:third]}
    labels.update({p.candidate_id: "bom" for p in ranked[-third:]})
    return labels


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
