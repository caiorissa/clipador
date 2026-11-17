"""Publishing (step 6): queue the upload of an approved clip's final render, and pull the
numbers back. Uploading is always an explicit click; nothing here runs on its own except the
metrics refresh of videos that were already published."""

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
)
from clipador.services import youtube
from clipador.utils import utcnow

PUBLISH_PRIORITY = 10  # an upload you asked for goes ahead of background work


def final_render(session: Session, settings: Settings, cand_id: int) -> Render | None:
    """Most recent finished final render whose file still exists."""
    rows = session.exec(
        select(Render)
        .where(Render.candidate_id == cand_id, Render.kind == RenderKind.FINAL,
               Render.status == RenderStatus.DONE)
        .order_by(col(Render.id).desc())
    ).all()  # fmt: skip
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


def request_youtube(
    session: Session, settings: Settings, cand_id: int, *, privacy: str,
    publish_at: datetime | None = None, title: str | None = None,
) -> Publication:  # fmt: skip
    """Validate and queue an upload. Does not commit."""
    c = session.get(Candidate, cand_id)
    if c is None:
        raise ValidationError("Candidato não encontrado.")
    if c.status not in (CandidateStatus.APPROVED, CandidateStatus.RENDERED):
        raise ValidationError("Só clips aprovados podem ser publicados.")
    if privacy not in youtube.PRIVACY:
        raise ValidationError("Visibilidade inválida.")
    youtube.load_client(settings)  # configured?
    if not youtube.connected(settings):
        raise ValidationError("Conecte sua conta do YouTube em Publicar antes.")
    r = final_render(session, settings, cand_id)
    if r is None:
        raise ValidationError("Renderize o corte final antes de publicar.")
    busy = session.exec(
        select(Publication).where(
            Publication.candidate_id == cand_id, Publication.platform == "youtube",
            col(Publication.status).in_([PublicationStatus.QUEUED, PublicationStatus.UPLOADING,
                                         PublicationStatus.PUBLISHED]),
        )
    ).first()  # fmt: skip
    if busy is not None:
        raise ValidationError("Este clip já foi (ou está sendo) enviado ao YouTube.")
    d = youtube_draft(session, cand_id)
    pub = Publication(
        candidate_id=cand_id, render_id=r.id, platform="youtube", privacy=privacy,
        publish_at=publish_at, title=(title or (d.title if d else "") or "").strip()[:100],
        description=d.description if d else "", tags=list(d.hashtags) if d else [],
    )  # fmt: skip
    session.add(pub)
    session.flush()
    from clipador.worker import queue

    job = queue.enqueue(session, "publish_youtube", c.source_id, {"publication_id": pub.id},
                        dedupe=False, priority=PUBLISH_PRIORITY)  # fmt: skip
    pub.job_id = job.id
    session.add(pub)
    return pub


def refresh_metrics(session: Session, settings: Settings, yt=None) -> int:
    """Update views/likes/comments of everything published. Returns how many were updated."""
    pubs = session.exec(
        select(Publication).where(
            Publication.platform == "youtube", Publication.status == PublicationStatus.PUBLISHED,
            Publication.external_id != None,  # noqa: E711
        )
    ).all()  # fmt: skip
    if not pubs:
        return 0
    yt = yt or youtube.YouTube(settings)
    stats = yt.stats([p.external_id for p in pubs])
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
    return len(stats)


def performance_labels(session: Session, min_views_total: int = 200) -> dict[int, str]:
    """{candidate_id: 'bom' | 'ruim'} from real views, relative to the user's own channel:
    top third of the published clips = bom, bottom third = ruim (only public clips older
    than 2 days, and only when there is enough data). Used by `clipador calibrate`."""
    pubs = [p for p in session.exec(select(Publication).where(
        Publication.status == PublicationStatus.PUBLISHED, Publication.views != None)).all()  # noqa: E711
        if p.privacy == "public" and (utcnow() - _aware(p.created_at)).days >= 2]  # fmt: skip
    if len(pubs) < 6 or sum(p.views or 0 for p in pubs) < min_views_total:
        return {}
    ranked = sorted(pubs, key=lambda p: p.views or 0)
    third = len(ranked) // 3
    labels = {p.candidate_id: "ruim" for p in ranked[:third]}
    labels.update({p.candidate_id: "bom" for p in ranked[-third:]})
    return labels


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
