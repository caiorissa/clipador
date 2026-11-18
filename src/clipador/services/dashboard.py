"""Numbers for the sidebar counters and the home screen ("what needs me now?")."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import Engine, func
from sqlmodel import Session, col, select

from clipador.models import (
    Job,
    JobStatus,
    LiveSession,
    Publication,
    PublicationStatus,
    SocialAccount,
    Source,
    Streamer,
)
from clipador.utils import utcnow

ACTIVE_LIVE = ("watching", "recording")


def _count(session: Session, q) -> int:
    return int(session.exec(q).one() or 0)


def nav_counts(engine: Engine, settings=None) -> dict[str, int]:
    """Small badges in the sidebar (+ "worker": 1/0). Never raises (the menu must render)."""
    try:
        from clipador.services import review
        from clipador.worker.runner import worker_alive

        with Session(engine) as s:
            prog = review.progress(s, review.ReviewFilters())
            day_ago = utcnow() - timedelta(hours=24)
            return {
                "review": prog["suggested"],
                "live": _count(s, select(func.count()).select_from(LiveSession).where(
                    col(LiveSession.status).in_(ACTIVE_LIVE))),
                "publish_held": _count(s, select(func.count()).select_from(Publication).where(
                    Publication.status == PublicationStatus.HELD)),
                "publish_queue": _count(s, select(func.count()).select_from(Publication).where(
                    col(Publication.status).in_([PublicationStatus.SCHEDULED,
                                                 PublicationStatus.QUEUED,
                                                 PublicationStatus.UPLOADING]))),
                "jobs_running": _count(s, select(func.count()).select_from(Job).where(
                    col(Job.status).in_([JobStatus.RUNNING, JobStatus.QUEUED]))),
                "jobs_failed": _count(s, select(func.count()).select_from(Job).where(
                    Job.status == JobStatus.FAILED, Job.updated_at >= day_ago)),
                "jobs_queued": _count(s, select(func.count()).select_from(Job).where(
                    Job.status == JobStatus.QUEUED)),
                "worker": int(worker_alive(settings)) if settings is not None else 1,
            }  # fmt: skip
    except Exception:
        return {}


def home(session: Session, settings) -> dict[str, Any]:
    from clipador.services import live as live_service
    from clipador.services import review

    prog = review.progress(session, review.ReviewFilters())
    now = utcnow()
    upcoming = session.exec(
        select(Publication)
        .where(Publication.status == PublicationStatus.SCHEDULED)
        .order_by(col(Publication.scheduled_for))
        .limit(6)
    ).all()
    held = session.exec(
        select(Publication).where(Publication.status == PublicationStatus.HELD)
    ).all()
    published_today = _count(
        session,
        select(func.count())
        .select_from(Publication)
        .where(
            Publication.status == PublicationStatus.PUBLISHED,
            Publication.updated_at >= now - timedelta(hours=24),
        ),
    )
    lives = session.exec(select(LiveSession).where(col(LiveSession.status).in_(ACTIVE_LIVE))).all()
    live_rows = []
    for ls in lives:
        src = session.get(Source, ls.source_id)
        live_rows.append(
            {"ls": ls, "source": src, "stats": live_service.stats(session, settings, ls)}
        )
    failed = session.exec(
        select(Job)
        .where(Job.status == JobStatus.FAILED, Job.updated_at >= now - timedelta(hours=24))
        .order_by(col(Job.id).desc())
        .limit(5)
    ).all()
    running = session.exec(
        select(Job).where(Job.status == JobStatus.RUNNING).order_by(col(Job.id).desc()).limit(5)
    ).all()
    sources = session.exec(select(Source).order_by(col(Source.id).desc()).limit(5)).all()
    accounts = session.exec(select(SocialAccount).where(SocialAccount.active == True)).all()  # noqa: E712
    n_streamers = _count(session, select(func.count()).select_from(Streamer))
    authorized = _count(
        session,
        select(func.count()).select_from(Streamer).where(Streamer.permission_confirmed == True),  # noqa: E712
    )
    return {
        "review": prog,
        "upcoming": upcoming,
        "held": held,
        "published_today": published_today,
        "lives": live_rows,
        "failed": failed,
        "running": running,
        "sources": sources,
        "accounts": accounts,
        "autopost_on": sum(1 for a in accounts if a.autopost and not a.paused),
        "setup": {
            "streamers": n_streamers,
            "authorized": authorized,
            "sources": _count(session, select(func.count()).select_from(Source)),
            "accounts": len(accounts),
        },
    }
