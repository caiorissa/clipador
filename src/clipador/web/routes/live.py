"""/ao-vivo: start/stop live captures, live status (HTMX polling) and the auto-clip alert
counter used by the menu badge and the browser notification (step 5E)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from sqlmodel import Session, col, select

from clipador.config import Settings
from clipador.errors import ClipadorError
from clipador.models import LiveSession, Source, Streamer
from clipador.services import live
from clipador.web.deps import get_session, get_settings_dep, redirect, render

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


def _rows(session: Session, settings: Settings) -> list[dict]:
    sessions = session.exec(select(LiveSession).order_by(col(LiveSession.id).desc()).limit(30))
    rows = []
    for ls in sessions.all():
        src = session.get(Source, ls.source_id)
        st = session.get(Streamer, src.streamer_id) if src else None
        rows.append({"ls": ls, "source": src, "streamer": st,
                     "stats": live.stats(session, settings, ls),
                     "wall": live.wall_seconds(ls)})  # fmt: skip
    return rows


def _streamers(session: Session) -> list[Streamer]:
    return list(session.exec(
        select(Streamer).where(Streamer.active == True).order_by(Streamer.display_name)  # noqa: E712
    ).all())  # fmt: skip


@router.get("/ao-vivo")
def live_page(request: Request, session: SessionDep, settings: SettingsDep):
    return render(request, "live/index.html", rows=_rows(session, settings),
                  streamers=_streamers(session), form={},
                  defaults=live.LiveOptions.defaults(settings))  # fmt: skip


@router.get("/ao-vivo/sessoes")
def live_sessions(request: Request, session: SessionDep, settings: SettingsDep):
    return render(request, "live/_sessions.html", rows=_rows(session, settings))


@router.post("/ao-vivo")
def start_live(
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
    url: Annotated[str, Form()],
    streamer_id: Annotated[str, Form()] = "",
    segment_seconds: Annotated[int, Form()] = 300,
    min_score: Annotated[float, Form()] = 70.0,
    auto_clip: Annotated[str | None, Form()] = None,
    live_from_start: Annotated[str | None, Form()] = None,
):
    opts = live.LiveOptions(segment_seconds, auto_clip is not None, min_score,
                            live_from_start is not None)  # fmt: skip
    try:
        ls = live.create_session(
            session, settings, streamer_id=int(streamer_id) if streamer_id else None,
            url=url.strip(), options=opts,
        )  # fmt: skip
        session.commit()
        live.start_session(request.app.state.engine, settings, ls.id)
    except (ClipadorError, OSError) as e:
        session.rollback()
        return render(request, "live/index.html", status_code=400,
                      rows=_rows(session, settings), streamers=_streamers(session),
                      defaults=opts, error=getattr(e, "message", str(e)),
                      form={"url": url, "streamer_id": streamer_id})  # fmt: skip
    msg = f"Live #{ls.id}: captura iniciada."
    if ls.warning:
        msg += f" {ls.warning}"
    return redirect("/ao-vivo", msg)


@router.post("/ao-vivo/{live_id}/parar")
def stop_live(request: Request, live_id: int, session: SessionDep, settings: SettingsDep):
    try:
        live.request_stop(session, live_id)
        session.commit()
    except ClipadorError as e:
        session.rollback()
        return render(request, "live/_sessions.html", rows=_rows(session, settings),
                      error=e.message)  # fmt: skip
    msg = f"Live #{live_id}: parando (o segmento atual é fechado e analisado)."
    return render(request, "live/_sessions.html", rows=_rows(session, settings), msg=msg)


@router.get("/ao-vivo/alertas.json")
def live_alerts(session: SessionDep):
    """Pending auto-clipped candidates: menu badge + browser notification."""
    pending = live.pending_auto_clips(session)
    active = session.exec(
        select(LiveSession.id).where(col(LiveSession.status).in_(live.ACTIVE_LIVE_STATUSES))
    ).all()
    return {
        "pending": len(pending),
        "latest_id": pending[0].id if pending else None,
        "latest": (pending[0].transcript_snippet or "")[:120] if pending else "",
        "active_lives": len(active),
    }
