"""/ (Início): what needs attention now, and shortcuts to the daily actions."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlmodel import Session, select

from clipador.config import Settings
from clipador.models import Streamer
from clipador.services import dashboard
from clipador.web.deps import get_session, get_settings_dep, render

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


@router.get("/")
def home(request: Request, session: SessionDep, settings: SettingsDep):
    streamers = session.exec(
        select(Streamer).where(Streamer.active == True).order_by(Streamer.display_name)  # noqa: E712
    ).all()
    return render(request, "home/index.html", d=dashboard.home(session, settings),
                  streamers=streamers)  # fmt: skip
