"""/aprovados: approved clips -> render what's missing -> post (now / next slot / date)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from sqlmodel import Session, select

from clipador.config import Settings
from clipador.errors import ClipadorError
from clipador.models import Streamer
from clipador.services import approved, publishing, youtube
from clipador.web.deps import get_session, get_settings_dep, redirect, render

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


def _card(request: Request, session: Session, settings: Settings, cand_id: int, **extra):
    it = next((i for i in approved.items(session, settings) if i.c.id == cand_id), None)
    return render(request, "approved/_item.html", it=it, accounts=approved.account_options(session),
                  privacy_labels=youtube.PRIVACY_LABELS, **extra)  # fmt: skip


@router.get("/aprovados")
def approved_page(request: Request, session: SessionDep, settings: SettingsDep,
                  filtro: str = "todos", streamer: int | None = None):  # fmt: skip
    filtro = filtro if filtro in approved.FILTERS else "todos"
    streamers = session.exec(select(Streamer).order_by(Streamer.display_name)).all()
    return render(
        request, "approved/index.html",
        items=approved.items(session, settings, streamer_id=streamer, state=filtro),
        counts=approved.counts(session, settings), filters=approved.FILTERS, filtro=filtro,
        streamers=streamers, streamer_id=streamer, accounts=approved.account_options(session),
        privacy_labels=youtube.PRIVACY_LABELS,
    )  # fmt: skip


@router.get("/aprovados/{cand_id}")
def approved_card(request: Request, cand_id: int, session: SessionDep, settings: SettingsDep):
    return _card(request, session, settings, cand_id)


@router.post("/aprovados/renderizar-faltando")
def render_all_missing(session: SessionDep, settings: SettingsDep):
    n = approved.render_missing(session, settings)
    session.commit()
    if not n:
        return redirect("/aprovados", "Nenhum clip aprovado sem render.")
    return redirect("/aprovados", f"{n} corte(s) final(is) na fila de render.")


@router.post("/aprovados/{cand_id}/renderizar")
def render_one(request: Request, cand_id: int, session: SessionDep, settings: SettingsDep):
    try:
        approved.render_missing(session, settings, [cand_id])
        session.commit()
    except ClipadorError as e:
        session.rollback()
        return _card(request, session, settings, cand_id, error=e.message)
    return _card(request, session, settings, cand_id)


@router.post("/aprovados/{cand_id}/postar")
def post_one(
    request: Request, cand_id: int, session: SessionDep, settings: SettingsDep,
    account_id: Annotated[int, Form()] = 0, when: Annotated[str, Form()] = "grade",
    privacy: Annotated[str, Form()] = "public", publish_at: Annotated[str, Form()] = "",
    confirm: Annotated[str | None, Form()] = None,
):  # fmt: skip
    if confirm is None:
        return _card(request, session, settings, cand_id,
                     error="Marque que conferiu direitos, música e crédito.")  # fmt: skip
    try:
        at = publishing.parse_publish_at(publish_at) if when == "data" else None
        pub = approved.post(session, settings, cand_id, account_id, when=when, privacy=privacy,
                            at=at)  # fmt: skip
        session.commit()
    except ClipadorError as e:
        session.rollback()
        return _card(request, session, settings, cand_id, error=e.message)
    if pub.scheduled_for:
        msg = f"Agendado para {pub.scheduled_for.astimezone():%d/%m %H:%M}."
    else:
        msg = "Enviando agora (veja em Publicar → Fila)."
    return _card(request, session, settings, cand_id, msg=msg)
