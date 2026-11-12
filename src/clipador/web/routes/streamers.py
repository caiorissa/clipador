"""/streamers: list, create, edit, (de)activate, channels."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session, col, select

from clipador.errors import ValidationError
from clipador.models import Layout, Platform, Streamer, StreamerChannel
from clipador.services.sources import add_channel, suggest_credit, touch
from clipador.web.deps import get_session, redirect, render

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]

FACECAM_FIELDS = ("facecam_x", "facecam_y", "facecam_w", "facecam_h")


def _get_streamer(session: Session, streamer_id: int) -> Streamer:
    streamer = session.get(Streamer, streamer_id)
    if streamer is None:
        raise HTTPException(status_code=404, detail="Streamer não encontrado")
    return streamer


def _apply_form(streamer: Streamer, form: dict[str, str]) -> None:
    """Copy validated form values onto the streamer. Raises ValidationError (PT-BR)."""
    name = form.get("display_name", "").strip()
    if not name:
        raise ValidationError("Informe o nome do streamer.")
    confirmed = form.get("permission_confirmed") == "on"
    raw_date = form.get("permission_date", "").strip()
    perm_date = None
    if raw_date:
        try:
            perm_date = date.fromisoformat(raw_date)
        except ValueError:
            raise ValidationError("Data da autorização inválida.") from None
    if confirmed and perm_date is None:
        raise ValidationError(
            "Informe a data em que o streamer autorizou os clipes (é o seu registro da permissão)."
        )
    if perm_date and perm_date > date.today():
        raise ValidationError("A data da autorização não pode estar no futuro.")

    layout = form.get("default_layout", Layout.BLUR)
    if layout not in tuple(Layout):
        raise ValidationError("Layout inválido.")

    rect_raw = [form.get(f, "").strip() for f in FACECAM_FIELDS]
    rect: list[int | None] = [None] * 4
    if any(rect_raw):
        if not all(rect_raw):
            raise ValidationError(
                "Preencha os 4 valores da facecam (x, y, largura, altura) ou nenhum."
            )
        try:
            rect = [int(v) for v in rect_raw]
        except ValueError:
            raise ValidationError(
                "Os valores da facecam devem ser números inteiros (pixels)."
            ) from None
        if rect[2] <= 0 or rect[3] <= 0 or rect[0] < 0 or rect[1] < 0:
            raise ValidationError("Facecam: x/y não podem ser negativos e largura/altura > 0.")

    streamer.display_name = name
    streamer.permission_confirmed = confirmed
    streamer.permission_date = perm_date
    streamer.permission_note = form.get("permission_note", "").strip()
    streamer.credit_text = form.get("credit_text", "").strip()
    streamer.default_layout = layout
    streamer.facecam_x, streamer.facecam_y, streamer.facecam_w, streamer.facecam_h = rect
    touch(streamer)


@router.get("/streamers")
def list_streamers(request: Request, session: SessionDep):
    streamers = session.exec(
        select(Streamer).order_by(col(Streamer.active).desc(), Streamer.display_name)
    ).all()
    return render(request, "streamers/list.html", streamers=streamers)


@router.get("/streamers/novo")
def new_streamer(request: Request):
    return render(request, "streamers/form.html", streamer=None, form={}, today=date.today())


@router.post("/streamers")
async def create_streamer(request: Request, session: SessionDep):
    form = {k: str(v) for k, v in (await request.form()).items()}
    streamer = Streamer(display_name="")
    try:
        _apply_form(streamer, form)
        session.add(streamer)
        session.flush()
        for p in Platform:
            handle = form.get(f"channel_{p.value}", "").strip()
            if handle:
                add_channel(session, streamer, p, handle)
        if not streamer.credit_text:
            streamer.credit_text = suggest_credit(streamer)
        session.commit()
    except ValidationError as e:
        session.rollback()
        return render(
            request,
            "streamers/form.html",
            status_code=422,
            streamer=None,
            form=form,
            error=e.message,
            today=date.today(),
        )
    return redirect(
        f"/streamers/{streamer.id}/editar", f"Streamer {streamer.display_name} cadastrado."
    )


@router.get("/streamers/{streamer_id}/editar")
def edit_streamer(request: Request, streamer_id: int, session: SessionDep):
    streamer = _get_streamer(session, streamer_id)
    return render(
        request,
        "streamers/form.html",
        streamer=streamer,
        form=None,
        today=date.today(),
        credit_suggestion=suggest_credit(streamer),
    )


@router.post("/streamers/{streamer_id}")
async def update_streamer(request: Request, streamer_id: int, session: SessionDep):
    streamer = _get_streamer(session, streamer_id)
    form = {k: str(v) for k, v in (await request.form()).items()}
    try:
        _apply_form(streamer, form)
        if not streamer.credit_text:
            streamer.credit_text = suggest_credit(streamer)
        session.add(streamer)
        session.commit()
    except ValidationError as e:
        session.rollback()
        return render(
            request,
            "streamers/form.html",
            status_code=422,
            streamer=_get_streamer(session, streamer_id),
            form=form,
            error=e.message,
            today=date.today(),
            credit_suggestion=suggest_credit(streamer),
        )
    return redirect(f"/streamers/{streamer_id}/editar", "Alterações salvas.")


@router.post("/streamers/{streamer_id}/ativo")
def toggle_active(streamer_id: int, session: SessionDep):
    streamer = _get_streamer(session, streamer_id)
    streamer.active = not streamer.active
    touch(streamer)
    session.add(streamer)
    session.commit()
    state = "reativado" if streamer.active else "desativado"
    return redirect("/streamers", f"{streamer.display_name} {state}.")


@router.post("/streamers/{streamer_id}/canais")
async def add_streamer_channel(request: Request, streamer_id: int, session: SessionDep):
    streamer = _get_streamer(session, streamer_id)
    form = await request.form()
    try:
        platform = str(form.get("platform", ""))
        if platform not in tuple(Platform):
            raise ValidationError("Escolha a plataforma do canal.")
        ch = add_channel(session, streamer, platform, str(form.get("channel_handle", "")))
        session.commit()
    except ValidationError as e:
        session.rollback()
        return redirect(f"/streamers/{streamer_id}/editar", e.message)
    return redirect(f"/streamers/{streamer_id}/editar", f"Canal {ch.channel_handle} adicionado.")


@router.post("/streamers/{streamer_id}/canais/{channel_id}/remover")
def remove_channel(streamer_id: int, channel_id: int, session: SessionDep):
    ch = session.get(StreamerChannel, channel_id)
    if ch is None or ch.streamer_id != streamer_id:
        raise HTTPException(status_code=404)
    session.delete(ch)
    session.commit()
    return redirect(f"/streamers/{streamer_id}/editar", f"Canal {ch.channel_handle} removido.")
