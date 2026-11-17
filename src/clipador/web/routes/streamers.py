"""/streamers: list, create, edit, (de)activate, channels."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlmodel import Session, col, select

from clipador.errors import ClipadorError, ValidationError
from clipador.models import Layout, Platform, Streamer, StreamerChannel
from clipador.render.styles import load_styles
from clipador.services import facecam, ffmpeg
from clipador.services.captions import add_glossary_entry, remove_glossary_entry
from clipador.services.sources import add_channel, suggest_credit, touch
from clipador.web.deps import get_session, redirect, render

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]

FACECAM_FIELDS = ("fc_x", "fc_y", "fc_w", "fc_h")  # normalized 0..1


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

    rect_raw = [form.get(f, "").strip().replace(",", ".") for f in FACECAM_FIELDS]
    box: list[float] | None = None
    if any(rect_raw):
        if not all(rect_raw):
            raise ValidationError("Facecam: desenhe o retângulo (ou preencha os 4 valores).")
        try:
            box = [round(float(v), 4) for v in rect_raw]
        except ValueError:
            raise ValidationError("Facecam: valores entre 0 e 1.") from None
        x, y, w, h = box
        if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValidationError("Facecam: valores entre 0 e 1 (fração da largura/altura).")
        if x + w > 1.0001 or y + h > 1.0001 or w < 0.03 or h < 0.03:
            raise ValidationError(
                "Facecam: o retângulo precisa caber no quadro e não ser minúsculo."
            )
    style = form.get("default_caption_style", "").strip()
    if style and style not in load_styles():
        raise ValidationError("Estilo de legenda inválido.")

    streamer.display_name = name
    streamer.permission_confirmed = confirmed
    streamer.permission_date = perm_date
    streamer.permission_note = form.get("permission_note", "").strip()
    streamer.credit_text = form.get("credit_text", "").strip()
    streamer.default_layout = layout
    old_box = [round(float(v), 4) for v in streamer.facecam_box] if streamer.facecam_box else None
    if box != old_box:  # drawn/typed/cleared by hand: never overwritten by the detector again
        streamer.facecam_auto = False
        streamer.facecam_confidence = None
    streamer.facecam_box = box
    streamer.default_caption_style = style or None
    streamer.show_credit = form.get("show_credit") == "on"
    streamer.show_opening_text = form.get("show_opening_text") == "on"
    streamer.auto_watch = form.get("auto_watch") == "on"
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


@router.get("/streamers/{streamer_id}")
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


@router.post("/streamers/{streamer_id}/glossario")
async def save_glossary_entry(request: Request, streamer_id: int, session: SessionDep):
    """Add an entry, or edit one (`antigo` = the entry's current 'wrong' word)."""
    streamer = _get_streamer(session, streamer_id)
    form = await request.form()
    old = str(form.get("antigo", "")).strip() or None
    try:
        key = add_glossary_entry(
            streamer, str(form.get("errado", "")), str(form.get("certo", "")), old
        )
        session.add(streamer)
        session.commit()
    except ValidationError as e:
        session.rollback()
        return redirect(f"/streamers/{streamer_id}", e.message)
    verb = "atualizada" if old else "adicionada"
    return redirect(f"/streamers/{streamer_id}", f"Entrada “{key}” {verb} no glossário.")


@router.post("/streamers/{streamer_id}/glossario/remover")
async def remove_glossary(request: Request, streamer_id: int, session: SessionDep):
    streamer = _get_streamer(session, streamer_id)
    wrong = str((await request.form()).get("errado", ""))
    removed = remove_glossary_entry(streamer, wrong)
    session.add(streamer)
    session.commit()
    msg = f"Entrada “{wrong}” removida." if removed else "Entrada não encontrada."
    return redirect(f"/streamers/{streamer_id}", msg)


@router.post("/streamers/{streamer_id}/facecam/detectar")
async def detect_streamer_facecam(request: Request, streamer_id: int, session: SessionDep):
    """'Detectar de novo': run the detector on the latest downloaded source. A hand-marked box
    is only replaced when the user confirmed it (`sobrescrever=1`)."""
    settings = request.app.state.settings
    _get_streamer(session, streamer_id)
    overwrite = str((await request.form()).get("sobrescrever", "")) == "1"
    src = facecam.latest_ready_source(session, streamer_id, settings)
    url = f"/streamers/{streamer_id}/editar"
    if src is None:
        return redirect(url, "Baixe uma fonte deste streamer antes de detectar a facecam.")
    try:
        out = facecam.run_detection(session, settings, src, overwrite_manual=overwrite)
    except ClipadorError as e:
        session.rollback()
        return redirect(url, e.message)
    return redirect(url, out.message)


@router.get("/streamers/{streamer_id}/frame.jpg")
def streamer_frame(request: Request, streamer_id: int, session: SessionDep):
    """A frame from this streamer's latest downloaded source, for drawing the facecam box."""
    settings = request.app.state.settings
    _get_streamer(session, streamer_id)
    src = facecam.latest_ready_source(session, streamer_id, settings)
    if src is None or not src.file_path:
        raise HTTPException(status_code=404, detail="sem fonte baixada")
    out = settings.tmp_dir / f"streamer_{streamer_id}_src{src.id}.jpg"
    if not out.is_file():
        at = (src.duration_seconds or 60) * 0.3
        ffmpeg.make_thumbnail(
            settings.abs_path(src.file_path), out, at, settings.ffmpeg_path, width=960
        )
    return FileResponse(out, media_type="image/jpeg")
