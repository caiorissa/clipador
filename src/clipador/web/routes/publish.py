"""/publicar: connect YouTube (OAuth on this machine), upload approved clips by an explicit
click, and see how the published ones are doing (step 6)."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from sqlmodel import Session, col, select

from clipador.config import Settings
from clipador.errors import ClipadorError
from clipador.models import Candidate, Publication, Source, Streamer
from clipador.services import publishing, youtube
from clipador.web.deps import get_session, get_settings_dep, redirect, render
from clipador.worker import queue

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
_PENDING_AUTH: dict[str, tuple[str, str]] = {}  # state -> (code verifier, redirect uri)


def _ctx(session: Session, settings: Settings, cand_id: int | None) -> dict:
    try:
        youtube.load_client(settings)
        configured, setup_error = True, None
    except ClipadorError as e:
        configured, setup_error = False, e.message
    tok = youtube.load_token(settings) or {}
    pubs = session.exec(select(Publication).order_by(col(Publication.id).desc()).limit(100)).all()
    cands = {c.id: c for c in session.exec(select(Candidate).where(
        col(Candidate.id).in_([p.candidate_id for p in pubs]))).all()}  # fmt: skip
    chosen = session.get(Candidate, cand_id) if cand_id else None
    form = None
    if chosen is not None:
        d = publishing.youtube_draft(session, chosen.id)
        src = session.get(Source, chosen.source_id)
        form = {
            "candidate": chosen,
            "draft": d,
            "source": src,
            "streamer": session.get(Streamer, src.streamer_id) if src else None,
            "render": publishing.final_render(session, settings, chosen.id),
        }
    total_views = sum(p.views or 0 for p in pubs)
    return {"configured": configured, "setup_error": setup_error,
            "connected": youtube.connected(settings), "channel": tok.get("channel"),
            "pubs": pubs, "cands": cands, "form": form, "total_views": total_views,
            "privacy_labels": youtube.PRIVACY_LABELS}  # fmt: skip


@router.get("/publicar")
def publish_page(request: Request, session: SessionDep, settings: SettingsDep,
                 candidato: int | None = None):  # fmt: skip
    return render(request, "publish/index.html", **_ctx(session, settings, candidato))


@router.get("/publicar/youtube/conectar")
def youtube_connect(request: Request, settings: SettingsDep):
    try:
        client = youtube.load_client(settings)
    except ClipadorError as e:
        return redirect("/publicar", e.message)
    verifier, challenge = youtube.new_pkce()
    state = secrets.token_urlsafe(24)
    redirect_uri = str(request.url_for("youtube_callback"))
    _PENDING_AUTH[state] = (verifier, redirect_uri)
    return RedirectResponse(youtube.auth_url(client, redirect_uri, state, challenge), 303)


@router.get("/publicar/youtube/callback", name="youtube_callback")
def youtube_callback(settings: SettingsDep, state: str = "", code: str = "", error: str = ""):
    pending = _PENDING_AUTH.pop(state, None)
    if error or pending is None or not code:
        return redirect(
            "/publicar", f"Conexão com o YouTube cancelada{f' ({error})' if error else ''}."
        )
    try:
        tok = youtube.exchange_code(settings, code, pending[1], pending[0])
    except ClipadorError as e:
        return redirect("/publicar", e.message)
    ch = (tok.get("channel") or {}).get("title")
    return redirect("/publicar", f"YouTube conectado{f': {ch}' if ch else ''}.")


@router.post("/publicar/youtube/desconectar")
def youtube_disconnect(settings: SettingsDep):
    youtube.disconnect(settings)
    return redirect("/publicar", "YouTube desconectado (o token foi apagado deste computador).")


@router.post("/publicar/youtube/numeros")
def youtube_refresh(session: SessionDep):
    queue.enqueue(session, "youtube_metrics", None, {})
    session.commit()
    return redirect("/publicar", "Atualização dos números na fila.")


@router.post("/candidatos/{cand_id}/publicar/youtube")
def publish_candidate(
    request: Request, cand_id: int, session: SessionDep, settings: SettingsDep,
    privacy: Annotated[str, Form()] = "private", publish_at: Annotated[str, Form()] = "",
    title: Annotated[str, Form()] = "", confirm: Annotated[str | None, Form()] = None,
):  # fmt: skip
    if confirm is None:
        return render(request, "publish/index.html", status_code=400,
                      error="Confirme que revisou o clip (direitos, música e crédito).",
                      **_ctx(session, settings, cand_id))  # fmt: skip
    try:
        when = publishing.parse_publish_at(publish_at)
        pub = publishing.request_youtube(session, settings, cand_id, privacy=privacy,
                                         publish_at=when, title=title or None)  # fmt: skip
        session.commit()
    except ClipadorError as e:
        session.rollback()
        return render(request, "publish/index.html", status_code=400, error=e.message,
                      **_ctx(session, settings, cand_id))  # fmt: skip
    return redirect("/publicar", f"Envio de '{pub.title}' na fila (job #{pub.job_id}).")
