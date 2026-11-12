"""/fontes: add a VOD/clip URL (platform auto-detected) and list sources."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session, col, select

from clipador.errors import ValidationError
from clipador.models import Source, Streamer
from clipador.platforms.base import InvalidUrlError
from clipador.platforms.registry import parse_url
from clipador.services.sources import SourceDraft, create_source
from clipador.utils import parse_timecode
from clipador.web.deps import get_session, redirect, render
from clipador.worker import queue

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


def _page_ctx(session: Session) -> dict:
    streamers = session.exec(
        select(Streamer).where(Streamer.active == True).order_by(Streamer.display_name)  # noqa: E712
    ).all()
    return {"streamers": streamers, "sources": _sources(session)}


def _sources(session: Session) -> list[Source]:
    return list(session.exec(select(Source).order_by(col(Source.id).desc()).limit(200)).all())


@router.get("/fontes")
def list_sources(request: Request, session: SessionDep):
    return render(request, "sources/list.html", form={}, **_page_ctx(session))


@router.get("/fontes/tabela")
def sources_table(request: Request, session: SessionDep):
    return render(request, "sources/_table.html", sources=_sources(session))


@router.post("/fontes/detectar")
async def detect_url(request: Request):
    """HTMX: show the detected platform/kind badge while the user types the URL."""
    url = str((await request.form()).get("url", "")).strip()
    parsed, error = None, None
    if url:
        try:
            parsed = parse_url(url)
        except InvalidUrlError as e:
            error = e.message
    return render(request, "sources/_detect.html", parsed=parsed, error=error)


@router.post("/fontes")
async def add_source(request: Request, session: SessionDep):
    form = {k: str(v) for k, v in (await request.form()).items()}
    try:
        try:
            streamer_id = int(form.get("streamer_id", ""))
        except ValueError:
            raise ValidationError("Escolha o streamer dono deste VOD/clip.") from None
        try:
            start = parse_timecode(form.get("start_offset"))
            end = parse_timecode(form.get("end_offset"))
        except ValueError as e:
            raise ValidationError(str(e)) from None
        source, _parsed = create_source(
            session,
            SourceDraft(
                url=form.get("url", ""), streamer_id=streamer_id, start_offset=start, end_offset=end
            ),
        )
        queue.enqueue(session, "ingest", source.id)
        session.commit()
    except ValidationError as e:
        session.rollback()
        return render(
            request,
            "sources/list.html",
            status_code=422,
            form=form,
            error=e.message,
            **_page_ctx(session),
        )
    msg = f"Fonte #{source.id} adicionada; o download foi para a fila."
    if source.warning:
        msg += " Atenção: " + source.warning
    return redirect("/fontes", msg)


@router.post("/fontes/{source_id}/baixar")
def redownload(source_id: int, session: SessionDep):
    source = session.get(Source, source_id)
    if source is None:
        raise HTTPException(status_code=404)
    job = queue.enqueue(session, "ingest", source.id)
    session.commit()
    return redirect("/jobs", f"Job #{job.id} na fila para a fonte #{source.id}.")
