"""Shared web helpers: DB session dependency, templates and PT-BR labels."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session

from clipador.config import Settings
from clipador.models import PLATFORM_LABELS, JobStatus, Platform, SourceKind, SourceStatus
from clipador.utils import format_duration, format_hms

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

SOURCE_STATUS_LABELS = {
    SourceStatus.PENDING: "Na fila",
    SourceStatus.DOWNLOADING: "Baixando",
    SourceStatus.READY: "Pronto",
    SourceStatus.FAILED: "Falhou",
    SourceStatus.LIVE: "Ao vivo",
}
JOB_STATUS_LABELS = {
    JobStatus.QUEUED: "Na fila",
    JobStatus.RUNNING: "Rodando",
    JobStatus.DONE: "Concluído",
    JobStatus.FAILED: "Falhou",
    JobStatus.CANCELED: "Cancelado",
}
JOB_TYPE_LABELS = {
    "ingest": "Download",
    "transcribe": "Transcrição",
    "detect": "Detecção",
    "render": "Renderização",
    "extract_audio": "Extrair áudio",
    "fetch_chat": "Chat",
    "enrich": "IA (Ollama)",
    "live_analyze": "Live: análise do segmento",
    "live_finalize": "Live: montar o vídeo",
    "compile_long": "Vídeo longo",
    "publish_youtube": "Publicar no YouTube",
    "publish": "Publicar",
    "youtube_metrics": "YouTube: números",
}
KIND_LABELS = {SourceKind.VOD: "VOD", SourceKind.CLIP: "Clip", SourceKind.LIVE: "Live"}


def _dt(value: datetime | None) -> str:
    if value is None:
        return "—"
    return value.astimezone().strftime("%d/%m/%Y %H:%M")


def _platform_label(value: str) -> str:
    try:
        return PLATFORM_LABELS[Platform(value)]
    except ValueError:
        return value


templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.filters["duration"] = format_duration
templates.env.filters["dt"] = _dt
templates.env.filters["hms"] = format_hms
templates.env.filters["platform_label"] = _platform_label
from clipador.render.filters import LAYOUT_LABELS  # noqa: E402
from clipador.render.styles import load_styles  # noqa: E402

templates.env.globals.update(
    LAYOUT_LABELS=LAYOUT_LABELS,
    caption_styles=load_styles,
    SOURCE_STATUS_LABELS=SOURCE_STATUS_LABELS,
    JOB_STATUS_LABELS=JOB_STATUS_LABELS,
    JOB_TYPE_LABELS=JOB_TYPE_LABELS,
    KIND_LABELS=KIND_LABELS,
    PLATFORMS=list(Platform),
    PLATFORM_LABELS=PLATFORM_LABELS,
)


def get_session(request: Request) -> Iterator[Session]:
    with Session(request.app.state.engine, expire_on_commit=False) as session:
        yield session


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def redirect(url: str, msg: str | None = None) -> RedirectResponse:
    if msg:
        url += ("&" if "?" in url else "?") + "msg=" + quote(msg)
    return RedirectResponse(url, status_code=303)


def render(request: Request, name: str, status_code: int = 200, **ctx):
    ctx.setdefault("msg", request.query_params.get("msg"))
    return templates.TemplateResponse(request, name, ctx, status_code=status_code)
