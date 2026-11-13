"""/sistema: environment diagnostics (doctor)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from clipador.config import Settings
from clipador.services import doctor
from clipador.web.deps import get_settings_dep, render

router = APIRouter()
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


@router.get("/sistema")
def system_page(request: Request, settings: SettingsDep):
    return render(request, "system/index.html", settings=settings)


@router.get("/sistema/checagens")
def system_checks(request: Request, settings: SettingsDep):
    # Sync route: FastAPI runs it in a thread pool (doctor spawns ffmpeg/nvidia-smi).
    return render(request, "system/_checks.html", results=doctor.run_all(settings))


@router.get("/sistema/ollama")
def ollama_partial(request: Request, settings: SettingsDep):
    from clipador.llm.client import ollama_status

    return render(request, "system/_ollama.html", st=ollama_status(settings))
