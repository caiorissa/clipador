"""/fontes/{id}: source detail, analysis stages, timeline, candidates, tuning, video."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from sqlmodel import Session, select

from clipador.config import Settings
from clipador.detection.params import PRESET_LABELS, PRESETS, SIGNALS, DetectParams
from clipador.errors import ValidationError
from clipador.models import (
    AnalysisRun,
    Candidate,
    CandidateFeedback,
    CandidateStatus,
    Source,
    Verdict,
)
from clipador.services import analysis
from clipador.utils import utcnow
from clipador.web.deps import get_session, get_settings_dep, redirect, render

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]

SIGNAL_LABELS = {"audio": "Áudio", "chat": "Chat", "speech": "Fala", "keywords": "Palavras"}


def _source(session: Session, source_id: int) -> Source:
    source = session.get(Source, source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Fonte não encontrada")
    return source


def _selected_run(session: Session, source_id: int, run_id: int | None) -> AnalysisRun | None:
    runs = analysis.runs_for(session, source_id)
    if run_id is not None:
        return next((r for r in runs if r.id == run_id), None)
    return runs[0] if runs else None


def _form_params(run: AnalysisRun | None, default_preset: str) -> DetectParams:
    if run is None:
        return DetectParams.from_preset(default_preset)
    data = {k: v for k, v in run.params.items() if k != "effective_weights"}
    return DetectParams.model_validate(data)


@router.get("/fontes/{source_id}")
def source_detail(
    request: Request,
    source_id: int,
    session: SessionDep,
    settings: SettingsDep,
    run: int | None = None,
):
    source = _source(session, source_id)
    runs = analysis.runs_for(session, source_id)
    selected = _selected_run(session, source_id, run)
    cands = analysis.candidates_for(session, selected.id) if selected else []
    params = _form_params(selected, settings.detect_preset)
    return render(
        request,
        "analysis/detail.html",
        source=source,
        runs=runs,
        run=selected,
        pinned_run=run is not None,
        candidates=cands,
        stages=analysis.stage_overview(session, source_id),
        latest_run_id=runs[0].id if runs else None,
        params=params,
        presets=PRESET_LABELS,
        preset_weights=PRESETS,
        signals=SIGNALS,
        signal_labels=SIGNAL_LABELS,
    )


@router.get("/fontes/{source_id}/estagios")
def stages_partial(request: Request, source_id: int, session: SessionDep):
    _source(session, source_id)
    runs = analysis.runs_for(session, source_id)
    return render(
        request,
        "analysis/_stages.html",
        source_id=source_id,
        stages=analysis.stage_overview(session, source_id),
        latest_run_id=runs[0].id if runs else None,
    )


@router.post("/fontes/{source_id}/analisar")
def start_analysis(source_id: int, session: SessionDep):
    try:
        job = analysis.start_analysis(session, source_id)
        session.commit()
    except ValidationError as e:
        session.rollback()
        return redirect(f"/fontes/{source_id}", e.message)
    return redirect(f"/fontes/{source_id}", f"Análise na fila (job #{job.id}).")


@router.post("/fontes/{source_id}/reanalisar")
async def reanalyze(request: Request, source_id: int, session: SessionDep):
    form = await request.form()
    _source(session, source_id)
    try:
        preset = str(form.get("preset", "equilibrado"))
        params = DetectParams.from_preset(preset)
        c = params.candidates

        def num(name: str, default: float, lo: float, hi: float) -> float:
            raw = str(form.get(name, "")).strip().replace(",", ".")
            if not raw:
                return default
            try:
                v = float(raw)
            except ValueError:
                raise ValidationError(f"Valor inválido em '{name}'.") from None
            return min(hi, max(lo, v))

        if form.get("custom_weights") == "on":
            c.weights = {k: num(f"w_{k}", c.weights[k], 0, 5) for k in SIGNALS}
        c.context_before_s = num("context_before_s", c.context_before_s, 0, 60)
        c.tail_after_s = num("tail_after_s", c.tail_after_s, 0, 60)
        c.min_duration_s = num("min_duration_s", c.min_duration_s, 5, 180)
        c.max_duration_s = num("max_duration_s", c.max_duration_s, 5, 180)
        c.num_candidates = int(num("num_candidates", c.num_candidates, 1, 100))
        if c.max_duration_s < c.min_duration_s:
            raise ValidationError("A duração máxima precisa ser maior que a mínima.")
        job = analysis.reanalyze(session, source_id, params)
        session.commit()
    except ValidationError as e:
        session.rollback()
        return redirect(f"/fontes/{source_id}", e.message)
    return redirect(
        f"/fontes/{source_id}", f"Reanalisando (job #{job.id})… a lista atualiza sozinha."
    )


@router.get("/fontes/{source_id}/timeline.json")
def timeline_json(
    source_id: int, session: SessionDep, settings: SettingsDep, run: int | None = None
):
    _source(session, source_id)
    selected = _selected_run(session, source_id, run)
    if selected is None:
        return JSONResponse({"t": []})
    feats = analysis.load_features(session.get_bind(), settings, source_id)
    if feats is None:
        return JSONResponse({"t": []})
    data = analysis.timeline(feats, selected)
    data["candidates"] = [
        {"id": c.id, "rank": c.rank, "start": c.eff_start, "end": c.eff_end, "peak": c.peak_t,
         "score": c.score, "status": c.status}
        for c, _fb in analysis.candidates_for(session, selected.id)
    ]  # fmt: skip
    return JSONResponse(data)


@router.get("/fontes/{source_id}/video")
def source_video(source_id: int, session: SessionDep, settings: SettingsDep):
    """Serves the downloaded file; Starlette's FileResponse answers HTTP Range (seek works)."""
    source = _source(session, source_id)
    if not source.file_path:
        raise HTTPException(status_code=404)
    path = settings.abs_path(source.file_path)
    if not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="video/mp4" if path.suffix == ".mp4" else None)


# --- candidate actions (HTMX, return the updated card) --------------------------------------


def _candidate(session: Session, candidate_id: int) -> Candidate:
    c = session.get(Candidate, candidate_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Candidato não encontrado")
    return c


def _card(request: Request, session: Session, c: Candidate, error: str | None = None):
    fb = session.exec(
        select(CandidateFeedback).where(CandidateFeedback.candidate_id == c.id)
    ).first()
    return render(
        request,
        "analysis/_candidate.html",
        c=c,
        fb=fb,
        source_id=c.source_id,
        signals=SIGNALS,
        signal_labels=SIGNAL_LABELS,
        card_error=error,
    )


@router.post("/candidatos/{candidate_id}/feedback")
async def candidate_feedback(request: Request, candidate_id: int, session: SessionDep):
    c = _candidate(session, candidate_id)
    form = await request.form()
    verdict = str(form.get("verdict", ""))
    if verdict not in tuple(Verdict):
        raise HTTPException(status_code=400, detail="verdict inválido")
    fb = session.exec(
        select(CandidateFeedback).where(CandidateFeedback.candidate_id == c.id)
    ).first()
    if fb is None:
        fb = CandidateFeedback(candidate_id=c.id, verdict=verdict)
    elif fb.verdict == verdict:  # clicking the same verdict again clears it
        session.delete(fb)
        session.commit()
        return _card(request, session, c)
    fb.verdict = verdict
    fb.note = str(form.get("note", fb.note or ""))[:500]
    fb.updated_at = utcnow()
    session.add(fb)
    session.commit()
    return _card(request, session, c)


@router.post("/candidatos/{candidate_id}/status")
async def candidate_status(request: Request, candidate_id: int, session: SessionDep):
    c = _candidate(session, candidate_id)
    status = str((await request.form()).get("status", ""))
    if status not in tuple(CandidateStatus):
        raise HTTPException(status_code=400, detail="status inválido")
    c.status = CandidateStatus.SUGGESTED if c.status == status else status
    c.updated_at = utcnow()
    session.add(c)
    session.commit()
    return _card(request, session, c)


@router.post("/candidatos/{candidate_id}/ajuste")
async def candidate_adjust(
    request: Request, candidate_id: int, session: SessionDep, settings: SettingsDep
):
    c = _candidate(session, candidate_id)
    form = await request.form()
    if form.get("reset") == "1":
        c.user_start = c.user_end = None
    else:
        try:
            start = float(str(form.get("user_start", "")).replace(",", "."))
            end = float(str(form.get("user_end", "")).replace(",", "."))
        except ValueError:
            return _card(request, session, c, "Início e fim devem ser números (segundos).")
        source = session.get(Source, c.source_id)
        limit = source.duration_seconds if source and source.duration_seconds else float("inf")
        if not (0 <= start < end <= limit):
            return _card(
                request, session, c, "O fim precisa ser depois do início, dentro do vídeo."
            )
        if end - start > 180:
            return _card(request, session, c, "Trecho longo demais (máx. 3 min).")
        c.user_start, c.user_end = round(start, 2), round(end, 2)
    c.updated_at = utcnow()
    session.add(c)
    session.commit()
    return _card(request, session, c)
