"""/fontes/{id}: source detail, analysis stages, timeline, candidates, tuning, video."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from sqlmodel import Session, col, select

from clipador.config import Settings
from clipador.detection.params import PRESET_LABELS, PRESETS, SIGNALS, DetectParams
from clipador.errors import ValidationError
from clipador.llm.drafts import DRAFT_LABELS, DRAFT_PLATFORMS, clean_hashtags, clean_text
from clipador.llm.scoring import RankingParams
from clipador.models import (
    AnalysisRun,
    Candidate,
    CandidateFeedback,
    CandidateStatus,
    PostDraft,
    RenderedClip,
    Source,
    Verdict,
)
from clipador.render.styles import STYLES
from clipador.render.subtitles import words_in_window
from clipador.services import analysis, rendering
from clipador.services.enrichment import drafts_for, enrichments_for, score_parts, set_ranking
from clipador.utils import utcnow
from clipador.web.deps import get_session, get_settings_dep, redirect, render

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]

SIGNAL_LABELS = {"audio": "Áudio", "chat": "Chat", "speech": "Fala", "keywords": "Palavras"}
SORTS = {"final": "Nota final", "detect": "Detecção", "ia": "IA", "time": "Horário"}


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
    sort: str = "final",
):
    source = _source(session, source_id)
    runs = analysis.runs_for(session, source_id)
    selected = _selected_run(session, source_id, run)
    cands = analysis.candidates_for(session, selected.id) if selected else []
    params = _form_params(selected, settings.detect_preset)
    ranking = RankingParams.from_run(selected.params, settings) if selected else None
    items = _card_items(session, cands, ranking or RankingParams.from_settings(settings), settings)
    items = _sorted(items, sort)
    enrich_art = analysis.artifact(session, source_id, "enrich")
    return render(
        request,
        "analysis/detail.html",
        source=source,
        runs=runs,
        run=selected,
        pinned_run=run is not None,
        candidates=cands,
        items=items,
        sort=sort if sort in SORTS else "final",
        sorts=SORTS,
        ranking=ranking,
        enrich_art=enrich_art,
        llm_model=(enrich_art.summary.get("model") if enrich_art and enrich_art.summary else None)
        or settings.llm_model,
        enrich_k=settings.enrich_k,
        game=source.game or settings.default_game,
        draft_platforms=DRAFT_PLATFORMS,
        draft_labels=DRAFT_LABELS,
        styles=STYLES,
        layout_labels=rendering.LAYOUT_LABELS,
        long_art=analysis.artifact(session, source_id, "compile_long"),
        approved_count=sum(1 for c, _ in cands if c.status == CandidateStatus.APPROVED),
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


def _card_items(
    session: Session,
    cands: list[tuple[Candidate, CandidateFeedback | None]],
    p: RankingParams,
    settings: Settings | None = None,
) -> list[dict]:
    ids = [c.id for c, _ in cands]
    enr = enrichments_for(session, ids)
    drafts = drafts_for(session, ids)
    clips = {
        r.candidate_id: r
        for r in session.exec(select(RenderedClip).where(col(RenderedClip.candidate_id).in_(ids)))
    }
    words = (
        rendering.transcript_words(session.get_bind(), settings, cands[0][0].source_id)
        if settings is not None and cands
        else []
    )
    items = []
    for c, fb in cands:
        d = drafts.get(c.id, {})
        auto_words = words_in_window(words, c.eff_start, c.eff_end)
        items.append(
            {
                "c": c,
                "fb": fb,
                "e": enr.get(c.id),
                "drafts": d,
                "parts": score_parts(c, enr.get(c.id), p),
                "clip": clips.get(c.id),
                "sub_text": c.subtitle_text
                if c.subtitle_text is not None
                else " ".join(w[2] for w in auto_words),
                "title_suggestion": rendering.suggested_title(enr.get(c.id), d.get("youtube")),
            }
        )
    return items


def _sorted(items: list[dict], sort: str) -> list[dict]:
    def final(i):
        return -(i["c"].final_score if i["c"].final_score is not None else i["parts"].final)

    keys = {
        "final": final,
        "detect": lambda i: -i["c"].score,
        "ia": lambda i: (i["parts"].llm is None, -(i["parts"].llm or 0), -i["c"].score),
        "time": lambda i: i["c"].eff_start,
    }
    return sorted(items, key=keys.get(sort, final))


def _card(
    request: Request,
    session: Session,
    c: Candidate,
    error: str | None = None,
    notice: str | None = None,
):
    settings: Settings = request.app.state.settings
    fb = session.exec(
        select(CandidateFeedback).where(CandidateFeedback.candidate_id == c.id)
    ).first()
    run = session.get(AnalysisRun, c.analysis_run_id)
    p = RankingParams.from_run(run.params, settings) if run else RankingParams()
    item = _card_items(session, [(c, fb)], p, settings)[0]
    return render(
        request,
        "analysis/_candidate.html",
        item=item,
        source_id=c.source_id,
        signals=SIGNALS,
        signal_labels=SIGNAL_LABELS,
        draft_platforms=DRAFT_PLATFORMS,
        draft_labels=DRAFT_LABELS,
        styles=STYLES,
        layout_labels=rendering.LAYOUT_LABELS,
        card_error=error,
        card_notice=notice,
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


# --- step 3: AI enrichment, ranking, drafts ---------------------------------------------------


@router.post("/fontes/{source_id}/enriquecer")
async def enrich_source(request: Request, source_id: int, session: SessionDep):
    _source(session, source_id)
    form = await request.form()
    model = str(form.get("model", "")).strip() or None
    try:
        k = int(str(form.get("k", "")).strip() or 0) or None
    except ValueError:
        k = None
    try:
        job = analysis.enqueue_enrich(session, source_id, model=model, k=k)
        session.commit()
    except ValidationError as e:
        session.rollback()
        return redirect(f"/fontes/{source_id}", e.message)
    return redirect(f"/fontes/{source_id}", f"Enriquecimento com IA na fila (job #{job.id}).")


@router.post("/candidatos/{candidate_id}/regerar")
async def regenerate_candidate(request: Request, candidate_id: int, session: SessionDep):
    c = _candidate(session, candidate_id)
    form = await request.form()
    model = str(form.get("model", "")).strip() or None
    try:
        analysis.enqueue_enrich(
            session,
            c.source_id,
            model=model,
            candidate_ids=[c.id],
            instruction=str(form.get("instruction", "")),
            overwrite_edited=form.get("overwrite_edited") == "on",
        )
        session.commit()
    except ValidationError as e:
        session.rollback()
        return _card(request, session, c, e.message)
    return _card(request, session, c, notice="Regerando com IA… recarregue a página em instantes.")


@router.post("/fontes/{source_id}/ordenacao")
async def set_ranking_route(
    request: Request, source_id: int, session: SessionDep, settings: SettingsDep
):
    _source(session, source_id)
    form = await request.form()
    run = _selected_run(session, source_id, int(form["run"]) if form.get("run") else None)
    if run is None:
        return redirect(f"/fontes/{source_id}", "Analise a fonte primeiro.")

    def num(name: str, lo: float, hi: float) -> float | None:
        raw = str(form.get(name, "")).strip().replace(",", ".")
        try:
            return min(hi, max(lo, float(raw))) if raw else None
        except ValueError:
            return None

    current = RankingParams.from_run(run.params, settings)
    data = current.model_dump()
    for name, lo, hi in (
        ("w_detect", 0, 1), ("w_llm", 0, 1), ("penalty_needs_context", 0, 50),
        ("penalty_low_standalone", 0, 50), ("low_standalone_threshold", 0, 10),
    ):  # fmt: skip
        v = num(name, lo, hi)
        if v is not None:
            data[name] = v
    if data["w_detect"] + data["w_llm"] <= 0:
        return redirect(f"/fontes/{source_id}", "Pelo menos um dos pesos precisa ser maior que 0.")
    set_ranking(session, run, RankingParams(**data), settings)
    session.commit()
    return redirect(f"/fontes/{source_id}?run={run.id}", "Ordem recalculada (sem chamar a IA).")


@router.post("/fontes/{source_id}/jogo")
async def set_game(request: Request, source_id: int, session: SessionDep):
    source = _source(session, source_id)
    game = " ".join(str((await request.form()).get("game", "")).split())[:60]
    source.game = game or None
    source.updated_at = utcnow()
    session.add(source)
    session.commit()
    return redirect(f"/fontes/{source_id}", f"Jogo definido: {game or 'padrão'}.")


@router.post("/rascunhos/{draft_id}")
async def save_draft(request: Request, draft_id: int, session: SessionDep):
    d = session.get(PostDraft, draft_id)
    if d is None:
        raise HTTPException(status_code=404, detail="Rascunho não encontrado")
    form = await request.form()
    if "title" in form:
        d.title = clean_text(str(form["title"]), one_line=True)[:100]
    if "description" in form:
        d.description = clean_text(str(form["description"]))[:5000]
    if "hashtags" in form:
        raw = str(form["hashtags"]).replace(",", " ").split()
        d.hashtags = clean_hashtags(raw, max_n=30)
    d.edited_by_user = True
    d.updated_at = utcnow()
    session.add(d)
    session.commit()
    return HTMLResponse(
        f'<span class="saved">salvo {utcnow().astimezone().strftime("%H:%M:%S")} · '
        f"{' '.join(d.hashtags)}</span>"
    )


# --- step 4: rendering ---------------------------------------------------------------------------


@router.post("/candidatos/{candidate_id}/corte")
async def render_candidate(request: Request, candidate_id: int, session: SessionDep):
    """Save the clip options (layout, style, title, corrected subtitles) and render it."""
    c = _candidate(session, candidate_id)
    form = await request.form()
    layout = str(form.get("layout", ""))
    c.render_layout = layout if layout in rendering.LAYOUT_LABELS else None
    style = str(form.get("style", ""))
    c.render_style = style if style in STYLES else None
    if "overlay_title" in form:
        c.overlay_title = " ".join(str(form["overlay_title"]).split())[:80]
    if "subtitle_text" in form:
        text = " ".join(str(form["subtitle_text"]).split())
        auto = " ".join(
            w[2]
            for w in words_in_window(
                rendering.transcript_words(session.get_bind(), request.app.state.settings,
                                           c.source_id), c.eff_start, c.eff_end)
        )  # fmt: skip
        c.subtitle_text = None if text == auto else text
    c.updated_at = utcnow()
    session.add(c)
    try:
        job = analysis.enqueue_render(session, c.source_id, [c.id])
        session.commit()
    except ValidationError as e:
        session.rollback()
        return _card(request, session, c, e.message)
    return _card(
        request, session, c, notice=f"Renderizando (job #{job.id})… a página atualiza sozinha."
    )


@router.post("/fontes/{source_id}/renderizar-aprovados")
async def render_approved(request: Request, source_id: int, session: SessionDep):
    _source(session, source_id)
    form = await request.form()
    run = _selected_run(session, source_id, int(form["run"]) if form.get("run") else None)
    if run is None:
        return redirect(f"/fontes/{source_id}", "Analise a fonte primeiro.")
    ids = [
        c.id
        for c in session.exec(
            select(Candidate)
            .where(
                Candidate.analysis_run_id == run.id, Candidate.status == CandidateStatus.APPROVED
            )
            .order_by(Candidate.final_rank, Candidate.rank)
        )
    ]
    try:
        job = analysis.enqueue_render(session, source_id, ids)
        session.commit()
    except ValidationError as e:
        session.rollback()
        return redirect(f"/fontes/{source_id}", e.message)
    return redirect(f"/fontes/{source_id}", f"Renderizando {len(ids)} corte(s) (job #{job.id}).")


@router.post("/fontes/{source_id}/video-longo")
async def build_long_video(request: Request, source_id: int, session: SessionDep):
    _source(session, source_id)
    form = await request.form()
    run = _selected_run(session, source_id, int(form["run"]) if form.get("run") else None)
    if run is None:
        return redirect(f"/fontes/{source_id}", "Analise a fonte primeiro.")
    approved = session.exec(
        select(Candidate).where(
            Candidate.analysis_run_id == run.id, Candidate.status == CandidateStatus.APPROVED
        )
    ).first()
    if approved is None:
        return redirect(f"/fontes/{source_id}", "Aprove pelo menos um candidato antes.")
    job = analysis.enqueue_compile_long(session, source_id, run.id)
    session.commit()
    return redirect(f"/fontes/{source_id}", f"Montando o vídeo longo 16:9 (job #{job.id}).")
