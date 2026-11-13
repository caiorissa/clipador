"""/fontes/{id}: source detail, analysis stages, timeline, candidates, tuning, video."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from sqlmodel import Session, col, select

from clipador.config import Settings
from clipador.detection.params import PRESET_LABELS, PRESETS, SIGNALS, DetectParams
from clipador.errors import ClipadorError, ValidationError
from clipador.llm.drafts import DRAFT_LABELS, DRAFT_PLATFORMS, clean_hashtags, clean_text
from clipador.llm.scoring import RankingParams
from clipador.models import (
    RENDERABLE_STATUSES,
    AnalysisRun,
    Candidate,
    CandidateFeedback,
    CandidateStatus,
    DecisionAction,
    PostDraft,
    RenderKind,
    Source,
    Streamer,
    Verdict,
)
from clipador.render.filters import LAYOUT_LABELS, normalize_layout, resolve_layout
from clipador.render.styles import load_styles
from clipador.render.subtitles import apply_glossary, words_in_window
from clipador.services import analysis, captions, rendering, review
from clipador.services.enrichment import drafts_for, enrichments_for, score_parts, set_ranking
from clipador.utils import utcnow
from clipador.web.deps import get_session, get_settings_dep, redirect, render

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]

STATUS_TO_ACTION = {
    CandidateStatus.APPROVED: DecisionAction.APPROVE,
    CandidateStatus.REJECTED: DecisionAction.REJECT,
    CandidateStatus.SKIPPED: DecisionAction.SKIP,
    CandidateStatus.SUGGESTED: DecisionAction.REOPEN,
}
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
    filtro: str = "",
):
    source = _source(session, source_id)
    runs = analysis.runs_for(session, source_id)
    selected = _selected_run(session, source_id, run)
    cands = analysis.candidates_for(session, selected.id) if selected else []
    params = _form_params(selected, settings.detect_preset)
    ranking = RankingParams.from_run(selected.params, settings) if selected else None
    items = _card_items(session, cands, ranking or RankingParams.from_settings(settings), settings)
    items = _sorted(items, sort)
    ready_count = sum(1 for i in items if i["ready"])
    if filtro == "prontos":
        items = [i for i in items if i["ready"]]
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
        filtro="prontos" if filtro == "prontos" else "",
        ready_count=ready_count,
        ranking=ranking,
        enrich_art=enrich_art,
        llm_model=(enrich_art.summary.get("model") if enrich_art and enrich_art.summary else None)
        or settings.llm_model,
        enrich_k=settings.enrich_k,
        game=source.game or settings.default_game,
        draft_platforms=DRAFT_PLATFORMS,
        draft_labels=DRAFT_LABELS,
        styles=load_styles(settings.caption_styles_path),
        layout_labels=LAYOUT_LABELS,
        long_art=analysis.artifact(session, source_id, "compile_long"),
        approved_count=sum(1 for c, _ in cands if c.status in RENDERABLE_STATUSES),
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
    renders = rendering.latest_renders(session, ids)
    words = (
        rendering.transcript_words(session.get_bind(), settings, cands[0][0].source_id)
        if settings is not None and cands
        else []
    )
    streamer = None
    if cands:
        src = session.get(Source, cands[0][0].source_id)
        streamer = session.get(Streamer, src.streamer_id) if src else None
    rect = streamer.facecam_rect if streamer else None
    items = []
    for c, fb in cands:
        d = drafts.get(c.id, {})
        shown, stale = captions.shown_words(words, c, streamer.glossary if streamer else None)
        r = renders.get(c.id, {})
        wanted = c.render_layout or (streamer.default_layout if streamer else "blur")
        items.append(
            {
                "c": c,
                "fb": fb,
                "e": enr.get(c.id),
                "drafts": d,
                "parts": score_parts(c, enr.get(c.id), p),
                "preview": r.get(RenderKind.PREVIEW),
                "final": r.get(RenderKind.FINAL),
                "progress": {k: rendering.job_progress(session, v) for k, v in r.items()},
                "sub_text": " ".join(w.text for w in shown),
                "stale_captions": stale,
                "ready": review.ready_to_publish(c, review.has_alerts(enr.get(c.id))),
                "title_suggestion": rendering.suggested_title(enr.get(c.id), d.get("youtube")),
                "layout": normalize_layout(wanted),
                "layout_warning": resolve_layout(wanted, rect)[1],
                "options": c.render_options or {},
                "streamer": streamer,
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
        styles=load_styles(settings.caption_styles_path),
        layout_labels=LAYOUT_LABELS,
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
    """Toggle approved/rejected from the source page. Goes through the review service, so it is
    recorded as a Decision (undoable in /revisao) and keeps the feedback in sync."""
    settings: Settings = request.app.state.settings
    c = _candidate(session, candidate_id)
    status = str((await request.form()).get("status", ""))
    if status not in tuple(CandidateStatus):
        raise HTTPException(status_code=400, detail="status inválido")
    action = DecisionAction.REOPEN if c.status == status else STATUS_TO_ACTION.get(status)
    if action is None:
        return _card(request, session, c, "Esse status só é definido pelo render.")
    try:
        res = review.apply_decision(session, c.id, action, settings=settings)
        session.commit()
    except ValidationError as e:
        session.rollback()
        return _card(request, session, _candidate(session, candidate_id), e.message)
    return _card(request, session, res.candidate, notice=res.notice)


def parse_seconds(raw) -> float | None:
    try:
        return float(str(raw if raw is not None else "").replace(",", ".").strip())
    except ValueError:
        return None


@router.post("/candidatos/{candidate_id}/ajuste")
async def candidate_adjust(
    request: Request, candidate_id: int, session: SessionDep, settings: SettingsDep
):
    c = _candidate(session, candidate_id)
    form = await request.form()
    try:
        c, warnings = review.edit_range(
            session, settings, c.id, parse_seconds(form.get("user_start")),
            parse_seconds(form.get("user_end")), reset=form.get("reset") == "1",
        )  # fmt: skip
        session.commit()
    except ValidationError as e:
        session.rollback()
        return _card(request, session, _candidate(session, candidate_id), e.message)
    return _card(request, session, c, notice=" ".join(warnings) or None)


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


def _save_render_options(
    c: Candidate, form, words_all, settings: Settings, glossary: dict | None = None
) -> str | None:
    """Apply the form to the candidate. Returns a PT-BR error or None."""
    c.render_layout = normalize_layout(str(form.get("layout"))) if form.get("layout") else None
    style = str(form.get("style", ""))
    c.render_style = style if style in load_styles(settings.caption_styles_path) else None
    if "overlay_title" in form:
        c.overlay_title = " ".join(str(form["overlay_title"]).split())[:80]
    opts = dict(c.render_options or {})
    opts["opening"] = form.get("opening") == "on"
    opts["credit"] = form.get("credit") == "on"
    opts["peak_zoom"] = form.get("peak_zoom") == "on"
    for key, lo, hi in (("zoom", 1.0, 3.0), ("foco_x", 0.0, 1.0)):
        raw = str(form.get(key, "")).replace(",", ".").strip()
        if raw:
            try:
                opts[key] = min(hi, max(lo, float(raw)))
            except ValueError:
                return f"Valor inválido em {key}."
    c.render_options = opts
    if "subtitle_text" in form:
        window = words_in_window(words_all, c.eff_start, c.eff_end)
        tokens = str(form["subtitle_text"]).split()
        if window and len(tokens) != len(window):
            return (
                f"A legenda tem {len(window)} palavras e você escreveu {len(tokens)}: corrija "
                "palavra por palavra (use '-' para esconder uma palavra) ou use o editor de "
                "legenda do card."
            )
        glossed = apply_glossary(window, glossary)  # what the caption shows without corrections
        in_window = {str(w.idx) for w in window}
        overrides = {k: v for k, v in (c.caption_overrides or {}).items() if k not in in_window}
        originals = {k: v for k, v in (c.caption_override_originals or {}).items()
                     if k not in in_window}  # fmt: skip
        for w, g, t in zip(window, glossed, tokens, strict=False):
            if t != g.text:
                overrides[str(w.idx)] = "" if t == "-" else t
                originals[str(w.idx)] = w.text
        c.caption_overrides = overrides or None
        c.caption_override_originals = originals or None
    c.updated_at = utcnow()
    return None


@router.post("/candidatos/{candidate_id}/corte")
async def render_candidate(request: Request, candidate_id: int, session: SessionDep):
    """Save the clip options and render a preview or the final clip."""
    settings: Settings = request.app.state.settings
    c = _candidate(session, candidate_id)
    form = await request.form()
    words_all = rendering.transcript_words(session.get_bind(), settings, c.source_id)
    st = captions.streamer_of(session, c)
    err = _save_render_options(c, form, words_all, settings, st.glossary if st else None)
    if err:
        session.rollback()
        return _card(request, session, _candidate(session, candidate_id), err)
    session.add(c)
    kind = "preview" if form.get("kind") == "preview" else "final"
    try:
        job, _todo, reused = analysis.enqueue_render(session, settings, c.source_id, [c.id], kind)
        session.commit()
    except (ValidationError, ClipadorError) as e:
        session.rollback()
        return _card(request, session, _candidate(session, candidate_id), e.message)
    label = "Prévia" if kind == "preview" else "Render final"
    if job is None and reused:
        notice = f"{label} com estes ajustes já existe: reaproveitada."
    else:
        notice = f"{label} na fila (job #{job.id})… a página atualiza sozinha."
    return _card(request, session, c, notice=notice)


@router.post("/fontes/{source_id}/renderizar-aprovados")
async def render_approved(request: Request, source_id: int, session: SessionDep):
    settings: Settings = request.app.state.settings
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
                Candidate.analysis_run_id == run.id,
                col(Candidate.status).in_(RENDERABLE_STATUSES),
            )
            .order_by(Candidate.final_rank, Candidate.rank)
        )
    ]
    try:
        job, todo, reused = analysis.enqueue_render(session, settings, source_id, ids, "final")
        session.commit()
    except (ValidationError, ClipadorError) as e:
        session.rollback()
        return redirect(f"/fontes/{source_id}", e.message)
    msg = f"{len(todo)} render(s) final(is) na fila" + (f" (job #{job.id})" if job else "")
    if reused:
        msg += f"; {len(reused)} já estavam prontos"
    return redirect(f"/fontes/{source_id}", msg + ".")


@router.post("/fontes/{source_id}/video-longo")
async def build_long_video(request: Request, source_id: int, session: SessionDep):
    _source(session, source_id)
    form = await request.form()
    run = _selected_run(session, source_id, int(form["run"]) if form.get("run") else None)
    if run is None:
        return redirect(f"/fontes/{source_id}", "Analise a fonte primeiro.")
    approved = session.exec(
        select(Candidate).where(
            Candidate.analysis_run_id == run.id, col(Candidate.status).in_(RENDERABLE_STATUSES)
        )
    ).first()
    if approved is None:
        return redirect(f"/fontes/{source_id}", "Aprove pelo menos um candidato antes.")
    job = analysis.enqueue_compile_long(session, source_id, run.id)
    session.commit()
    return redirect(f"/fontes/{source_id}", f"Montando o vídeo longo 16:9 (job #{job.id}).")
