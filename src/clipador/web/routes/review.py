"""/revisao: one candidate at a time across all sources, keyboard-first, with a live in-browser
preview (approximate), a waveform timeline to adjust the cut and an undoable decision history."""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from sqlmodel import Session, col, select

from clipador.config import Settings
from clipador.detection.params import SIGNALS
from clipador.errors import ClipadorError, ValidationError
from clipador.llm.scoring import RankingParams
from clipador.models import (
    AnalysisRun,
    Candidate,
    CandidateFeedback,
    Platform,
    Render,
    RenderKind,
    Source,
    Streamer,
)
from clipador.render import filters
from clipador.render.styles import get_style, load_styles
from clipador.render.subtitles import credit_overlay
from clipador.services import analysis, captions, rendering, review, waveform
from clipador.services.review import REJECT_REASONS, ReviewFilters
from clipador.utils import utcnow
from clipador.web.deps import get_session, get_settings_dep, redirect, render
from clipador.web.routes.analysis import SIGNAL_LABELS, _card_items, parse_seconds

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]

# key -> reason, in the order of the number keys 1..7
REASON_KEYS = list(REJECT_REASONS)
LAYOUT_CYCLE = ("blur", "facecam_embaixo", "jogo_cheio")


def _candidate(session: Session, candidate_id: int) -> Candidate:
    c = session.get(Candidate, candidate_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Candidato não encontrado")
    return c


def _streamer(session: Session, c: Candidate) -> tuple[Source, Streamer | None]:
    src = session.get(Source, c.source_id)
    return src, (session.get(Streamer, src.streamer_id) if src else None)


def _style_key(settings: Settings, c: Candidate, st: Streamer | None) -> str:
    return c.render_style or (st.default_caption_style if st else None) or settings.render_style


def live_config(session: Session, settings: Settings, c: Candidate, item: dict) -> dict:
    """Everything the browser needs to draw the approximate preview (mirrors plan_render)."""
    src, st = _streamer(session, c)
    rect = st.facecam_rect if st else None
    wanted = c.render_layout or (st.default_layout if st else "blur")
    layout, layout_warning = filters.resolve_layout(wanted, rect)
    o = c.render_options or {}
    show_opening = o.get("opening", st.show_opening_text if st else True)
    show_credit = o.get("credit", st.show_credit if st else True)
    opening = None
    if show_opening:
        opening = (c.overlay_title.strip() or None) if c.overlay_title is not None else (
            item.get("title_suggestion"))  # fmt: skip
    info = (src.media_info if src else None) or {}
    style = get_style(_style_key(settings, c, st), settings.caption_styles_path)
    return {
        "candidate": c.id, "source": c.source_id, "start": c.eff_start, "end": c.eff_end,
        "suggested": [c.start, c.end], "peak": c.peak_t,
        "duration": (src.duration_seconds if src else None) or None,
        "src_w": int(info.get("width") or 1920), "src_h": int(info.get("height") or 1080),
        "layout": layout, "layout_warning": layout_warning, "rect": rect,
        "zoom": float(o.get("zoom", 1.0)), "foco_x": float(o.get("foco_x", 0.5)),
        "style": asdict(style),
        "safe": {"bottom": settings.safe_bottom, "top": settings.safe_top,
                 "lift": settings.caption_lift},
        "opening": opening, "opening_seconds": settings.opening_text_seconds,
        "credit": credit_overlay(st.credit_text, st.display_name) if show_credit and st else None,
        "game_h": filters.GAME_H / filters.OUT_H,
        "min_s": settings.clip_min_seconds, "max_s": settings.clip_max_seconds,
        "reasons": REASON_KEYS,
    }  # fmt: skip


def _item(session: Session, settings: Settings, c: Candidate) -> dict:
    fb = session.exec(
        select(CandidateFeedback).where(CandidateFeedback.candidate_id == c.id)
    ).first()
    run = session.get(AnalysisRun, c.analysis_run_id)
    p = RankingParams.from_run(run.params, settings) if run else RankingParams()
    return _card_items(session, [(c, fb)], p, settings)[0]


def _filter_choices(session: Session) -> dict:
    sources = session.exec(select(Source).order_by(col(Source.id).desc())).all()
    streamers = session.exec(select(Streamer).order_by(Streamer.display_name)).all()
    return {"sources": sources, "streamers": streamers, "platforms": list(Platform)}


@router.get("/revisao")
def review_page(
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
    c: int | None = None,
    after: int | None = None,
):
    f = ReviewFilters.from_params(request.query_params)
    cand = session.get(Candidate, c) if c is not None else review.next_pending(session, f, after)
    ctx = {
        "f": f, "orders": review.ORDERS, "status_filters": review.STATUS_FILTERS,
        "reasons": REJECT_REASONS, "reason_keys": REASON_KEYS,
        "history": review.history(session), "status_labels": review.STATUS_LABELS,
        **_filter_choices(session),
    }  # fmt: skip
    if cand is None:
        return render(request, "review/index.html", cand=None,
                      progress=review.progress(session, f), **ctx)  # fmt: skip
    item = _item(session, settings, cand)
    if settings.review_prefetch > 0:
        nxt = review.upcoming(session, f, cand.id, settings.review_prefetch)
        if review.prefetch_previews(session, settings, nxt):
            session.commit()
    src, st = _streamer(session, cand)
    live = live_config(session, settings, cand, item)
    styles = load_styles(settings.caption_styles_path)
    alerted = review.has_alerts(item["e"])
    return render(
        request, "review/index.html", cand=cand, item=item, source=src, streamer=st, live=live,
        has_alerts=alerted, ready=review.ready_to_publish(cand, alerted), checks=cand.checks or {},
        progress=review.progress(session, f, cand.id), signals=SIGNALS,
        signal_labels=SIGNAL_LABELS, styles=styles, layout_labels=filters.LAYOUT_LABELS, **ctx,
    )  # fmt: skip


def _back(f: ReviewFilters, msg: str | None = None, **extra) -> str:
    q = f.query(**extra)
    return redirect("/revisao" + (f"?{q}" if q else ""), msg)


@router.post("/revisao/{candidate_id}/decisao")
async def decide(request: Request, candidate_id: int, session: SessionDep, settings: SettingsDep):
    _candidate(session, candidate_id)
    form = await request.form()
    f = ReviewFilters.from_params(form)
    action = str(form.get("action", ""))
    reason = str(form.get("reason_tag", "")).strip() or None
    note = str(form.get("note", "")).strip() or None
    try:
        res = review.apply_decision(session, candidate_id, action, reason, note, settings=settings)
        session.commit()
    except ValidationError as e:
        session.rollback()
        return _back(f, e.message, c=candidate_id)
    label = {"approve": "Aprovado", "reject": "Rejeitado", "skip": "Pulado",
             "reopen": "Reaberto"}.get(action, "Feito")  # fmt: skip
    msg = res.notice or f"{label}."
    if action == "reopen":
        return _back(f, msg, c=candidate_id)
    return _back(f, msg, after=candidate_id)


@router.post("/revisao/desfazer")
async def undo(request: Request, session: SessionDep, settings: SettingsDep):
    f = ReviewFilters.from_params(await request.form())
    d = review.undo_last(session, settings)
    session.commit()
    if d is None:
        return _back(f, "Nada para desfazer.")
    verb = review.ACTION_LABELS.get(d.action, d.action)
    return _back(f, f"Desfeito: {verb} o candidato.", c=d.candidate_id)


@router.post("/revisao/{candidate_id}/intervalo")
async def set_range(
    request: Request, candidate_id: int, session: SessionDep, settings: SettingsDep
):
    """JSON: save start/end (optionally snapped to word edges on the server)."""
    c = _candidate(session, candidate_id)
    form = await request.form()
    snap_words = None
    if form.get("snap") == "1":
        snap_words = [(a, b) for a, b, _w in rendering.transcript_words(
            session.get_bind(), settings, c.source_id)]  # fmt: skip
    try:
        c, warnings = review.edit_range(
            session, settings, c.id, parse_seconds(form.get("start")),
            parse_seconds(form.get("end")), reset=form.get("reset") == "1", snap_words=snap_words,
        )  # fmt: skip
        session.commit()
    except ValidationError as e:
        session.rollback()
        return JSONResponse({"ok": False, "error": e.message}, status_code=422)
    return JSONResponse(
        {
            "ok": True,
            "start": c.eff_start,
            "end": c.eff_end,
            "warnings": warnings,
            "adjusted": c.user_start is not None,
        }
    )


@router.post("/revisao/{candidate_id}/verificacoes")
async def set_checks(request: Request, candidate_id: int, session: SessionDep):
    """JSON: tick/untick the pre-publish checks (only the keys sent change)."""
    _candidate(session, candidate_id)
    form = await request.form()
    values = {k: str(form[k]) in ("1", "on", "true") for k in review.CHECK_KEYS if k in form}
    try:
        c = review.set_checks(session, candidate_id, **values)
        session.commit()
    except ValidationError as e:
        session.rollback()
        return JSONResponse({"ok": False, "error": e.message}, status_code=422)
    alerted = bool(review.alerted_ids(session, [c.id]))
    return JSONResponse({"ok": True, "checks": c.checks or {},
                         "ready": review.ready_to_publish(c, alerted)})  # fmt: skip


@router.post("/revisao/{candidate_id}/layout")
async def set_layout(request: Request, candidate_id: int, session: SessionDep):
    """JSON: set the layout (or cycle to the next one with layout=proximo)."""
    c = _candidate(session, candidate_id)
    _src, st = _streamer(session, c)
    rect = st.facecam_rect if st else None
    wanted = str((await request.form()).get("layout", "proximo"))
    if wanted == "proximo":
        current = filters.resolve_layout(c.render_layout or (st.default_layout if st else None),
                                         rect)[0]  # fmt: skip
        options = [x for x in LAYOUT_CYCLE if x != "facecam_embaixo" or rect]
        wanted = options[(options.index(current) + 1) % len(options)]
    layout, warning = filters.resolve_layout(wanted, rect)
    c.render_layout = layout
    c.updated_at = utcnow()
    session.add(c)
    session.commit()
    return JSONResponse({"layout": layout, "label": filters.LAYOUT_LABELS[layout],
                         "warning": warning})  # fmt: skip


def _preview_state(session: Session, settings: Settings, c: Candidate) -> dict:
    try:
        plan = rendering.plan_render(session, session.get_bind(), settings, c, RenderKind.PREVIEW)
    except ClipadorError as e:
        return {"status": "indisponivel", "error": e.message}
    r = session.exec(
        select(Render)
        .where(Render.candidate_id == c.id, Render.kind == RenderKind.PREVIEW,
               Render.params_hash == plan.phash)
        .order_by(col(Render.id).desc())
    ).first()  # fmt: skip
    if r is None:
        return {"status": "nenhuma"}
    out = {"status": r.status, "render_id": r.id, "progress": rendering.job_progress(session, r)}
    if r.status == "done" and r.path and settings.abs_path(r.path).is_file():
        out["url"] = f"/arquivos/{r.path}"
    if r.error:
        out["error"] = r.error
    return out


@router.get("/revisao/{candidate_id}/previa.json")
def preview_status(candidate_id: int, session: SessionDep, settings: SettingsDep):
    return JSONResponse(_preview_state(session, settings, _candidate(session, candidate_id)))


@router.post("/revisao/{candidate_id}/previa")
def request_preview(candidate_id: int, session: SessionDep, settings: SettingsDep):
    """P: faithful preview (real render) with the saved options, ahead of the prefetch queue."""
    c = _candidate(session, candidate_id)
    try:
        analysis.enqueue_render(session, settings, c.source_id, [c.id], RenderKind.PREVIEW)
        session.commit()
    except (ValidationError, ClipadorError) as e:
        session.rollback()
        return JSONResponse({"status": "indisponivel", "error": e.message}, status_code=422)
    return JSONResponse(_preview_state(session, settings, c))


@router.get("/revisao/fonte.ttf", include_in_schema=False)
def caption_font(settings: SettingsDep):
    fonts = sorted(rendering.fonts_dir(settings).glob("*.[ot]tf"))
    if not fonts:
        raise HTTPException(status_code=404)
    return FileResponse(fonts[0], media_type="font/ttf")


# --- JSON for the live preview and the timeline ---------------------------------------------------


@router.get("/candidatos/{candidate_id}/palavras.json")
def words_json(
    candidate_id: int,
    session: SessionDep,
    settings: SettingsDep,
    start: float | None = None,
    end: float | None = None,
    pad: float = 0.0,
):
    """Words of [start - pad, end + pad] (default: the current range) with their GLOBAL index,
    ABSOLUTE times (original video) and the caption corrections already applied."""
    c = _candidate(session, candidate_id)
    a = (start if start is not None else c.eff_start) - max(0.0, min(pad, 300.0))
    b = (end if end is not None else c.eff_end) + max(0.0, min(pad, 300.0))
    a = max(0.0, a)
    return JSONResponse(_words_payload(session, settings, c, a, b))


def _words_payload(session: Session, settings: Settings, c: Candidate, a: float, b: float) -> dict:
    words = rendering.transcript_words(session.get_bind(), settings, c.source_id)
    _src, st = _streamer(session, c)
    glossary = st.glossary if st else None
    shown, stale = captions.shown_words(words, c, glossary, a, b)
    rows, _ = captions.editor_words(words, c, glossary, a, b)
    return {
        "start": round(a, 3), "end": round(b, 3), "style": _style_key(settings, c, st),
        "words": [{"i": w.idx, "s": round(a + w.start, 3), "e": round(a + w.end, 3), "t": w.text}
                  for w in shown],
        "editor": [r.as_json() for r in rows],
        "stale": [asdict(s) for s in stale],
        "streamer": st.display_name if st else None,
    }  # fmt: skip


@router.post("/candidatos/{candidate_id}/legenda/palavra")
async def edit_caption_word(
    request: Request, candidate_id: int, session: SessionDep, settings: SettingsDep
):
    """JSON: correct / hide / un-correct one word (timing stays the transcript's); optionally
    also add the correction to the streamer glossary."""
    c = _candidate(session, candidate_id)
    form = await request.form()
    try:
        idx = int(str(form.get("i", "")))
    except ValueError:
        return JSONResponse({"ok": False, "error": "Palavra inválida."}, status_code=422)
    try:
        msg = captions.edit_word(
            session, settings, c, idx, str(form.get("text", "")),
            hide=form.get("hide") == "1", undo=form.get("undo") == "1",
            to_glossary=form.get("glossary") in ("1", "on"),
        )  # fmt: skip
        session.add(c)
        session.commit()
    except ValidationError as e:
        session.rollback()
        return JSONResponse({"ok": False, "error": e.message}, status_code=422)
    return JSONResponse({"ok": True, "message": msg})


@router.post("/candidatos/{candidate_id}/legenda/obsoletas/remover")
async def remove_stale_captions(
    request: Request, candidate_id: int, session: SessionDep, settings: SettingsDep
):
    c = _candidate(session, candidate_id)
    raw = str((await request.form()).get("i", "")).strip()
    n = captions.remove_stale(session, settings, c, int(raw) if raw.isdigit() else None)
    session.add(c)
    session.commit()
    return JSONResponse({"ok": True, "removed": n})


@router.get("/fontes/{source_id}/forma-de-onda.json")
def waveform_json(
    source_id: int, start: float, end: float, session: SessionDep, settings: SettingsDep
):
    if session.get(Source, source_id) is None:
        raise HTTPException(status_code=404, detail="Fonte não encontrada")
    if end < start or end - start > 3600:
        raise HTTPException(status_code=400, detail="intervalo inválido")
    try:
        data = waveform.waveform(session.get_bind(), settings, source_id, start, end)
    except ClipadorError as e:
        data = {"peaks": [], "error": e.message}
    return JSONResponse(data)
