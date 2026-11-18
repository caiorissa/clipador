"""live_analyze / live_finalize: the worker side of a live capture (step 5E).

live_analyze (one job per closed segment, GPU lock because of Whisper):
1. 16 kHz WAV of the segment (data/live/<id>/seg_N.wav) and its transcript (seg_N.words.json,
   word times on the session's global timeline);
2. features over a ROLLING history: the last `live_baseline_minutes` of the session (audio,
   speech, chat) ending at this segment. The audio/speech/chat signals are already measured
   against a rolling local median, so a score is comparable between segments, and the first
   minutes of a live simply use what exists (warm-up);
3. candidates whose peak falls in the new part (this segment + `live_context_seconds` of the
   previous one), merged with the session's existing candidates (overlap -> keep the best and
   extend the limits, never touching a decision already taken);
4. AI enrichment of the new ones when Ollama is up, then auto-clip: final score >= the
   session's minimum -> faithful preview queued with top priority. Never approved by itself.

live_finalize (when the capture ends): joins the segments into one video (concat, -c copy),
turning the live into a normal VOD source that can be analyzed whole.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

import numpy as np
from sqlmodel import col, select

from clipador.detection import audio_events
from clipador.detection.candidates import CandidateResult, detect_candidates
from clipador.detection.features import compute_features
from clipador.detection.hype import ensure_hype_file, load_lexicon
from clipador.detection.params import DetectParams
from clipador.errors import ClipadorError
from clipador.models import (
    AnalysisRun,
    Candidate,
    CandidateStatus,
    LiveSegment,
    LiveSession,
    SegmentStatus,
    Source,
    SourceKind,
    SourceStatus,
)
from clipador.pipeline.base import Stage, StageContext
from clipador.services import live
from clipador.services import transcriber as tx
from clipador.services.audio import WavReader, extract_wav
from clipador.utils import utcnow

ANALYZE = "live_analyze"
FINALIZE = "live_finalize"


def _load(ctx: StageContext) -> tuple[LiveSession, LiveSegment]:
    with ctx.session() as s:
        ls = s.get(LiveSession, ctx.payload.get("session_id"))
        seg = s.get(LiveSegment, ctx.payload.get("segment_id"))
    if ls is None or seg is None:
        raise ClipadorError("Sessão ou segmento da live não existe mais.")
    return ls, seg


def _history(ctx: StageContext, ls: LiveSession, seg: LiveSegment) -> list[LiveSegment]:
    """Contiguous segments (with their WAV on disk) ending at `seg`, at most
    live_baseline_minutes long."""
    settings = ctx.settings
    with ctx.session() as s:
        segs = [g for g in live.segments(s, ls.id) if g.index <= seg.index]
    out: list[LiveSegment] = []
    span = 0.0
    limit = settings.live_baseline_minutes * 60
    for g in reversed(segs):
        wav = live.segment_wav(settings, ls.id, g.index)
        if g.index != seg.index and (not wav.is_file() or (out and g.index != out[-1].index - 1)):
            break
        out.append(g)
        span += g.duration_s
        if span >= limit:
            break
    return list(reversed(out))


def _blocks(readers: list[WavReader]):
    """Chain the history WAVs as one stream of (t, samples) blocks."""
    t = 0.0
    for r in readers:
        for bt, samples in r.blocks(60.0):
            yield t + bt, samples
        t += r.duration


def merge_into_session(
    s, ls: LiveSession, results: list[CandidateResult], new_from: float, max_duration: float
) -> list[Candidate]:
    """Add/merge candidates found in the new part of the timeline. Overlapping an existing
    candidate: the better score wins (its limits and numbers), and a moment that continues is
    extended (up to max_duration). Decided candidates (approved/rejected/rendered) are never
    changed. Returns the candidates created or changed."""
    existing = list(s.exec(select(Candidate).where(Candidate.live_session_id == ls.id)).all())
    touched: list[Candidate] = []
    for r in sorted(results, key=lambda r: -r.score):
        if r.peak_t < new_from:
            continue
        old = next((c for c in existing if c.start < r.end and r.start < c.end), None)
        if old is None:
            c = Candidate(
                source_id=ls.source_id, analysis_run_id=ls.run_id, rank=0, start=r.start,
                end=r.end, peak_t=r.peak_t, score=r.score, breakdown=r.breakdown,
                transcript_snippet=r.snippet, live_session_id=ls.id,
            )  # fmt: skip
            s.add(c)
            existing.append(c)
            touched.append(c)
            continue
        if old.status not in (CandidateStatus.SUGGESTED, CandidateStatus.SKIPPED):
            continue
        changed = False
        if r.score > old.score:
            old.start, old.end, old.peak_t = r.start, r.end, r.peak_t
            old.score, old.breakdown, old.transcript_snippet = r.score, r.breakdown, r.snippet
            changed = True
        elif r.end > old.end and min(r.end, old.start + max_duration) > old.end:
            old.end = round(min(r.end, old.start + max_duration), 2)
            changed = True
        if changed:
            old.final_score = None  # the AI looks at it again
            old.updated_at = utcnow()
            s.add(old)
            touched.append(old)
    s.flush()
    # ranks over the whole session (best first)
    ranked = sorted(existing, key=lambda c: (-(c.final_score or c.score), c.start))
    for i, c in enumerate(ranked, 1):
        if c.rank != i:
            c.rank = i
            s.add(c)
    return touched


def apply_auto_clip(s, settings, ls: LiveSession) -> list[Candidate]:
    """Candidates of the session at/above min_score get auto_clipped=True and their faithful
    preview at top priority. Never approved: the user decides in /revisao. Does not commit."""
    from clipador.services import previews

    if not ls.auto_clip:
        return []
    cands = s.exec(
        select(Candidate).where(
            Candidate.live_session_id == ls.id,
            Candidate.auto_clipped == False,  # noqa: E712
            col(Candidate.status).in_([CandidateStatus.SUGGESTED, CandidateStatus.SKIPPED]),
        )
    ).all()
    hits = [c for c in cands if (c.final_score if c.final_score is not None else c.score)
            >= ls.min_score]  # fmt: skip
    for c in hits:
        c.auto_clipped = True
        c.updated_at = utcnow()
        s.add(c)
    if hits:
        previews.enqueue_previews(
            s, settings, previews.without_preview(s, hits), previews.AUTO_CLIP_PRIORITY
        )
    return hits


def _unscored(s, live_id: int, skip: set[int], limit: int = 20) -> list[int]:
    """Clips of the session the AI hasn't scored yet (e.g. the subscription limit was hit):
    retried with the next segment, best detection score first."""
    from clipador.models import CandidateEnrichment, EnrichStatus

    ok = {e.candidate_id for e in s.exec(select(CandidateEnrichment).where(
        CandidateEnrichment.status == EnrichStatus.OK)).all()}  # fmt: skip
    rows = s.exec(select(Candidate).where(
        Candidate.live_session_id == live_id,
        col(Candidate.status).in_([CandidateStatus.SUGGESTED, CandidateStatus.SKIPPED]),
    ).order_by(col(Candidate.score).desc())).all()  # fmt: skip
    return [c.id for c in rows if c.id not in ok and c.id not in skip][:limit]


def _llm_up(settings) -> bool:
    try:
        from clipador.llm.client import ollama_status

        return bool(ollama_status(settings).ok)
    except Exception:
        return False


_RESIDENT: dict[str, Any] = {}


def _live_transcriber(settings, log_cb) -> tuple[Any, bool]:
    """On a Mac (MLX, unified memory) the Whisper model stays loaded between live segments:
    reloading it for every 5-min segment was a big part of each analysis. On the PC
    (faster-whisper on an 8 GB GPU) it is still freed after each segment, so the local AI
    can use the VRAM. Returns (transcriber, keep_loaded)."""
    if tx.resolve_backend(settings) != "mlx":
        return tx.make_transcriber(settings, log_cb), False
    key = tx.resolve_model(settings, "mlx")
    t = _RESIDENT.get(key)
    if t is None:
        t = _RESIDENT[key] = tx.make_transcriber(settings, log_cb)
    t.log_cb = log_cb
    return t, True


class LiveAnalyzeStage(Stage):
    name = ANALYZE
    needs_gpu = True  # Whisper
    max_attempts = 2

    def is_done(self, ctx: StageContext) -> bool:
        with ctx.session() as s:
            seg = s.get(LiveSegment, ctx.payload.get("segment_id"))
        return seg is not None and seg.status in (SegmentStatus.ANALYZED, SegmentStatus.EXPIRED)

    def on_failure(self, ctx: StageContext, message: str, final: bool) -> None:
        if not final:
            return
        with ctx.session() as s:
            seg = s.get(LiveSegment, ctx.payload.get("segment_id"))
            if seg is not None:
                seg.status, seg.error = SegmentStatus.FAILED, message[:500]
                seg.updated_at = utcnow()
                s.add(seg)
                s.commit()

    def run(self, ctx: StageContext) -> None:
        settings = ctx.settings
        t_start = time.monotonic()
        ls, seg = _load(ctx)
        if not seg.path or not settings.abs_path(seg.path).is_file():
            raise ClipadorError(f"O arquivo do segmento {seg.index} não está mais no disco.")
        wav = live.segment_wav(settings, ls.id, seg.index)
        if not wav.is_file():
            ctx.progress(2, f"Segmento {seg.index}: extraindo o áudio…")
            extract_wav(settings.abs_path(seg.path), wav, start=None, end=None,
                        duration_hint=seg.duration_s, progress_cb=lambda p: None,
                        ffmpeg=settings.ffmpeg_path)  # fmt: skip

        words_path = live.segment_words_path(settings, ls.id, seg.index)
        if not words_path.is_file():
            ctx.progress(5, f"Segmento {seg.index}: transcrevendo {seg.duration_s:.0f}s…")
            lang = live.session_language(settings, ls.id)
            transcriber, keep = _live_transcriber(settings, ctx.log)
            try:
                words = transcriber.transcribe(wav, lang, lambda f: ctx.progress(5 + 70 * f))
                lang = lang or getattr(transcriber, "detected_language", None)
            finally:
                if not keep and hasattr(transcriber, "close"):
                    transcriber.close()
            live.save_segment_words(words_path, [
                (w.start + seg.start_offset_s, w.end + seg.start_offset_s, w.word, w.prob)
                for w in words
            ])  # fmt: skip
            if lang:
                live.save_session_language(settings, ls.id, lang)

        ctx.progress(80, "Calculando a nota com a linha de base da sessão…")
        hist = _history(ctx, ls, seg)
        readers = [WavReader(live.segment_wav(settings, ls.id, g.index)) for g in hist]
        t0 = hist[0].start_offset_s
        t1 = seg.end_offset_s
        hist_words = live.load_words(settings, ls.id, [g.index for g in hist])
        words3 = [(a, b, w) for a, b, w, _p in hist_words]
        chat = [m for m in live.session_chat(settings, ls.id) if t0 <= m.t < t1 + 30]
        with ctx.session() as s:
            run = s.get(AnalysisRun, ls.run_id) if ls.run_id else None
        params = DetectParams.from_payload(run.params if run else None, settings.detect_preset)
        events = None
        if settings.audio_events:
            try:
                model = audio_events.make_model(settings, ctx.log)
                n = int(np.ceil(sum(r.duration for r in readers)))
                events = audio_events.compute_events(_blocks(readers), n, t0, model).series
            except audio_events.AudioEventsUnavailable as e:
                ctx.log(f"Sem eventos de áudio ({e}).")
        feats = compute_features(
            blocks=_blocks(readers), sample_rate=readers[0].sample_rate, t0=t0, words=words3,
            chat=chat or None, lex=load_lexicon(ensure_hype_file(settings.hype_words_path)),
            p=params.features, events=events,
        )  # fmt: skip
        results, _curve, _eff = detect_candidates(feats, words3, params.candidates)
        new_from = max(t0, seg.start_offset_s - settings.live_context_seconds)

        with ctx.session() as s:
            ls = s.get(LiveSession, ls.id)
            touched = merge_into_session(s, ls, results, new_from,
                                         params.candidates.max_duration_s)  # fmt: skip
            seg = s.get(LiveSegment, seg.id)
            seg.status, seg.analyzed_at, seg.error = SegmentStatus.ANALYZED, utcnow(), None
            seg.updated_at = utcnow()
            s.add(seg)
            run = s.get(AnalysisRun, ls.run_id) if ls.run_id else None
            if run is not None:
                run.candidate_count = len(s.exec(select(Candidate.id).where(
                    Candidate.live_session_id == ls.id)).all())  # fmt: skip
                run.has_chat = run.has_chat or bool(chat)
                s.add(run)
            from clipador.worker import queue

            to_score = [c.id for c in touched] + _unscored(s, ls.id, {c.id for c in touched})
            llm = bool(to_score) and _llm_up(settings)
            if llm:  # the AI scores them; auto-clip runs at the end of that job
                queue.enqueue(s, "enrich", ls.source_id,
                              {"run_id": ls.run_id, "candidate_ids": to_score,
                               "live_session_id": ls.id},
                              dedupe=False, priority=live.ANALYZE_PRIORITY)  # fmt: skip
            clipped = [] if llm else apply_auto_clip(s, settings, ls)
            live.retention_sweep(s, settings, ls)
            s.commit()
        elapsed = time.monotonic() - t_start
        ctx.progress(
            100,
            f"Segmento {seg.index} ({seg.start_offset_s / 60:.1f}–{seg.end_offset_s / 60:.1f} min) "
            f"analisado em {elapsed:.0f}s: {len(touched)} candidato(s) novo(s)/atualizado(s)"
            + (f", {len(clipped)} acima da nota mínima (prévia na fila)" if clipped else "")
            + (", IA na fila" if llm else "")
            + ".",
        )


class LiveFinalizeStage(Stage):
    """Join the captured segments into one video: the live becomes a normal VOD source."""

    name = FINALIZE
    needs_gpu = False
    max_attempts = 2

    def run(self, ctx: StageContext) -> None:
        settings = ctx.settings
        with ctx.session() as s:
            ls = s.get(LiveSession, ctx.payload.get("session_id"))
            if ls is None:
                raise ClipadorError("Sessão de live não existe mais.")
            src = s.get(Source, ls.source_id)
            segs = [g for g in live.segments(s, ls.id) if g.status != SegmentStatus.RECORDING]
        if src is None:
            raise ClipadorError("A fonte desta live não existe mais.")
        if src.file_path and settings.abs_path(src.file_path).is_file():
            ctx.progress(100, "O vídeo da live já estava montado.")
            return
        on_disk = [g for g in segs if g.path and settings.abs_path(g.path).is_file()]
        if not segs:
            self._mark(ctx, src.id, None, "A live terminou sem nenhum segmento gravado.")
            ctx.progress(100, "Nada foi gravado nesta live.")
            return
        if len(on_disk) != len(segs):
            missing = len(segs) - len(on_disk)
            note = (
                "VOD completo indisponível: segmentos antigos já foram apagados pela retenção "
                f"({missing} de {len(segs)}). Os cortes continuam valendo."
            )
            self._mark(ctx, src.id, None, note)
            ctx.progress(100, note)
            return
        out = settings.data_dir / "vods" / str(src.id) / "video.mp4"
        ctx.progress(10, f"Juntando {len(on_disk)} segmento(s) num vídeo só (sem reencodar)…")
        live.concat_files([settings.abs_path(g.path) for g in on_disk], out, settings.ffmpeg_path)
        self._mark(ctx, src.id, out, None)
        with ctx.session() as s:
            ls = s.get(LiveSession, ls.id)
            expired = live.retention_sweep(s, settings, ls)
            s.commit()
        ctx.progress(
            100,
            f"Vídeo da live montado ({out.name}); {len(expired)} segmento(s) apagado(s). "
            "Já dá para clicar em Analisar para a análise completa.",
        )

    def _mark(
        self, ctx: StageContext, source_id: int, video: Path | None, note: str | None
    ) -> None:
        from clipador.errors import ToolError
        from clipador.services import ffmpeg

        with ctx.session() as s:
            src = s.get(Source, source_id)
            src.is_live = False
            if video is not None:
                src.file_path = ctx.settings.rel_path(video)
                with contextlib.suppress(ToolError):
                    info = ffmpeg.probe(video, ctx.settings.ffprobe_path)
                    src.media_info, src.duration_seconds = info.to_dict(), info.duration
                src.kind, src.status = SourceKind.VOD, SourceStatus.READY
            else:
                src.status = SourceStatus.READY
                if note:
                    src.warning = f"{src.warning} · {note}" if src.warning else note
            src.updated_at = utcnow()
            s.add(src)
            s.commit()
