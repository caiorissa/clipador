"""detect: features (cached in features.npz) -> interest curve -> ranked candidates.

Changing only candidate params/weights re-uses cached features: runs in well under a second.
Each run that produces results creates an AnalysisRun; re-running with identical params is a
no-op (idempotent).
"""

from __future__ import annotations

import time

import numpy as np
from sqlmodel import col, select

from clipador.detection.candidates import detect_candidates
from clipador.detection.features import Features, compute_features
from clipador.detection.hype import ensure_hype_file, load_lexicon
from clipador.detection.params import DetectParams
from clipador.errors import ClipadorError
from clipador.models import AnalysisRun, ArtifactStatus, Candidate
from clipador.pipeline.base import Stage, StageContext
from clipador.pipeline.stages.extract_audio import load_source
from clipador.pipeline.stages.fetch_chat import STAGE as CHAT_STAGE
from clipador.pipeline.stages.fetch_chat import chat_params_hash
from clipador.pipeline.stages.transcribe import STAGE as TRANSCRIBE_STAGE
from clipador.pipeline.stages.transcribe import audio_artifact, transcript_params_hash
from clipador.platforms.chat import load_chat
from clipador.platforms.registry import get_adapter
from clipador.services.artifacts import (
    file_hash,
    get_artifact,
    params_hash,
    save_artifact,
    valid_artifact,
)
from clipador.services.audio import WavReader
from clipador.services.downloader import YtDlp
from clipador.services.transcriber import load_transcript

STAGE = "detect"
FEATURES = "features"
VERSION = 1


class DetectInputs:
    """Resolves (and validates) everything detect needs for a source."""

    def __init__(self, ctx: StageContext):
        settings = ctx.settings
        self.source = load_source(ctx)
        self.audio = audio_artifact(ctx, self.source)
        tr_hash = transcript_params_hash(ctx, self.audio.params_hash)
        self.transcript = valid_artifact(
            ctx.engine, settings, self.source.id, TRANSCRIBE_STAGE, tr_hash
        )
        if self.transcript is None:
            raise ClipadorError(
                "Esta fonte ainda não foi transcrita (ou a configuração do Whisper mudou). "
                "Clique em Analisar para rodar a cadeia completa."
            )
        provider = get_adapter(self.source.platform, YtDlp(settings)).chat_provider()
        chat = get_artifact(ctx.engine, self.source.id, CHAT_STAGE)
        ok = (
            chat is not None
            and chat.status == ArtifactStatus.OK
            and chat.params_hash == chat_params_hash(self.source, provider)
            and all(settings.abs_path(f).is_file() for f in chat.files.values())
        )
        self.chat = chat if ok else None
        self.chat_skip_reason = None if ok else (chat.message if chat else "chat não buscado")
        self.params = DetectParams.from_payload(
            ctx.payload.get("detect_params"), settings.detect_preset
        )
        hype_path = ensure_hype_file(settings.hype_words_path)
        self.hype_path = hype_path
        self.features_hash = params_hash(
            {
                "v": VERSION,
                "audio": self.audio.params_hash,
                "transcript": self.transcript.params_hash,
                "chat": self.chat.params_hash if self.chat else None,
                "hype": file_hash(hype_path),
                "features": self.params.features.model_dump(),
            }
        )
        self.run_hash = params_hash(
            {"features": self.features_hash, "candidates": self.params.candidates.model_dump()}
        )


def latest_run(ctx: StageContext, source_id: int) -> AnalysisRun | None:
    with ctx.session() as s:
        return s.exec(
            select(AnalysisRun)
            .where(AnalysisRun.source_id == source_id)
            .order_by(col(AnalysisRun.id).desc())
        ).first()


class DetectStage(Stage):
    name = STAGE
    needs_gpu = False
    max_attempts = 2

    def is_done(self, ctx: StageContext) -> bool:
        inputs = DetectInputs(ctx)
        run = latest_run(ctx, inputs.source.id)
        return run is not None and run.params_hash == inputs.run_hash

    def run(self, ctx: StageContext) -> None:
        t_start = time.monotonic()
        settings = ctx.settings
        inp = DetectInputs(ctx)
        source = inp.source
        if inp.chat is None:
            ctx.log(f"Sem dados de chat ({inp.chat_skip_reason}): nota só com áudio e fala.")

        words_full, _meta = load_transcript(settings.abs_path(inp.transcript.files["transcript"]))
        words = [(w.start, w.end, w.word) for w in words_full]

        feat_art = valid_artifact(ctx.engine, settings, source.id, FEATURES, inp.features_hash)
        if feat_art is not None:
            feats = Features.load(settings.abs_path(feat_art.files["features"]))
            ctx.log("Features em cache; recalculando só a nota e os candidatos.")
        else:
            ctx.progress(5, "Calculando features por segundo (áudio, fala, chat)…")
            reader = WavReader(settings.abs_path(inp.audio.files["audio"]))
            chat_msgs = (
                load_chat(settings.abs_path(inp.chat.files["chat"])).messages if inp.chat else None
            )
            t_feat = time.monotonic()
            feats = compute_features(
                blocks=reader.blocks(60.0),
                sample_rate=reader.sample_rate,
                t0=float(inp.audio.summary.get("offset", 0.0)),
                words=words,
                chat=chat_msgs,
                lex=load_lexicon(inp.hype_path),
                p=inp.params.features,
            )
            out = settings.source_work_dir(source.id) / "features.npz"
            feats.save(out)
            save_artifact(
                ctx.engine, source.id, FEATURES, inp.features_hash,
                files={"features": settings.rel_path(out)},
                summary={"seconds": feats.n, "has_chat": feats.has_chat},
                elapsed=time.monotonic() - t_feat,
            )  # fmt: skip
        ctx.progress(70, "Procurando picos e montando candidatos…")

        results, curve, eff = detect_candidates(feats, words, inp.params.candidates)
        elapsed = time.monotonic() - t_start
        params = inp.params.model_dump()
        params["effective_weights"] = {k: round(v, 3) for k, v in eff.items()}
        with ctx.session() as s:
            run = AnalysisRun(
                source_id=source.id,
                params_hash=inp.run_hash,
                params=params,
                candidate_count=len(results),
                elapsed_seconds=round(elapsed, 3),
                has_chat=feats.has_chat,
            )
            s.add(run)
            s.flush()
            for rank, r in enumerate(results, 1):
                s.add(
                    Candidate(
                        source_id=source.id,
                        analysis_run_id=run.id,
                        rank=rank,
                        start=r.start,
                        end=r.end,
                        peak_t=r.peak_t,
                        score=r.score,
                        breakdown=r.breakdown,
                        transcript_snippet=r.snippet,
                    )
                )
            s.commit()
            run_id = run.id
        summary = {
            "run_id": run_id,
            "candidates": len(results),
            "max_curve": float(np.max(curve, initial=0)),
        }
        save_artifact(ctx.engine, source.id, STAGE, inp.run_hash, summary=summary, elapsed=elapsed)
        ctx.progress(100, f"Análise #{run_id}: {len(results)} candidatos em {elapsed:.1f}s.")
