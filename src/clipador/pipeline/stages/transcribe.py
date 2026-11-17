"""transcribe: Whisper on the GPU (CUDA on the PC, MLX on a Mac), in ~10 min chunks with overlap.

Each finished chunk is saved to disk, so a crash/kill resumes from the last finished chunk.
Output: work/<id>/transcript.json.gz with word timestamps relative to the ORIGINAL video.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import time
from dataclasses import asdict

from clipador.errors import ClipadorError
from clipador.pipeline.base import Stage, StageContext
from clipador.pipeline.stages.extract_audio import STAGE as AUDIO_STAGE
from clipador.pipeline.stages.extract_audio import audio_params_hash, load_source
from clipador.services import transcriber as tx
from clipador.services.artifacts import params_hash, save_artifact, valid_artifact
from clipador.services.audio import WavReader
from clipador.services.gpu import vram_used_mb

STAGE = "transcribe"
VERSION = 1


def audio_artifact(ctx: StageContext, source):
    art = valid_artifact(
        ctx.engine, ctx.settings, source.id, AUDIO_STAGE, audio_params_hash(source)
    )
    if art is None:
        raise ClipadorError(
            "O áudio desta fonte ainda não foi extraído (ou mudou). Clique em Analisar de novo."
        )
    return art


def transcript_params_hash(ctx: StageContext, audio_hash: str) -> str:
    s = ctx.settings
    backend = tx.resolve_backend(s)
    extra = {} if backend == "faster-whisper" else {"backend": backend}  # old hashes stay valid
    return params_hash(
        {
            "v": VERSION,
            "audio": audio_hash,
            **extra,
            "model": tx.resolve_model(s, backend),
            "compute": s.whisper_compute_type if backend == "faster-whisper" else "float16",
            "language": s.language,
            "beam": s.whisper_beam_size,
            "vad": s.whisper_vad_filter,
            "chunk": s.transcribe_chunk_seconds,
            "overlap": s.transcribe_chunk_overlap_seconds,
        }
    )


class TranscribeStage(Stage):
    name = STAGE
    needs_gpu = True
    max_attempts = 3

    def _hash(self, ctx: StageContext) -> tuple[object, object, str]:
        source = load_source(ctx)
        audio_art = audio_artifact(ctx, source)
        return source, audio_art, transcript_params_hash(ctx, audio_art.params_hash)

    def is_done(self, ctx: StageContext) -> bool:
        source, _audio, phash = self._hash(ctx)
        return bool(valid_artifact(ctx.engine, ctx.settings, source.id, STAGE, phash))

    def run(self, ctx: StageContext) -> None:
        settings = ctx.settings
        source, audio_art, phash = self._hash(ctx)
        work = settings.source_work_dir(source.id)
        wav_path = settings.abs_path(audio_art.files["audio"])
        offset = float(audio_art.summary.get("offset", 0.0))
        reader = WavReader(wav_path)
        chunks = tx.plan_chunks(
            reader.duration,
            settings.transcribe_chunk_seconds,
            settings.transcribe_chunk_overlap_seconds,
        )
        chunk_dir = work / f"transcribe_{phash}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        language = None if settings.language == "auto" else settings.language

        done = [c for c in chunks if (chunk_dir / f"chunk_{c.index:04d}.json").is_file()]
        if done:
            ctx.log(f"Retomando: {len(done)}/{len(chunks)} blocos já transcritos.")
            # keep the language detected in the first run consistent
            first = json.loads((chunk_dir / f"chunk_{done[0].index:04d}.json").read_text("utf-8"))
            language = language or first.get("language")
        ctx.log(
            f"Transcrevendo {reader.duration / 60:.1f} min em {len(chunks)} bloco(s) de "
            f"{settings.transcribe_chunk_seconds // 60} min."
        )

        t0 = time.monotonic()
        transcribed_s = 0.0
        transcriber = None
        try:
            for c in chunks:
                out = chunk_dir / f"chunk_{c.index:04d}.json"
                if out.is_file():
                    continue
                if transcriber is None:
                    transcriber = tx.make_transcriber(settings, ctx.log)
                tmp_wav = chunk_dir / f"chunk_{c.index:04d}.wav"
                reader.write_slice(c.read_start, c.read_end, tmp_wav)
                base = c.index / len(chunks) * 100

                def on_progress(frac: float, base=base) -> None:
                    ctx.progress(base + frac * 100 / len(chunks))

                words = transcriber.transcribe(tmp_wav, language, on_progress)
                language = language or getattr(transcriber, "detected_language", None)
                shifted = [
                    asdict(tx.Word(w.start + c.read_start + offset, w.end + c.read_start + offset,
                                   w.word, w.prob))
                    for w in words
                ]  # fmt: skip
                tmp_json = out.with_suffix(".tmp")
                tmp_json.write_text(
                    json.dumps({"language": language, "words": shifted}, ensure_ascii=False),
                    "utf-8",
                )
                tmp_json.replace(out)
                transcribed_s += c.read_end - c.read_start
                rtf = tx.realtime_factor(transcribed_s, t0)
                ctx.log(
                    f"Bloco {c.index + 1}/{len(chunks)} pronto: {len(words)} palavras "
                    f"({rtf:.1f}x tempo real)."
                )
                with contextlib.suppress(OSError):
                    tmp_wav.unlink()
        finally:
            if transcriber is not None and hasattr(transcriber, "close"):
                transcriber.close()
                vram = vram_used_mb()
                if vram is not None:
                    ctx.log(f"Whisper descarregado; VRAM em uso agora: {vram} MiB.")

        parts = []
        for c in chunks:
            data = json.loads((chunk_dir / f"chunk_{c.index:04d}.json").read_text("utf-8"))
            words = [tx.Word(**w) for w in data["words"]]
            shifted_chunk = tx.Chunk(
                c.index,
                c.start + offset,
                c.end + offset,
                c.read_start + offset,
                c.read_end + offset,
            )
            parts.append((shifted_chunk, words))
        merged = tx.merge_chunk_words(parts)
        out_path = work / "transcript.json.gz"
        elapsed = time.monotonic() - t0
        meta = {
            "model": tx.resolve_model(settings),
            "backend": tx.resolve_backend(settings),
            "device": getattr(transcriber, "device", None),
            "compute_type": getattr(transcriber, "compute_type", None),
            "language": language,
            "duration": reader.duration,
            "offset": offset,
        }
        tx.save_transcript(out_path, merged, meta)
        summary = {**meta, "words": len(merged), "elapsed": round(elapsed, 1)}
        if transcribed_s > 0:
            summary["realtime_factor"] = round(transcribed_s / max(elapsed, 1e-6), 2)
        save_artifact(
            ctx.engine,
            source.id,
            STAGE,
            phash,
            files={"transcript": settings.rel_path(out_path)},
            summary=summary,
            elapsed=elapsed,
        )
        shutil.rmtree(chunk_dir, ignore_errors=True)
        ctx.progress(100, f"Transcrição pronta: {len(merged)} palavras (idioma: {language}).")
