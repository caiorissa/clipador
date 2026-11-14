"""extract_audio: video -> work/<id>/audio.wav (16 kHz mono s16), honoring start/end offsets.

All later stages report times relative to the ORIGINAL video: they add `offset` (stored in the
artifact summary) to times measured inside the WAV.
"""

from __future__ import annotations

import time

from clipador.errors import ClipadorError
from clipador.models import Source, SourceStatus
from clipador.pipeline.base import Stage, StageContext
from clipador.services import audio
from clipador.services.artifacts import params_hash, save_artifact, valid_artifact
from clipador.services.facecam import autodetect_in_stage

STAGE = "extract_audio"
VERSION = 1


def audio_params_hash(source: Source) -> str:
    return params_hash(
        {
            "v": VERSION,
            "file": source.file_path,
            "start": source.start_offset,
            "end": source.end_offset,
            "sr": audio.SAMPLE_RATE,
        }
    )


def load_source(ctx: StageContext) -> Source:
    with ctx.session() as s:
        source = s.get(Source, ctx.source_id)
    if source is None:
        raise ClipadorError(f"Fonte #{ctx.source_id} não existe.")
    if source.status != SourceStatus.READY or not source.file_path:
        raise ClipadorError(
            "A fonte ainda não foi baixada. Espere o download terminar antes de analisar."
        )
    return source


class ExtractAudioStage(Stage):
    name = STAGE
    max_attempts = 2

    def is_done(self, ctx: StageContext) -> bool:
        source = load_source(ctx)
        return bool(
            valid_artifact(ctx.engine, ctx.settings, source.id, STAGE, audio_params_hash(source))
        )

    def run(self, ctx: StageContext) -> None:
        t_start = time.monotonic()
        source = load_source(ctx)
        settings = ctx.settings
        video = settings.abs_path(source.file_path)
        out = settings.source_work_dir(source.id) / "audio.wav"
        start = source.start_offset or 0.0
        ctx.log(
            "Extraindo áudio (16 kHz mono)"
            + (f" a partir de {start:.0f}s" if start else "")
            + (f" até {source.end_offset:.0f}s" if source.end_offset is not None else "")
            + "…"
        )
        audio.extract_wav(
            video,
            out,
            start=source.start_offset,
            end=source.end_offset,
            duration_hint=source.duration_seconds,
            progress_cb=lambda p: ctx.progress(p),
            ffmpeg=settings.ffmpeg_path,
        )
        reader = audio.WavReader(out)
        save_artifact(
            ctx.engine,
            source.id,
            STAGE,
            audio_params_hash(source),
            files={"audio": settings.rel_path(out)},
            summary={
                "offset": start,
                "duration": reader.duration,
                "sample_rate": reader.sample_rate,
            },
            elapsed=time.monotonic() - t_start,
        )
        # Sources downloaded before step 5C (or whose streamer had its box cleared) get their
        # facecam detected when analyzed. No-op when the streamer already has a box.
        autodetect_in_stage(ctx)
        ctx.progress(100, f"Áudio extraído: {reader.duration / 60:.1f} min.")
