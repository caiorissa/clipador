"""Ingest: metadata + download (yt-dlp, <=1080p mp4) + ffprobe + thumbnail.

Idempotent: metadata is fetched only once, an existing intact file skips the download, and an
interrupted download resumes from yt-dlp's partial files. start/end offsets are only validated
and kept for later stages; the file is never cut here.
"""

from __future__ import annotations

from pathlib import Path

from clipador.models import Source, SourceStatus, Streamer
from clipador.pipeline.base import Stage, StageContext
from clipador.platforms.registry import get_adapter
from clipador.services import ffmpeg
from clipador.services.downloader import THUMB_BASENAME, YtDlp, find_video_file
from clipador.services.sources import channel_warning
from clipador.utils import format_duration, utcnow

# Progress budget: metadata 0-3, download 3-95, probe + thumbnail 95-100.
_DL_START, _DL_END = 3.0, 95.0


def _duration_matches(actual: float, expected: float | None) -> bool:
    if not expected:
        return True
    return abs(actual - expected) <= max(5.0, expected * 0.02)


class IngestStage(Stage):
    name = "ingest"
    needs_gpu = False
    max_attempts = 4

    # --- helpers ---------------------------------------------------------------------------
    def _dest_dir(self, ctx: StageContext, source: Source) -> Path:
        return ctx.settings.vods_dir / str(source.id)

    def _intact(self, ctx: StageContext, video: Path, expected: float | None) -> bool:
        try:
            info = ffmpeg.probe(video, ctx.settings.ffprobe_path)
        except Exception as e:  # corrupt/partial file -> re-download
            ctx.log(f"Arquivo existente não passou na verificação: {e}")
            return False
        return _duration_matches(info.duration, expected)

    def _update_source(self, ctx: StageContext, **fields) -> Source:
        with ctx.session() as s:
            source = s.get(Source, ctx.source_id)
            for k, v in fields.items():
                setattr(source, k, v)
            source.updated_at = utcnow()
            s.add(source)
            s.commit()
            return source

    # --- Stage API ---------------------------------------------------------------------------
    def is_done(self, ctx: StageContext) -> bool:
        with ctx.session() as s:
            source = s.get(Source, ctx.source_id)
            if source is None or source.status != SourceStatus.READY or not source.file_path:
                return False
            video = ctx.settings.abs_path(source.file_path)
            expected = source.duration_seconds
        return video.is_file() and self._intact(ctx, video, expected)

    def run(self, ctx: StageContext) -> None:
        settings = ctx.settings
        with ctx.session() as s:
            source = s.get(Source, ctx.source_id)
            if source is None:
                raise ValueError(f"fonte {ctx.source_id} não existe")
            streamer = s.get(Streamer, source.streamer_id)
            _ = streamer.channels  # load while session is open
        adapter = get_adapter(source.platform, YtDlp(settings))
        source = self._update_source(ctx, status=SourceStatus.DOWNLOADING, error=None)

        # 1) metadata (once)
        expected = source.duration_seconds
        if source.title is None:
            ctx.progress(1, f"Buscando dados do vídeo: {source.url}")
            meta = adapter.fetch_metadata(source.url)
            warning = source.warning or channel_warning(
                streamer, source.platform, meta.channel_candidates
            )
            if warning:
                ctx.log(f"AVISO: {warning}")
            source = self._update_source(
                ctx,
                title=meta.title,
                channel_name=meta.channel_name,
                published_at=meta.published_at,
                duration_seconds=meta.duration_seconds,
                thumbnail_url=meta.thumbnail_url,
                warning=warning,
            )
            expected = meta.duration_seconds
            ctx.log(f"'{meta.title}' · {format_duration(meta.duration_seconds)}")
        ctx.progress(_DL_START)

        # 2) download (skipped if an intact file is already there)
        dest = self._dest_dir(ctx, source)
        video = find_video_file(dest) if dest.is_dir() else None
        if video and self._intact(ctx, video, expected):
            ctx.log("Arquivo já baixado e íntegro; pulando o download.")
        else:
            if video:
                video.unlink()
            ctx.log("Baixando (até 1080p, mp4)…")

            def on_progress(pct: float, message: str | None) -> None:
                ctx.progress(_DL_START + pct / 100.0 * (_DL_END - _DL_START), message)

            video = adapter.download(source.url, dest, on_progress, ctx.log).video_path
        ctx.progress(_DL_END, "Lendo informações do arquivo (ffprobe)…")

        # 3) probe
        info = ffmpeg.probe(video, settings.ffprobe_path)
        ctx.log(
            f"{info.width}x{info.height} · {info.fps or '?'} fps · {info.video_codec}/"
            f"{info.audio_codec} · {format_duration(info.duration)}"
        )

        # 4) thumbnail: the platform's, or a frame grabbed by ffmpeg
        thumb = dest / f"{THUMB_BASENAME}.jpg"
        if not thumb.is_file():
            ffmpeg.make_thumbnail(
                video, thumb, min(info.duration * 0.1, 30.0), settings.ffmpeg_path
            )

        # 5) offsets sanity (kept for later stages, file not cut)
        warning = source.warning
        if source.start_offset is not None and source.start_offset >= info.duration:
            msg = (
                f"O início do trecho ({format_duration(source.start_offset)}) passa da duração "
                f"do vídeo ({format_duration(info.duration)})."
            )
            ctx.log(f"AVISO: {msg}")
            warning = f"{warning} {msg}".strip() if warning else msg

        self._update_source(
            ctx,
            file_path=settings.rel_path(video),
            thumbnail_path=settings.rel_path(thumb),
            duration_seconds=info.duration,
            media_info=info.to_dict(),
            status=SourceStatus.READY,
            warning=warning,
            error=None,
        )
        ctx.progress(100, "Pronto.")

    def on_failure(self, ctx: StageContext, message: str, final: bool) -> None:
        if final:
            self._update_source(ctx, status=SourceStatus.FAILED, error=message)
        else:
            self._update_source(
                ctx, status=SourceStatus.PENDING, error=f"{message} (vai tentar de novo)"
            )
