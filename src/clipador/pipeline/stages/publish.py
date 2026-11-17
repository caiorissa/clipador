"""publish_youtube (upload one approved clip) and youtube_metrics (refresh the numbers)."""

from __future__ import annotations

from clipador.errors import ClipadorError
from clipador.models import Publication, PublicationStatus, Render
from clipador.pipeline.base import Stage, StageContext
from clipador.services import publishing, youtube
from clipador.utils import utcnow


class PublishYouTubeStage(Stage):
    name = "publish_youtube"
    needs_gpu = False
    max_attempts = 3  # network hiccups / 5xx are retried; quota/auth errors are not

    def is_done(self, ctx: StageContext) -> bool:
        with ctx.session() as s:
            p = s.get(Publication, ctx.payload.get("publication_id"))
        return p is not None and p.status == PublicationStatus.PUBLISHED

    def on_failure(self, ctx: StageContext, message: str, final: bool) -> None:
        with ctx.session() as s:
            p = s.get(Publication, ctx.payload.get("publication_id"))
            if p is not None and p.status != PublicationStatus.PUBLISHED:
                p.status = PublicationStatus.FAILED if final else PublicationStatus.QUEUED
                p.error, p.updated_at = message[:1000], utcnow()
                s.add(p)
                s.commit()

    def run(self, ctx: StageContext) -> None:
        settings = ctx.settings
        with ctx.session() as s:
            p = s.get(Publication, ctx.payload.get("publication_id"))
            if p is None:
                raise ClipadorError("Publicação não encontrada.")
            r = s.get(Render, p.render_id)
            p.status, p.error, p.updated_at = PublicationStatus.UPLOADING, None, utcnow()
            s.add(p)
            s.commit()
        video = settings.abs_path(r.path) if r and r.path else None
        if video is None or not video.is_file():
            raise ClipadorError("O arquivo do corte final não existe mais; renderize de novo.")
        publish_at = p.publish_at.isoformat().replace("+00:00", "Z") if p.publish_at else None
        meta = youtube.video_meta(p.title, p.description, p.tags, p.privacy, publish_at)
        ctx.log(f"Enviando '{p.title}' ({video.stat().st_size / 1e6:.1f} MB, "
                f"{youtube.PRIVACY_LABELS[p.privacy]}"
                + (f", agendado para {p.publish_at.astimezone():%d/%m %H:%M}" if p.publish_at
                   else "") + ")…")  # fmt: skip
        vid = youtube.YouTube(settings).upload(video, meta, lambda f: ctx.progress(99 * f))
        with ctx.session() as s:
            p = s.get(Publication, p.id)
            p.status, p.external_id = PublicationStatus.PUBLISHED, vid
            p.url = f"https://youtube.com/shorts/{vid}"
            p.updated_at = utcnow()
            s.add(p)
            s.commit()
        ctx.progress(100, f"Publicado: {p.url}")


class YouTubeMetricsStage(Stage):
    name = "youtube_metrics"
    needs_gpu = False
    max_attempts = 2

    def run(self, ctx: StageContext) -> None:
        with ctx.session() as s:
            n = publishing.refresh_metrics(s, ctx.settings)
            s.commit()
        ctx.progress(100, f"Números de {n} vídeo(s) atualizados.")
