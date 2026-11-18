"""publish (upload one approved clip to its account: YouTube or TikTok) and youtube_metrics
(refresh the numbers of every account)."""

from __future__ import annotations

from clipador.errors import ClipadorError
from clipador.models import Publication, PublicationStatus, Render, SocialAccount
from clipador.pipeline.base import Stage, StageContext
from clipador.services import accounts, publishing, tiktok, youtube
from clipador.utils import utcnow


class PublishStage(Stage):
    name = "publish"
    needs_gpu = False
    max_attempts = 3  # network hiccups / 5xx are retried; quota/auth errors are not

    def is_done(self, ctx: StageContext) -> bool:
        with ctx.session() as s:
            p = s.get(Publication, ctx.payload.get("publication_id"))
        return p is not None and p.status in (
            PublicationStatus.PUBLISHED,
            PublicationStatus.CANCELED,
        )

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
            acc = s.get(SocialAccount, p.account_id) if p.account_id else None
            if acc is None:
                acc = accounts.import_legacy_youtube(s, settings)
                if acc is None:
                    raise ClipadorError("A conta desta postagem não existe mais; conecte de novo.")
                p.account_id = acc.id
            if acc.paused:
                raise ClipadorError(f"A conta {acc.name} está pausada.", retryable=True)
            r = s.get(Render, p.render_id)
            p.status, p.error, p.updated_at = PublicationStatus.UPLOADING, None, utcnow()
            s.add(p)
            s.commit()
        video = settings.abs_path(r.path) if r and r.path else None
        if video is None or not video.is_file():
            raise ClipadorError("O arquivo do corte final não existe mais; renderize de novo.")
        client = accounts.client_for(ctx.engine, settings, acc)
        size = f"{video.stat().st_size / 1e6:.1f} MB"
        if acc.platform == "tiktok":
            privacy = tiktok.FROM_GENERIC.get(p.privacy, "SELF_ONLY")
            ctx.log(f"Enviando ao TikTok ({acc.name}, {size})…")
            publish_id, used = client.upload(
                video, p.description, privacy, lambda f: ctx.progress(80 * f)
            )
            if used == tiktok.INBOX:
                pass
            elif used != privacy:
                ctx.log(
                    f"AVISO: o TikTok só permitiu '{tiktok.PRIVACY_LABELS.get(used, used)}' "
                    "(app ainda não aprovado pelo TikTok?)."
                )
            ctx.progress(85, "Esperando o TikTok processar…")
            post_id = client.wait_published(publish_id)
            if used == tiktok.INBOX:
                ctx.log(
                    "Rascunho enviado para a caixa de entrada do app do TikTok: abra a "
                    "notificação no celular, cole a legenda abaixo e publique."
                )
                ctx.log(f"Legenda: {p.description}")
            ext, url = post_id or publish_id, None
            if post_id:
                url = f"https://www.tiktok.com/@{acc.name}/video/{post_id}"
            privacy_used = "public" if used == "PUBLIC_TO_EVERYONE" else "private"
            if used == tiktok.INBOX:
                privacy_used = "private"  # unknown until you post it in the app
        else:
            publish_at = p.publish_at
            if publish_at is None and p.auto and p.scheduled_for:
                sched = accounts.aware(p.scheduled_for)
                if (sched - utcnow()).total_seconds() > 15 * 60:  # uploaded ahead of its slot
                    publish_at = sched
            at = publish_at.isoformat().replace("+00:00", "Z") if publish_at else None
            meta = youtube.video_meta(p.title, p.description, p.tags, p.privacy, at)
            ctx.log(
                f"Enviando ao YouTube ({acc.name}, '{p.title}', {size}"
                + (f", público às {publish_at.astimezone():%d/%m %H:%M}" if publish_at else "")
                + ")…"
            )
            ext = client.upload(video, meta, lambda f: ctx.progress(99 * f))
            url, privacy_used = f"https://youtube.com/shorts/{ext}", p.privacy
            with ctx.session() as s:
                row = s.get(Publication, p.id)
                row.publish_at = publish_at
                s.add(row)
                s.commit()
        with ctx.session() as s:
            row = s.get(Publication, p.id)
            row.status, row.external_id, row.url = PublicationStatus.PUBLISHED, ext, url
            row.privacy, row.updated_at = privacy_used, utcnow()
            s.add(row)
            s.commit()
        ctx.progress(100, f"Publicado em {acc.name}" + (f": {url}" if url else "."))


class PublishYouTubeStage(PublishStage):
    """Job type of the first version (jobs already in the queue keep working)."""

    name = "publish_youtube"


class YouTubeMetricsStage(Stage):
    name = "youtube_metrics"
    needs_gpu = False
    max_attempts = 2

    def run(self, ctx: StageContext) -> None:
        with ctx.session() as s:
            n = publishing.refresh_metrics(s, ctx.settings, ctx.engine)
            s.commit()
        ctx.progress(100, f"Números de {n} vídeo(s) atualizados.")
