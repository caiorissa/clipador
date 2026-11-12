"""fetch_chat: chat replay via the platform's ChatProvider -> work/<id>/chat.jsonl.gz.

No chat is NOT a failure: the artifact is saved as `skipped` with the reason, and `detect`
renormalizes its weights without the chat signal.
"""

from __future__ import annotations

import logging
import time

from clipador.models import ArtifactStatus
from clipador.pipeline.base import Stage, StageContext
from clipador.pipeline.stages.extract_audio import load_source
from clipador.platforms.base import ChatUnavailable
from clipador.platforms.chat import save_chat
from clipador.platforms.registry import get_adapter
from clipador.services.artifacts import get_artifact, params_hash, save_artifact, valid_artifact
from clipador.services.downloader import YtDlp

log = logging.getLogger(__name__)

STAGE = "fetch_chat"
VERSION = 1


def chat_params_hash(source, provider) -> str:
    return params_hash(
        {
            "v": VERSION,
            "platform": source.platform,
            "kind": source.kind,
            "id": source.external_id,
            "provider": getattr(provider, "version", None),
        }
    )


class FetchChatStage(Stage):
    name = STAGE
    max_attempts = 2

    def _provider(self, ctx: StageContext, source):
        return get_adapter(source.platform, YtDlp(ctx.settings)).chat_provider()

    def is_done(self, ctx: StageContext) -> bool:
        source = load_source(ctx)
        phash = chat_params_hash(source, self._provider(ctx, source))
        art = valid_artifact(ctx.engine, ctx.settings, source.id, STAGE, phash)
        # A skipped chat is retried on the next analysis (e.g. after installing the tool).
        return art is not None and art.status == ArtifactStatus.OK

    def run(self, ctx: StageContext) -> None:
        t0 = time.monotonic()
        source = load_source(ctx)
        provider = self._provider(ctx, source)
        phash = chat_params_hash(source, provider)
        work = ctx.settings.source_work_dir(source.id)

        def skip(reason: str, detail: str | None = None) -> None:
            ctx.log(f"Chat pulado: {reason}")
            if detail:
                log.info("chat detail (source %s): %s", source.id, detail[-2000:])
            save_artifact(
                ctx.engine, source.id, STAGE, phash, status=ArtifactStatus.SKIPPED,
                message=reason, elapsed=time.monotonic() - t0,
            )  # fmt: skip
            ctx.progress(100)

        if provider is None:
            return skip("esta plataforma não tem provedor de chat.")
        ctx.log("Baixando o chat…")
        try:
            chat = provider.fetch(source, work, lambda f: ctx.progress(f * 95))
        except ChatUnavailable as e:
            return skip(e.reason, e.detail)
        if chat is None or not chat.messages:
            return skip("este vídeo não tem replay de chat.")

        out = work / "chat.jsonl.gz"
        save_chat(out, chat)
        prev = get_artifact(ctx.engine, source.id, STAGE)
        if prev is not None and prev.status == ArtifactStatus.SKIPPED:
            ctx.log("Chat agora disponível (antes tinha sido pulado).")
        save_artifact(
            ctx.engine,
            source.id,
            STAGE,
            phash,
            files={"chat": ctx.settings.rel_path(out)},
            summary={"messages": len(chat.messages), "provider": chat.provider},
            elapsed=time.monotonic() - t0,
        )
        ctx.progress(100, f"Chat salvo: {len(chat.messages)} mensagens ({chat.provider}).")
