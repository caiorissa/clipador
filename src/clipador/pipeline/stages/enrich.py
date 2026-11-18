"""enrich: local LLM (Ollama) judges the top-K candidates and writes post texts.

- Runs after `detect`, holds the GPU lock (Whisper is already done and freed its VRAM).
- Per-candidate cache: the input hash covers the transcript window, chat summary, model, prompt
  version and regeneration instruction; an unchanged candidate never calls the LLM again.
- One bad answer marks that candidate `llm_failed` and the stage moves on.
- Without Ollama (or the model) the stage is `skipped`: candidates keep the detection score and
  get simple editable drafts.
- The model is unloaded at the end (keep_alive=0) so it doesn't keep VRAM.
"""

from __future__ import annotations

import time

from sqlmodel import col, select

from clipador.errors import ClipadorError
from clipador.llm import client as llm_client
from clipador.llm.drafts import build_drafts
from clipador.llm.enrich import build_context, build_messages, call_enrich, input_hash, load_prompt
from clipador.llm.schema import EnrichOutput
from clipador.models import (
    AnalysisRun,
    ArtifactStatus,
    Candidate,
    CandidateEnrichment,
    EnrichStatus,
    Source,
    Streamer,
)
from clipador.pipeline.base import Stage, StageContext
from clipador.pipeline.stages.detect import DetectInputs
from clipador.pipeline.stages.extract_audio import load_source
from clipador.platforms.chat import load_chat
from clipador.services.artifacts import params_hash, save_artifact
from clipador.services.enrichment import enrichments_for, rerank_run, save_drafts
from clipador.services.gpu import vram_used_mb
from clipador.services.transcriber import load_transcript
from clipador.utils import utcnow

STAGE = "enrich"


def _output_of(row: CandidateEnrichment) -> EnrichOutput:
    return EnrichOutput.model_validate(
        {
            "scores": row.scores,
            "motivo": row.motivo,
            "precisa_de_contexto": row.precisa_de_contexto,
            "titulos": row.titulos or ["(sem título)"],
            "texto_de_abertura": row.texto_de_abertura,
            "descricao": row.descricao,
            "hashtags": row.hashtags,
            "alertas": row.alertas,
        }
    )


def queue_auto_previews(ctx: StageContext, run_id: int) -> None:
    """End of an analysis: faithful previews of the top N in the background (best effort)."""
    from clipador.services.previews import auto_previews

    if ctx.settings.auto_preview_top_n <= 0:
        return
    with ctx.session() as s:
        job, n = auto_previews(s, ctx.settings, run_id)
        s.commit()
    if job is not None and n:
        ctx.log(f"Prévias fiéis dos {n} melhores na fila (job #{job.id}, prioridade baixa).")


def live_auto_clip(s, settings, live_id: int | None) -> int:
    """After the AI scored live candidates: clip the ones above the session's minimum."""
    if not live_id:
        return 0
    from clipador.models import LiveSession
    from clipador.pipeline.stages.live import apply_auto_clip

    ls = s.get(LiveSession, live_id)
    return len(apply_auto_clip(s, settings, ls)) if ls is not None else 0


class EnrichStage(Stage):
    name = STAGE
    needs_gpu = True  # Ollama runs on the GPU: never alongside Whisper
    max_attempts = 2

    def run(self, ctx: StageContext) -> None:
        settings = ctx.settings
        t_start = time.monotonic()
        if ctx.payload.get("live_session_id"):  # a live still being captured: no full file yet
            with ctx.session() as s:
                source = s.get(Source, ctx.source_id)
            if source is None:
                raise ClipadorError(f"Fonte #{ctx.source_id} não existe.")
        else:
            source = load_source(ctx)
        payload = ctx.payload
        with ctx.session() as s:
            q = select(AnalysisRun).where(AnalysisRun.source_id == source.id)
            if payload.get("run_id"):
                q = q.where(AnalysisRun.id == payload["run_id"])
            run = s.exec(q.order_by(col(AnalysisRun.id).desc())).first()
            if run is None:
                raise ClipadorError("Analise a fonte antes de enriquecer com IA.")
            cands = list(
                s.exec(
                    select(Candidate)
                    .where(Candidate.analysis_run_id == run.id)
                    .order_by(Candidate.rank)
                ).all()
            )
            streamer = s.get(Streamer, source.streamer_id)
        targets = set(payload.get("candidate_ids") or [])
        k = int(payload.get("k") or settings.enrich_k)
        cands = [c for c in cands if c.id in targets] if targets else cands[:k]
        model = payload.get("model") or llm_client.default_model(settings)
        instruction = payload.get("instruction", "")
        overwrite = bool(payload.get("overwrite_edited"))
        game = source.game or settings.default_game

        live_id = payload.get("live_session_id")
        if live_id:  # a live: words/chat of the captured segments (no full transcript yet)
            from clipador.services import live

            words = [(a, b, w) for a, b, w, _p in live.load_words(settings, live_id)]
            chat = live.session_chat(settings, live_id) or None
        else:
            inp = DetectInputs(ctx)
            words_full, _ = load_transcript(settings.abs_path(inp.transcript.files["transcript"]))
            words = [(w.start, w.end, w.word) for w in words_full]
            chat = (
                load_chat(settings.abs_path(inp.chat.files["chat"])).messages if inp.chat else None
            )
        chat_delay = float(run.params.get("features", {}).get("chat_delay_s", 5.0))
        prompt = load_prompt(settings.prompts_path, llm_client.prompt_name(settings))
        phash = params_hash({"run": run.id, "model": model, "prompt": prompt.version})

        def drafts_for(c: Candidate, out: EnrichOutput | None, s) -> str | None:
            """Save drafts inside the open session; returns a log line (log only after commit:
            logging opens another write transaction and SQLite would lock)."""
            kept = save_drafts(
                s, c.id,
                build_drafts(out, streamer=streamer.display_name, credit_text=streamer.credit_text,
                             source_platform=source.platform, source_url=source.url, game=game,
                             snippet=c.transcript_snippet, start=c.eff_start),
                overwrite_edited=overwrite,
            )  # fmt: skip
            if kept:
                return f"#{c.rank}: rascunhos editados por você mantidos ({', '.join(kept)})."
            return None

        client = llm_client.make_llm_client(settings)
        try:
            installed = client.list_models()
            if not llm_client.model_installed(installed, model):
                raise llm_client.LLMUnavailable(
                    f"Modelo '{model}' não está instalado no Ollama "
                    f"(instalados: {', '.join(installed) or 'nenhum'}).",
                    f"Rode `ollama pull {model}` ou escolha outro modelo nos ajustes.",
                )
        except llm_client.LLMUnavailable as e:
            reason = f"{e.message} {e.hint}".strip()
            ctx.log(f"IA pulada: {reason}")
            with ctx.session() as s:
                existing = enrichments_for(s, [c.id for c in cands])
                notes = [drafts_for(c, None, s) for c in cands if c.id not in existing]
                rerank_run(s, s.get(AnalysisRun, run.id), settings)
                live_auto_clip(s, settings, live_id)
                s.commit()
            for note in filter(None, notes):
                ctx.log(note)
            if not targets:
                queue_auto_previews(ctx, run.id)
            save_artifact(ctx.engine, source.id, STAGE, phash, status=ArtifactStatus.SKIPPED,
                          message=reason, elapsed=time.monotonic() - t_start)  # fmt: skip
            ctx.progress(
                100, "Sem IA: rascunhos simples criados; a nota de detecção continua valendo."
            )
            return

        vram0 = vram_used_mb()
        ctx.log(f"IA: modelo {model}, {len(cands)} candidato(s), prompt {prompt.version}. "
                f"VRAM em uso antes: {vram0 if vram0 is not None else '?'} MiB.")  # fmt: skip
        stats = {"ok": 0, "cached": 0, "failed": 0, "latency_ms": 0, "invalid_json": 0}
        try:
            for i, c in enumerate(cands):
                ctx.progress(100 * i / max(1, len(cands)))
                with ctx.session() as s:
                    row = s.exec(
                        select(CandidateEnrichment).where(CandidateEnrichment.candidate_id == c.id)
                    ).first()
                instr = (
                    instruction
                    if (targets and c.id in targets)
                    else (row.instruction if row else "")
                )
                cctx = build_context(
                    words=words, start=c.eff_start, end=c.eff_end, peak=c.peak_t, chat=chat,
                    chat_delay=chat_delay, streamer=streamer.display_name, game=game,
                    platform=source.platform, detection_score=c.score, breakdown=c.breakdown,
                    instruction=instr,
                )  # fmt: skip
                h = input_hash(cctx, model, prompt.version)
                if (
                    row
                    and row.input_hash == h
                    and row.status == EnrichStatus.OK
                    and not payload.get("force")
                ):
                    stats["cached"] += 1
                    continue
                with ctx.session() as s:
                    donor = s.exec(
                        select(CandidateEnrichment).where(
                            CandidateEnrichment.input_hash == h,
                            CandidateEnrichment.status == EnrichStatus.OK,
                        )
                    ).first()
                if donor is not None and not payload.get("force"):
                    out, status, err, latency, tokens = (
                        _output_of(donor),
                        EnrichStatus.OK,
                        None,
                        0,
                        None,
                    )
                    stats["cached"] += 1
                else:
                    res = call_enrich(
                        client, model, build_messages(prompt, cctx),
                        temperature=settings.llm_temperature, num_ctx=settings.llm_num_ctx,
                        keep_alive=settings.llm_keep_alive, timeout=settings.llm_timeout_seconds,
                    )  # fmt: skip
                    out, err, latency, tokens = res.output, res.error, res.latency_ms, res.tokens
                    status = EnrichStatus.OK if res.status == "ok" else EnrichStatus.LLM_FAILED
                    stats["latency_ms"] += latency
                    stats["invalid_json"] += res.invalid_json
                    if status == EnrichStatus.OK:
                        stats["ok"] += 1
                        ctx.log(
                            f"#{c.rank}: nota IA {out.scores.nota_geral} · {latency / 1000:.1f}s"
                            f" · {out.motivo}"
                        )
                    else:
                        stats["failed"] += 1
                        ctx.log(f"#{c.rank}: IA falhou ({err}); fica só a nota de detecção.")
                with ctx.session() as s:
                    row = s.exec(
                        select(CandidateEnrichment).where(CandidateEnrichment.candidate_id == c.id)
                    ).first() or CandidateEnrichment(candidate_id=c.id, source_id=source.id,
                                                     prompt_version=prompt.version, model=model,
                                                     input_hash=h)  # fmt: skip
                    row.prompt_version, row.model, row.input_hash = prompt.version, model, h
                    row.instruction, row.status, row.error = instr, status, err
                    row.latency_ms, row.tokens, row.updated_at = latency, tokens, utcnow()
                    if out is not None:
                        row.scores = out.scores.model_dump()
                        row.motivo = out.motivo
                        row.precisa_de_contexto = out.precisa_de_contexto
                        row.alertas = [a.model_dump() for a in out.alertas]
                        row.titulos = out.titulos
                        row.texto_de_abertura = out.texto_de_abertura
                        row.descricao = out.descricao
                        row.hashtags = out.hashtags
                    s.add(row)
                    note = drafts_for(c, out, s)
                    s.commit()
                if note:
                    ctx.log(note)
        except llm_client.LLMUnavailable as e:
            ctx.log(
                f"A IA ficou indisponível no meio do estágio: {e.message} Os candidatos restantes "
                "ficam sem nota da IA e são avaliados numa próxima vez."
            )
        finally:
            client.unload(model)
            if vram0 is not None:
                time.sleep(1.0)  # give the Ollama server a moment to free the memory
            vram1 = vram_used_mb() if vram0 is not None else None
            ctx.log(f"Modelo descarregado (keep_alive=0). VRAM em uso agora: "
                    f"{vram1 if vram1 is not None else '?'} MiB.")  # fmt: skip

        with ctx.session() as s:
            rerank_run(s, s.get(AnalysisRun, run.id), settings)
            clipped = live_auto_clip(s, settings, live_id)
            s.commit()
        if clipped:
            ctx.log(f"{clipped} candidato(s) da live acima da nota mínima: prévia na fila.")
        if not targets:
            queue_auto_previews(ctx, run.id)
        called = stats["ok"] + stats["failed"]
        summary = {
            "model": model,
            "prompt": prompt.version,
            "candidates": len(cands),
            "ok": stats["ok"],
            "cached": stats["cached"],
            "failed": stats["failed"],
            "invalid_json": stats["invalid_json"],
            "avg_latency_ms": round(stats["latency_ms"] / called) if called else None,
        }
        save_artifact(ctx.engine, source.id, STAGE, phash, summary=summary,
                      elapsed=time.monotonic() - t_start)  # fmt: skip
        ctx.progress(
            100,
            f"IA: {stats['ok']} avaliados, {stats['cached']} do cache, {stats['failed']} falharam.",
        )
