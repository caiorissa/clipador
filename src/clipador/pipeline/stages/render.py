"""render: 9:16 clips for the chosen candidates; compile_long: 16:9 compilation of approved ones.

Both use the GPU lock (NVENC), are idempotent per clip (params hash) and resume after a crash:
clips already rendered with the same settings are skipped.
"""

from __future__ import annotations

import time

from sqlmodel import col, select

from clipador.errors import ClipadorError
from clipador.models import Candidate, CandidateStatus
from clipador.pipeline.base import Stage, StageContext
from clipador.services import rendering
from clipador.services.artifacts import save_artifact


class RenderStage(Stage):
    name = "render"
    needs_gpu = True  # NVENC; never alongside Whisper/Ollama
    max_attempts = 2

    def run(self, ctx: StageContext) -> None:
        ids = list(ctx.payload.get("candidate_ids") or [])
        if not ids:
            raise ClipadorError("Nenhum candidato escolhido para renderizar.")
        t0 = time.monotonic()
        rendered = []
        with ctx.session() as s:
            cands = s.exec(select(Candidate).where(col(Candidate.id).in_(ids))).all()
            cands = sorted(cands, key=lambda c: ids.index(c.id))
            for i, c in enumerate(cands):
                plan = rendering.plan_render(s, ctx.engine, ctx.settings, c)

                def progress(f: float, i=i) -> None:
                    ctx.progress(100 * (i + f) / len(cands))

                row = rendering.render_clip(s, ctx.settings, c, plan, progress, ctx.log)
                rendered.append(row.file_path)
        save_artifact(
            ctx.engine, ctx.source_id, "render", f"job{ctx.job_id}",
            files={}, summary={"clips": len(rendered), "last": rendered[-1:]},
            elapsed=time.monotonic() - t0,
        )  # fmt: skip
        ctx.progress(100, f"{len(rendered)} corte(s) pronto(s) em data/clips/{ctx.source_id}/.")


class CompileLongStage(Stage):
    name = "compile_long"
    needs_gpu = True
    max_attempts = 2

    def run(self, ctx: StageContext) -> None:
        t0 = time.monotonic()
        with ctx.session() as s:
            q = select(Candidate).where(
                Candidate.source_id == ctx.source_id, Candidate.status == CandidateStatus.APPROVED
            )
            if ctx.payload.get("run_id"):
                q = q.where(Candidate.analysis_run_id == ctx.payload["run_id"])
            cands = s.exec(q).all()
            plans = [rendering.plan_render(s, ctx.engine, ctx.settings, c) for c in cands]
        ctx.log(f"Vídeo longo com {len(plans)} momento(s) aprovado(s), em ordem cronológica.")
        video, chapters, phash = rendering.compile_long(
            None, ctx.settings, plans, lambda f: ctx.progress(100 * f), ctx.log
        )
        save_artifact(
            ctx.engine, ctx.source_id, "compile_long", phash,
            files={
                "video": ctx.settings.rel_path(video),
                "chapters": ctx.settings.rel_path(chapters),
            },
            summary={"moments": len(plans), "duration": round(sum(p.duration for p in plans), 1)},
            elapsed=time.monotonic() - t0,
        )  # fmt: skip
        ctx.progress(100, f"Vídeo longo pronto: {video.name}")
