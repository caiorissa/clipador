"""render: preview / final 9:16 clips (Render rows); compile_long: 16:9 compilation.

Renders hold the GPU lock by default (NVENC) unless render_parallel_with_gpu_jobs=true; either
way the worker runs one render at a time. Each Render is cached by its params hash, written to a
temp file and validated with ffprobe before getting its final name.
"""

from __future__ import annotations

import time

from sqlmodel import col, select

from clipador.errors import ClipadorError
from clipador.models import RENDERABLE_STATUSES, Candidate, Render, RenderStatus
from clipador.pipeline.base import Stage, StageContext
from clipador.services import rendering
from clipador.services.artifacts import save_artifact

RENDER_JOB_TYPES = ("render", "compile_long")


class RenderStage(Stage):
    name = "render"
    needs_gpu = True
    max_attempts = 2

    def run(self, ctx: StageContext) -> None:
        ids = list(ctx.payload.get("render_ids") or [])
        if not ids:
            raise ClipadorError("Nenhum render na fila deste job.")
        t0 = time.monotonic()
        done = []
        with ctx.session() as s:
            rows = s.exec(select(Render).where(col(Render.id).in_(ids))).all()
            rows = sorted(rows, key=lambda r: ids.index(r.id))
            for i, row in enumerate(rows):
                ready = row.path and ctx.settings.abs_path(row.path).is_file()
                if row.status == RenderStatus.DONE and ready:
                    ctx.log(f"Render #{row.id} já pronto; reaproveitado.")
                    done.append(row.id)
                    continue
                c = s.get(Candidate, row.candidate_id)
                plan = rendering.plan_render(s, ctx.engine, ctx.settings, c, row.kind,
                                             row.variant)  # fmt: skip

                def progress(f: float, i=i) -> None:
                    ctx.progress(100 * (i + f) / len(rows))

                try:
                    rendering.execute_render(s, ctx.settings, row, plan, progress, ctx.log,
                                             job_id=ctx.job_id)  # fmt: skip
                except Exception:
                    s.rollback()
                    raise
                done.append(row.id)
        save_artifact(
            ctx.engine, ctx.source_id, "render", f"job{ctx.job_id}",
            summary={"clips": len(done)}, elapsed=time.monotonic() - t0,
        )  # fmt: skip
        ctx.progress(100, f"{len(done)} render(s) pronto(s) em data/clips/{ctx.source_id}/.")

    def on_failure(self, ctx: StageContext, message: str, final: bool) -> None:
        with ctx.session() as s:
            canceled = message.startswith("Cancelado")
            status = RenderStatus.CANCELED if canceled else RenderStatus.FAILED
            if final or canceled:
                rendering.mark_unfinished(s, ctx.job_id, status, message)
            else:  # will be retried: back to queued
                rendering.mark_unfinished(
                    s, ctx.job_id, RenderStatus.QUEUED, f"{message} (nova tentativa)"
                )


class CompileLongStage(Stage):
    name = "compile_long"
    needs_gpu = True
    max_attempts = 2

    def run(self, ctx: StageContext) -> None:
        t0 = time.monotonic()
        with ctx.session() as s:
            q = select(Candidate).where(
                Candidate.source_id == ctx.source_id, col(Candidate.status).in_(RENDERABLE_STATUSES)
            )
            if ctx.payload.get("run_id"):
                q = q.where(Candidate.analysis_run_id == ctx.payload["run_id"])
            cands = s.exec(q).all()
            plans = [rendering.plan_render(s, ctx.engine, ctx.settings, c, "final") for c in cands]
        ctx.log(f"Vídeo longo com {len(plans)} momento(s) aprovado(s), em ordem cronológica.")
        video, chapters, phash = rendering.compile_long(
            ctx.settings, plans, lambda f: ctx.progress(100 * f), ctx.log
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
