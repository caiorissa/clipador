"""Worker loop: claims jobs, runs stages in threads, enforces the single GPU lock, retries."""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import IO

from sqlalchemy import Engine

from clipador.config import Settings
from clipador.db import make_engine, run_migrations, session_scope
from clipador.errors import ClipadorError, JobCanceled
from clipador.models import Job, JobStatus
from clipador.pipeline.base import Stage, StageContext, WorkerStopping
from clipador.pipeline.stages import STAGES
from clipador.services.proc import kill_children_on_exit
from clipador.worker import queue

log = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 15
CANCEL_CHECK_SECONDS = 1.0


class WorkerAlreadyRunning(ClipadorError):
    pass


class InstanceLock:
    """Exclusive lock file so only one worker runs per data dir (makes startup recovery safe)."""

    def __init__(self, path: Path):
        self.path = path
        self._fh: IO[str] | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")  # noqa: SIM115 - kept open for the process lifetime
        try:
            if sys.platform == "win32":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


class _Reporter:
    """Persists progress/log for StageContext; caches cancel checks to avoid DB spam."""

    def __init__(self, worker: Worker):
        self.worker = worker
        self._cancel_cache: dict[int, tuple[float, bool]] = {}

    def report_progress(self, job_id: int, percent: float) -> None:
        queue.set_progress(self.worker.engine, job_id, self.worker.worker_id, percent)

    def append_log(self, job_id: int, line: str) -> None:
        queue.append_log(self.worker.engine, job_id, line)

    def is_canceled(self, job_id: int) -> bool:
        now = time.monotonic()
        cached = self._cancel_cache.get(job_id)
        if cached and now - cached[0] < CANCEL_CHECK_SECONDS:
            return cached[1]
        status = queue.get_status(self.worker.engine, job_id)
        canceled = status == JobStatus.CANCELED
        self._cancel_cache[job_id] = (now, canceled)
        return canceled

    def is_stopping(self) -> bool:
        return self.worker.stopping.is_set()


class Worker:
    def __init__(
        self,
        settings: Settings,
        engine: Engine | None = None,
        stages: dict[str, type[Stage]] | None = None,
    ):
        self.settings = settings
        self.engine = engine or make_engine(settings)
        self.stages = stages or STAGES
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}"
        self.gpu_lock = threading.BoundedSemaphore(1)  # one GPU job at a time, ever
        self.stopping = threading.Event()
        self.reporter = _Reporter(self)
        self._active: dict[int, threading.Thread] = {}
        self._active_types: dict[int, str] = {}
        self._active_lock = threading.Lock()

    # --- main loop -----------------------------------------------------------------------------
    def run_forever(self) -> None:
        lock = InstanceLock(self.settings.data_dir / "worker.lock")
        if not lock.acquire():
            raise WorkerAlreadyRunning(
                "Já existe um worker rodando para esta pasta de dados. Feche o outro antes."
            )
        try:
            run_migrations(self.settings)
            if kill_children_on_exit():
                log.info("Processos filhos (yt-dlp/ffmpeg) serão encerrados junto com o worker.")
            recovered = queue.recover_all_running(self.engine)
            if recovered:
                log.info("Jobs interrompidos recolocados na fila: %s", recovered)
            log.info(
                "Worker %s pronto (até %d jobs simultâneos, 1 de GPU). Ctrl+C para parar.",
                self.worker_id,
                self.settings.worker_concurrency,
            )
            last_beat = last_recover = 0.0
            while not self.stopping.is_set():
                started = self.tick()
                now = time.monotonic()
                if now - last_beat >= HEARTBEAT_SECONDS:
                    queue.heartbeat(self.engine, self.active_job_ids(), self.worker_id)
                    last_beat = now
                if now - last_recover >= self.settings.job_stale_seconds / 3:
                    for jid in queue.recover_stale(self.engine, self.settings.job_stale_seconds):
                        log.warning("Job %s estava travado e voltou para a fila", jid)
                    last_recover = now
                if not started:
                    time.sleep(self.settings.worker_poll_seconds)
        except KeyboardInterrupt:
            log.info("Parando o worker…")
        finally:
            self.shutdown()
            lock.release()

    def shutdown(self, timeout: float = 20.0) -> None:
        self.stopping.set()
        deadline = time.monotonic() + timeout
        for t in list(self._active.values()):
            t.join(max(0.0, deadline - time.monotonic()))
        # Anything still running is requeued now (or at next startup).
        for jid in self.active_job_ids():
            queue.requeue(
                self.engine, jid, self.worker_id, "Worker encerrado; job voltou para a fila."
            )

    def active_job_ids(self) -> list[int]:
        with self._active_lock:
            return list(self._active)

    def _busy_types(self) -> tuple[str, ...]:
        """One render at a time, even when renders don't take the GPU lock."""
        from clipador.pipeline.stages.render import RENDER_JOB_TYPES

        with self._active_lock:
            types = set(self._active_types.values())
        return RENDER_JOB_TYPES if types & set(RENDER_JOB_TYPES) else ()

    def tick(self) -> bool:
        """Claim and start at most one job. Returns True if a job was started."""
        if len(self.active_job_ids()) >= self.settings.worker_concurrency:
            return False
        gpu_free = self.gpu_lock.acquire(blocking=False)
        try:
            job = queue.claim(
                self.engine, self.worker_id, allow_gpu=gpu_free, exclude_types=self._busy_types()
            )
        except Exception:
            if gpu_free:
                self.gpu_lock.release()
            raise
        holds_gpu = bool(job and job.needs_gpu)
        if gpu_free and not holds_gpu:
            self.gpu_lock.release()
        if job is None:
            return False
        t = threading.Thread(
            target=self._run_thread, args=(job.id, holds_gpu), name=f"job-{job.id}", daemon=True
        )
        with self._active_lock:
            self._active[job.id] = t
            self._active_types[job.id] = job.type
        t.start()
        return True

    def _run_thread(self, job_id: int, holds_gpu: bool) -> None:
        try:
            self.execute(job_id)
        finally:
            if holds_gpu:
                self.gpu_lock.release()
            with self._active_lock:
                self._active.pop(job_id, None)
                self._active_types.pop(job_id, None)

    # --- running one job -----------------------------------------------------------------------
    def execute(self, job_id: int) -> None:
        with session_scope(self.engine) as s:
            job = s.get(Job, job_id)
            assert job is not None
            ctx = StageContext(
                job_id=job.id,
                job_type=job.type,
                source_id=job.source_id,
                payload=dict(job.payload or {}),
                attempt=job.attempts,
                settings=self.settings,
                engine=self.engine,
                reporter=self.reporter,
            )
            max_attempts = job.max_attempts
        stage_cls = self.stages.get(ctx.job_type)
        if stage_cls is None:
            self._fail(ctx, None, f"Tipo de job desconhecido: '{ctx.job_type}'.", False, None)
            return
        stage = stage_cls()
        ctx.log(f"Iniciando '{ctx.job_type}' (tentativa {ctx.attempt}/{max_attempts}).")
        log.info("job %s (%s) iniciado", job_id, ctx.job_type)
        try:
            if stage.is_done(ctx):
                ctx.log("Resultado já existe e está íntegro; nada a fazer.")
            else:
                stage.run(ctx)
        except JobCanceled:
            ctx.log("Job cancelado; processo interrompido.")
            self._safe_hook(stage, ctx, "Cancelado pelo usuário.", True)
            log.info("job %s cancelado", job_id)
            return
        except WorkerStopping:
            queue.requeue(
                self.engine, job_id, self.worker_id, "Worker encerrado; job voltou para a fila."
            )
            return
        except NotImplementedError as e:
            self._fail(
                ctx, stage, f"Estágio '{ctx.job_type}' ainda não implementado ({e}).", False, None
            )
            return
        except ClipadorError as e:
            if self.stopping.is_set():  # e.g. Ctrl+C also killed yt-dlp
                queue.requeue(
                    self.engine, job_id, self.worker_id, "Worker encerrado; job voltou para a fila."
                )
                return
            self._fail(ctx, stage, e.message, e.retryable, e.detail)
            return
        except Exception as e:
            log.exception("job %s: erro inesperado", job_id)
            if self.stopping.is_set():
                queue.requeue(
                    self.engine, job_id, self.worker_id, "Worker encerrado; job voltou para a fila."
                )
                return
            self._fail(ctx, stage, f"Erro inesperado: {e}", True, traceback.format_exc())
            return
        if queue.complete(self.engine, job_id, self.worker_id):
            log.info("job %s concluído", job_id)
            nxt = queue.enqueue_next_in_chain(self.engine, job_id)
            if nxt is not None:
                ctx.log(f"Próximo estágio na fila: '{nxt.type}' (job #{nxt.id}).")

    def _fail(
        self,
        ctx: StageContext,
        stage: Stage | None,
        message: str,
        retryable: bool,
        detail: str | None,
    ) -> None:
        log.warning("job %s falhou: %s", ctx.job_id, message)
        if detail:
            log.debug("detalhe: %s", detail)
        status = queue.fail(
            self.engine, ctx.job_id, self.worker_id, message, retryable=retryable, detail=detail
        )
        if stage is not None and status is not None:
            self._safe_hook(stage, ctx, message, status == JobStatus.FAILED)

    @staticmethod
    def _safe_hook(stage: Stage, ctx: StageContext, message: str, final: bool) -> None:
        try:
            stage.on_failure(ctx, message, final)
        except Exception:
            log.exception("on_failure do estágio %s falhou", ctx.job_type)
