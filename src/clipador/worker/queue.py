"""Job queue persisted in SQLite (table `job`). No Redis/Celery.

Worker-side updates are guarded by `status='running' AND locked_by=<worker>` so a job that was
canceled or recovered by someone else is never clobbered.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import Engine, update
from sqlmodel import Session, col, select

from clipador.errors import ValidationError
from clipador.models import ACTIVE_JOB_STATUSES, Job, JobStatus
from clipador.pipeline.stages import get_stage
from clipador.utils import tail_lines, utcnow

LOG_MAX_LINES = 200
BACKOFF_BASE_SECONDS = 30
BACKOFF_MAX_SECONDS = 600


def backoff_seconds(attempt: int) -> int:
    """30 s, 60 s, 120 s… capped at 10 min."""
    return min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * 2 ** max(0, attempt - 1))


def _stamp(line: str) -> str:
    return f"{utcnow().astimezone().strftime('%H:%M:%S')} {line}"


def _append(log: str, line: str) -> str:
    return tail_lines(f"{log}\n{_stamp(line)}" if log else _stamp(line), LOG_MAX_LINES)


# --- producer side (web) ------------------------------------------------------------------------


def enqueue(
    session: Session,
    job_type: str,
    source_id: int | None = None,
    payload: dict[str, Any] | None = None,
) -> Job:
    """Add a job (or return the already queued/running one for the same type+source)."""
    stage = get_stage(job_type)
    existing = session.exec(
        select(Job).where(
            Job.type == job_type,
            Job.source_id == source_id,
            col(Job.status).in_(ACTIVE_JOB_STATUSES),
        )
    ).first()
    if existing:
        return existing
    job = Job(
        type=job_type,
        source_id=source_id,
        payload=payload or {},
        needs_gpu=stage.needs_gpu,
        max_attempts=stage.max_attempts,
        log=_stamp("Job criado e na fila."),
    )
    session.add(job)
    session.flush()
    return job


def enqueue_next_in_chain(engine: Engine, job_id: int) -> Job | None:
    """Jobs may carry payload["chain"] = [next stages...]. When one finishes, queue the next
    with the rest of the chain (and the same extra payload, e.g. detection params)."""
    with Session(engine, expire_on_commit=False) as s:
        job = s.get(Job, job_id)
        chain = list((job.payload or {}).get("chain") or []) if job else []
        if not chain:
            return None
        payload = {**job.payload, "chain": chain[1:]}
        nxt = enqueue(s, chain[0], job.source_id, payload)
        s.commit()
        return nxt


def cancel(session: Session, job_id: int) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise ValidationError("Job não encontrado.")
    if job.status not in ACTIVE_JOB_STATUSES:
        raise ValidationError("Só dá para cancelar jobs na fila ou rodando.")
    was_running = job.status == JobStatus.RUNNING
    job.status = JobStatus.CANCELED
    job.finished_at = job.updated_at = utcnow()
    job.log = _append(
        job.log, "Cancelado pelo usuário." + (" Parando o processo…" if was_running else "")
    )
    session.add(job)
    return job


def retry(session: Session, job_id: int) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise ValidationError("Job não encontrado.")
    if job.status not in (JobStatus.FAILED, JobStatus.CANCELED):
        raise ValidationError("Só dá para tentar de novo jobs que falharam ou foram cancelados.")
    job.status = JobStatus.QUEUED
    job.attempts = 0
    job.error = None
    job.progress = 0
    job.locked_at = job.locked_by = job.finished_at = None
    job.run_after = job.updated_at = utcnow()
    job.log = _append(job.log, "Recolocado na fila pelo usuário.")
    session.add(job)
    return job


# --- consumer side (worker) ---------------------------------------------------------------------


def claim(engine: Engine, worker_id: str, allow_gpu: bool = True) -> Job | None:
    """Atomically take the oldest runnable queued job. A single UPDATE…WHERE id=(SELECT…) AND
    status='queued' is atomic in SQLite, so two workers can never claim the same job."""
    now = utcnow()
    candidate = (
        select(Job.id)
        .where(Job.status == JobStatus.QUEUED, Job.run_after <= now)
        .order_by(Job.created_at, Job.id)
        .limit(1)
    )
    if not allow_gpu:
        candidate = candidate.where(Job.needs_gpu == False)  # noqa: E712
    stmt = (
        update(Job)
        .where(Job.id == candidate.scalar_subquery(), Job.status == JobStatus.QUEUED)
        .values(
            status=JobStatus.RUNNING,
            locked_at=now,
            locked_by=worker_id,
            attempts=Job.attempts + 1,
            started_at=now,
            updated_at=now,
            error=None,
        )
        .returning(Job.id)
    )
    with engine.begin() as conn:
        row = conn.execute(stmt).first()
    if row is None:
        return None
    with Session(engine, expire_on_commit=False) as s:
        return s.get(Job, row[0])


def _owned(job_id: int, worker_id: str):
    return update(Job).where(
        Job.id == job_id, Job.status == JobStatus.RUNNING, Job.locked_by == worker_id
    )


def heartbeat(engine: Engine, job_ids: list[int], worker_id: str) -> None:
    if not job_ids:
        return
    now = utcnow()
    with engine.begin() as conn:
        conn.execute(
            update(Job)
            .where(
                col(Job.id).in_(job_ids),
                Job.status == JobStatus.RUNNING,
                Job.locked_by == worker_id,
            )
            .values(locked_at=now)
        )


def set_progress(engine: Engine, job_id: int, worker_id: str, percent: float) -> None:
    now = utcnow()
    with engine.begin() as conn:
        conn.execute(
            _owned(job_id, worker_id).values(
                progress=round(percent, 1), locked_at=now, updated_at=now
            )
        )


def append_log(engine: Engine, job_id: int, line: str) -> None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        if job is None:
            return
        job.log = _append(job.log, line)
        s.add(job)
        s.commit()


def get_status(engine: Engine, job_id: int) -> str | None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        return job.status if job else None


def complete(engine: Engine, job_id: int, worker_id: str) -> bool:
    now = utcnow()
    with engine.begin() as conn:
        res = conn.execute(
            _owned(job_id, worker_id).values(
                status=JobStatus.DONE,
                progress=100,
                locked_at=None,
                finished_at=now,
                updated_at=now,
                error=None,
            )
        )
    return res.rowcount == 1


def fail(
    engine: Engine,
    job_id: int,
    worker_id: str,
    message: str,
    *,
    retryable: bool,
    detail: str | None = None,
) -> str | None:
    """Record a failure. Re-queues with exponential backoff while attempts remain (and the error
    is retryable); otherwise marks the job failed. Returns the new status (None if not owned)."""
    with Session(engine) as s:
        job = s.get(Job, job_id)
        if job is None or job.status != JobStatus.RUNNING or job.locked_by != worker_id:
            return None
        now = utcnow()
        job.error = message
        lines = [f"ERRO: {message}"]
        if detail:
            lines.append("detalhe: " + tail_lines(detail, 15))
        if retryable and job.attempts < job.max_attempts:
            wait = backoff_seconds(job.attempts)
            job.status = JobStatus.QUEUED
            job.run_after = now + timedelta(seconds=wait)
            lines.append(f"Nova tentativa ({job.attempts + 1}/{job.max_attempts}) em {wait} s.")
        else:
            job.status = JobStatus.FAILED
            job.finished_at = now
            if retryable:
                lines.append(f"Desistindo após {job.attempts} tentativa(s).")
        job.locked_at = job.locked_by = None
        job.updated_at = now
        for ln in lines:
            job.log = _append(job.log, ln)
        s.add(job)
        s.commit()
        return job.status


def requeue(engine: Engine, job_id: int, worker_id: str | None, reason: str) -> bool:
    """Put a running job back in the queue without counting the interrupted attempt."""
    with Session(engine) as s:
        job = s.get(Job, job_id)
        if job is None or job.status != JobStatus.RUNNING:
            return False
        if worker_id is not None and job.locked_by != worker_id:
            return False
        job.status = JobStatus.QUEUED
        job.attempts = max(0, job.attempts - 1)
        job.locked_at = job.locked_by = None
        job.run_after = job.updated_at = utcnow()
        job.log = _append(job.log, reason)
        s.add(job)
        s.commit()
        return True


def recover_stale(engine: Engine, stale_seconds: int) -> list[int]:
    """Running jobs whose heartbeat stopped (worker died/hung) go back to the queue; if they
    already used all attempts, they are marked failed."""
    cutoff = utcnow() - timedelta(seconds=stale_seconds)
    recovered = []
    with Session(engine) as s:
        stuck = s.exec(
            select(Job).where(
                Job.status == JobStatus.RUNNING,
                (col(Job.locked_at).is_(None)) | (col(Job.locked_at) < cutoff),
            )
        ).all()
        for job in stuck:
            now = utcnow()
            if job.attempts >= job.max_attempts:
                job.status = JobStatus.FAILED
                job.finished_at = now
                job.error = "O worker parou de responder durante a última tentativa."
                job.log = _append(job.log, "Job travado e sem tentativas restantes: falhou.")
            else:
                job.status = JobStatus.QUEUED
                job.run_after = now
                job.log = _append(job.log, "Job travado (worker sem sinal); voltou para a fila.")
            job.locked_at = job.locked_by = None
            job.updated_at = now
            s.add(job)
            recovered.append(job.id)
        s.commit()
    return recovered


def recover_all_running(engine: Engine) -> list[int]:
    """At worker startup (single worker per data dir), every 'running' job is an orphan from a
    previous worker process that died: put them back without counting that attempt."""
    with Session(engine) as s:
        ids = list(s.exec(select(Job.id).where(Job.status == JobStatus.RUNNING)).all())
    return [
        i
        for i in ids
        if requeue(engine, i, None, "Worker reiniciado; job voltou para a fila e vai continuar.")
    ]
