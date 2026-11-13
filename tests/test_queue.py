import threading
import time
from datetime import timedelta

import pytest
from sqlmodel import Session

from clipador.errors import ClipadorError, ValidationError
from clipador.models import Job, JobStatus
from clipador.pipeline.base import Stage, StageContext
from clipador.utils import utcnow
from clipador.worker import queue
from clipador.worker.runner import InstanceLock, Worker

W = "test-host:1"


def _job(engine, job_type="ingest", **kw) -> Job:
    with Session(engine, expire_on_commit=False) as s:
        job = Job(type=job_type, **kw)
        s.add(job)
        s.commit()
        return job


def _get(engine, job_id) -> Job:
    with Session(engine) as s:
        return s.get(Job, job_id)


def test_enqueue_dedupes_active_jobs_and_sets_gpu_flag(session):
    a = queue.enqueue(session, "ingest", None)
    b = queue.enqueue(session, "ingest", None)
    t = queue.enqueue(session, "transcribe", None)
    session.commit()
    assert a.id == b.id
    assert a.needs_gpu is False and t.needs_gpu is True


def test_enqueue_unknown_type(session):
    with pytest.raises(ClipadorError):
        queue.enqueue(session, "nope")


def test_claim_is_atomic_under_concurrency(engine):
    jobs = [_job(engine) for _ in range(5)]
    claimed: list[int] = []
    lock = threading.Lock()

    def grab(n):
        while True:
            job = queue.claim(engine, f"w{n}")
            if job is None:
                return
            with lock:
                claimed.append(job.id)

    threads = [threading.Thread(target=grab, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(claimed) == sorted(j.id for j in jobs)  # each job exactly once
    for j in jobs:
        got = _get(engine, j.id)
        assert got.status == JobStatus.RUNNING and got.attempts == 1 and got.locked_at


def test_claim_respects_gpu_and_run_after(engine):
    gpu = _job(engine, "transcribe", needs_gpu=True)
    _job(engine, run_after=utcnow() + timedelta(hours=1))  # backoff in the future
    assert queue.claim(engine, W, allow_gpu=False) is None
    assert queue.claim(engine, W, allow_gpu=True).id == gpu.id


def test_fail_retries_with_backoff_then_fails(engine):
    job = _job(engine, max_attempts=2)
    queue.claim(engine, W)
    assert queue.fail(engine, job.id, W, "rede caiu", retryable=True) == JobStatus.QUEUED
    j = _get(engine, job.id)
    assert j.error == "rede caiu" and j.run_after > utcnow() + timedelta(seconds=20)
    assert queue.claim(engine, W) is None  # still in backoff

    with Session(engine) as s:  # skip the wait
        j = s.get(Job, job.id)
        j.run_after = utcnow()
        s.add(j)
        s.commit()
    assert queue.claim(engine, W).attempts == 2
    assert queue.fail(engine, job.id, W, "rede caiu", retryable=True) == JobStatus.FAILED
    assert "Desistindo" in _get(engine, job.id).log


def test_non_retryable_fails_immediately(engine):
    job = _job(engine)
    queue.claim(engine, W)
    assert queue.fail(engine, job.id, W, "privado", retryable=False) == JobStatus.FAILED


def test_backoff_growth():
    assert [queue.backoff_seconds(n) for n in (1, 2, 3, 10)] == [30, 60, 120, 600]


def test_recover_stale_job(engine):
    job = _job(engine)
    queue.claim(engine, W)
    with Session(engine) as s:
        j = s.get(Job, job.id)
        j.locked_at = utcnow() - timedelta(minutes=10)
        s.add(j)
        s.commit()
    assert queue.recover_stale(engine, stale_seconds=90) == [job.id]
    j = _get(engine, job.id)
    assert j.status == JobStatus.QUEUED and j.locked_by is None


def test_fresh_running_job_is_not_recovered(engine):
    _job(engine)
    queue.claim(engine, W)
    assert queue.recover_stale(engine, stale_seconds=90) == []


def test_recover_all_running_at_startup_does_not_count_attempt(engine):
    job = _job(engine)
    queue.claim(engine, W)
    assert queue.recover_all_running(engine) == [job.id]
    j = _get(engine, job.id)
    assert j.status == JobStatus.QUEUED and j.attempts == 0
    assert "Worker reiniciado" in j.log


def test_cancel_and_retry(session, engine):
    job = queue.enqueue(session, "ingest")
    session.commit()
    queue.cancel(session, job.id)
    session.commit()
    assert _get(engine, job.id).status == JobStatus.CANCELED
    with pytest.raises(ValidationError):
        queue.cancel(session, job.id)
    queue.retry(session, job.id)
    session.commit()
    j = _get(engine, job.id)
    assert j.status == JobStatus.QUEUED and j.attempts == 0


def test_worker_updates_are_ignored_after_cancel(session, engine):
    job = queue.enqueue(session, "ingest")
    session.commit()
    queue.claim(engine, W)
    queue.cancel(session, job.id)
    session.commit()
    assert queue.complete(engine, job.id, W) is False
    assert queue.fail(engine, job.id, W, "x", retryable=True) is None
    assert _get(engine, job.id).status == JobStatus.CANCELED


# --- runner ------------------------------------------------------------------------------------


class _OkStage(Stage):
    name = "ok"

    def run(self, ctx: StageContext) -> None:
        for p in (25, 50, 100):
            ctx.progress(p)


class _BoomStage(Stage):
    name = "boom"

    def run(self, ctx: StageContext) -> None:
        raise ClipadorError("falhou feio", retryable=False, detail="trace")


class _SlowStage(Stage):
    name = "slow"

    def run(self, ctx: StageContext) -> None:
        for _ in range(200):
            time.sleep(0.02)
            ctx.progress(1, None)
            ctx.check_canceled()


class _GpuStage(Stage):
    name = "gpu"
    needs_gpu = True
    running = 0
    peak = 0
    lock = threading.Lock()

    def run(self, ctx: StageContext) -> None:
        with _GpuStage.lock:
            _GpuStage.running += 1
            _GpuStage.peak = max(_GpuStage.peak, _GpuStage.running)
        time.sleep(0.2)
        with _GpuStage.lock:
            _GpuStage.running -= 1


STAGES = {s.name: s for s in (_OkStage, _BoomStage, _SlowStage, _GpuStage)}


def _worker(settings, engine, concurrency=1) -> Worker:
    settings.worker_concurrency = concurrency
    return Worker(settings, engine, stages=STAGES)


def _drain(worker: Worker, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        worker.tick()
        if not worker.active_job_ids():
            with Session(worker.engine) as s:
                from sqlmodel import select

                pending = s.exec(
                    select(Job).where(Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]))
                ).all()
            if not pending:
                return
        time.sleep(0.02)
    raise AssertionError("timeout")


def test_runner_completes_job_with_progress_and_log(settings, engine):
    job = _job(engine, "ok")
    w = _worker(settings, engine)
    _drain(w)
    j = _get(engine, job.id)
    assert j.status == JobStatus.DONE and j.progress == 100
    assert "Iniciando 'ok'" in j.log


def test_runner_records_failure(settings, engine):
    job = _job(engine, "boom")
    _drain(_worker(settings, engine))
    j = _get(engine, job.id)
    assert j.status == JobStatus.FAILED and j.error == "falhou feio"
    assert "ERRO: falhou feio" in j.log


class _StubStage(Stage):
    name = "stub"

    def run(self, ctx: StageContext) -> None:
        raise NotImplementedError("em breve")


def test_runner_stub_stage_not_implemented(settings, engine):
    job = _job(engine, "stub")
    w = Worker(settings, engine, stages={**STAGES, "stub": _StubStage})
    _drain(w)
    j = _get(engine, job.id)
    assert j.status == JobStatus.FAILED and "não implementado" in j.error


def test_runner_cancel_running_job(settings, engine, session):
    job = _job(engine, "slow")
    w = _worker(settings, engine)
    w.tick()
    time.sleep(0.2)
    queue.cancel(session, job.id)
    session.commit()
    deadline = time.monotonic() + 5
    while w.active_job_ids() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not w.active_job_ids()
    j = _get(engine, job.id)
    assert j.status == JobStatus.CANCELED and "cancelado" in j.log.lower()


def test_runner_stop_requeues_running_job(settings, engine):
    job = _job(engine, "slow")
    w = _worker(settings, engine)
    w.tick()
    time.sleep(0.2)
    w.shutdown(timeout=5)
    j = _get(engine, job.id)
    assert j.status == JobStatus.QUEUED and j.attempts == 0


def test_gpu_jobs_never_run_concurrently(settings, engine):
    _GpuStage.peak = 0
    for _ in range(3):
        _job(engine, "gpu", needs_gpu=True)
    ok = _job(engine, "ok")
    w = _worker(settings, engine, concurrency=3)
    _drain(w)
    assert _GpuStage.peak == 1
    assert _get(engine, ok.id).status == JobStatus.DONE


def test_instance_lock_is_exclusive(tmp_path):
    a, b = InstanceLock(tmp_path / "w.lock"), InstanceLock(tmp_path / "w.lock")
    assert a.acquire()
    assert not b.acquire()
    a.release()
    assert b.acquire()
    b.release()
