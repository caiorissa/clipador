"""Pluggable processing stages.

Every stage has the same shape: it reads its input from the DB/disk (via StageContext), writes its
output to the DB/disk, reports progress and log lines, and raises on error. `is_done` makes each
stage idempotent: a retried job skips work that already succeeded, and any single stage can be
re-run by enqueueing a job of its type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

from sqlalchemy import Engine
from sqlmodel import Session

from clipador.config import Settings


class WorkerStopping(Exception):
    """Raised inside a stage when the worker is shutting down; the job goes back to the queue."""


class JobReporter(Protocol):
    """Implemented by the worker; persists progress/log and answers cancel/stop questions."""

    def report_progress(self, job_id: int, percent: float) -> None: ...
    def append_log(self, job_id: int, line: str) -> None: ...
    def is_canceled(self, job_id: int) -> bool: ...
    def is_stopping(self) -> bool: ...


@dataclass
class StageContext:
    job_id: int
    job_type: str
    source_id: int | None
    payload: dict[str, Any]
    attempt: int
    settings: Settings
    engine: Engine
    reporter: JobReporter
    _last_pct: float = field(default=-1.0, repr=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with Session(self.engine, expire_on_commit=False) as s:
            yield s

    def log(self, line: str) -> None:
        self.reporter.append_log(self.job_id, line)

    def check_canceled(self) -> None:
        from clipador.errors import JobCanceled

        if self.reporter.is_stopping():
            raise WorkerStopping()
        if self.reporter.is_canceled(self.job_id):
            raise JobCanceled()

    def progress(self, percent: float, message: str | None = None) -> None:
        """Report 0-100 progress. Raises JobCanceled / WorkerStopping when appropriate."""
        percent = max(0.0, min(100.0, percent))
        if message:
            self.log(message)
        if percent - self._last_pct >= 0.5 or percent >= 100 or message:
            self._last_pct = percent
            self.reporter.report_progress(self.job_id, percent)
            self.check_canceled()


class Stage(ABC):
    name: ClassVar[str]
    needs_gpu: ClassVar[bool] = False  # GPU stages run one at a time (worker GPU lock)
    max_attempts: ClassVar[int] = 3

    def is_done(self, ctx: StageContext) -> bool:
        """True if the output already exists and is valid (job is skipped)."""
        return False

    @abstractmethod
    def run(self, ctx: StageContext) -> None: ...

    def on_failure(self, ctx: StageContext, message: str, final: bool) -> None:  # noqa: B027
        """Hook to reflect a failure on domain rows (e.g. Source.status). Default: nothing."""
