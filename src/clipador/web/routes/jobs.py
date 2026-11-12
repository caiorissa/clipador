"""/jobs: queue view with live progress (HTMX polling), cancel, retry and log."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session, col, select

from clipador.errors import ValidationError
from clipador.models import Job
from clipador.web.deps import get_session, render
from clipador.worker import queue

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


def _jobs(session: Session) -> list[Job]:
    return list(session.exec(select(Job).order_by(col(Job.id).desc()).limit(100)).all())


def _table(request: Request, session: Session, error: str | None = None, status_code=200):
    return render(
        request, "jobs/_table.html", status_code=status_code, jobs=_jobs(session), error=error
    )


@router.get("/jobs")
def list_jobs(request: Request, session: SessionDep):
    return render(request, "jobs/list.html", jobs=_jobs(session))


@router.get("/jobs/tabela")
def jobs_table(request: Request, session: SessionDep):
    return _table(request, session)


@router.post("/jobs/{job_id}/cancelar")
def cancel_job(request: Request, job_id: int, session: SessionDep):
    try:
        queue.cancel(session, job_id)
        session.commit()
    except ValidationError as e:
        session.rollback()
        return _table(request, session, e.message)
    return _table(request, session)


@router.post("/jobs/{job_id}/tentar-de-novo")
def retry_job(request: Request, job_id: int, session: SessionDep):
    try:
        queue.retry(session, job_id)
        session.commit()
    except ValidationError as e:
        session.rollback()
        return _table(request, session, e.message)
    return _table(request, session)


def _get_job(session: Session, job_id: int) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    return job


@router.get("/jobs/{job_id}")
def job_detail(request: Request, job_id: int, session: SessionDep):
    return render(request, "jobs/detail.html", job=_get_job(session, job_id))


@router.get("/jobs/{job_id}/log")
def job_log(request: Request, job_id: int, session: SessionDep):
    return render(request, "jobs/_log.html", job=_get_job(session, job_id))
