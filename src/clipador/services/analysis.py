"""Analysis orchestration used by the web UI and CLI (enqueue chains, read runs/timeline)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sqlalchemy import Engine
from sqlmodel import Session, col, select

from clipador.config import Settings
from clipador.detection.candidates import interest_curve
from clipador.detection.features import Features
from clipador.detection.params import DetectParams
from clipador.errors import ValidationError
from clipador.models import (
    ACTIVE_JOB_STATUSES,
    AnalysisRun,
    Candidate,
    CandidateFeedback,
    Job,
    JobStatus,
    Source,
    SourceStatus,
    StageArtifact,
)
from clipador.worker import queue

ANALYSIS_CHAIN = ["extract_audio", "transcribe", "fetch_chat", "detect", "enrich"]
STAGE_LABELS = {
    "extract_audio": "Extrair áudio",
    "transcribe": "Transcrever",
    "fetch_chat": "Chat",
    "detect": "Detectar",
    "enrich": "IA (Ollama)",
}
EXTRA_STAGES = {"render": "Cortes verticais", "compile_long": "Vídeo longo 16:9"}


def start_analysis(
    session: Session, source_id: int, detect_params: dict[str, Any] | None = None
) -> Job:
    source = session.get(Source, source_id)
    if source is None:
        raise ValidationError("Fonte não encontrada.")
    if source.status != SourceStatus.READY:
        raise ValidationError("Espere o download da fonte terminar antes de analisar.")
    payload: dict[str, Any] = {"chain": ANALYSIS_CHAIN[1:]}
    if detect_params:
        payload["detect_params"] = detect_params
    return queue.enqueue(session, ANALYSIS_CHAIN[0], source_id, payload)


def reanalyze(session: Session, source_id: int, params: DetectParams) -> Job:
    """Only re-run `detect` (features are cached, so this takes seconds)."""
    has_transcript = session.exec(
        select(StageArtifact).where(
            StageArtifact.source_id == source_id, StageArtifact.stage == "transcribe"
        )
    ).first()
    if has_transcript is None:
        raise ValidationError("Analise a fonte primeiro (botão Analisar).")
    payload = {"detect_params": params.model_dump()}
    active = session.exec(
        select(Job).where(
            Job.type == "detect",
            Job.source_id == source_id,
            col(Job.status).in_(ACTIVE_JOB_STATUSES),
        )
    ).first()
    if active is not None and active.status == JobStatus.QUEUED:
        # not started yet: the latest params win (e.g. apply then revert a calibrated preset)
        active.payload = {**(active.payload or {}), **payload}
        session.add(active)
        session.flush()
        return active
    # a detect already running keeps its params; this one runs after it
    return queue.enqueue(session, "detect", source_id, payload, dedupe=active is None)


def artifact(session: Session, source_id: int, stage: str) -> StageArtifact | None:
    return session.exec(
        select(StageArtifact).where(
            StageArtifact.source_id == source_id, StageArtifact.stage == stage
        )
    ).first()


def enqueue_enrich(
    session: Session,
    source_id: int,
    *,
    model: str | None = None,
    k: int | None = None,
    candidate_ids: list[int] | None = None,
    instruction: str = "",
    overwrite_edited: bool = False,
) -> Job:
    if not runs_for(session, source_id):
        raise ValidationError("Analise a fonte antes de enriquecer com IA.")
    payload: dict[str, Any] = {}
    if model:
        payload["model"] = model
    if k:
        payload["k"] = int(k)
    if candidate_ids:
        payload["candidate_ids"] = candidate_ids
        payload["instruction"] = " ".join(instruction.split())[:200]
        payload["overwrite_edited"] = overwrite_edited
    return queue.enqueue(session, "enrich", source_id, payload)


def enqueue_render(
    session: Session,
    settings: Settings,
    source_id: int,
    candidate_ids: list[int],
    kind: str = "final",
    priority: int = 0,
) -> tuple[Job | None, list, list]:
    """Create Render rows (reusing cached ones) and queue one job for the missing ones.
    Returns (job or None if everything was cached, to_render, reused).
    A render already waiting in a lower-priority job (e.g. review prefetch) is bumped up."""
    from clipador.services import rendering

    if not candidate_ids:
        raise ValidationError("Escolha (ou aprove) pelo menos um candidato para renderizar.")
    todo, reused = rendering.request_renders(
        session, session.get_bind(), settings, candidate_ids, kind
    )
    for r in todo:
        waiting = session.get(Job, r.job_id) if r.job_id is not None else None
        if waiting and waiting.status == JobStatus.QUEUED and waiting.priority < priority:
            waiting.priority = priority
            session.add(waiting)
    fresh = [r for r in todo if r.job_id is None]
    job = None
    if fresh:
        job = queue.enqueue(
            session, "render", source_id, {"render_ids": [r.id for r in fresh]}, dedupe=False,
            needs_gpu=not settings.render_parallel_with_gpu_jobs, priority=priority,
        )  # fmt: skip
        for r in fresh:
            r.job_id = job.id
            session.add(r)
    return job, todo, reused


def enqueue_compile_long(session: Session, source_id: int, run_id: int | None) -> Job:
    return queue.enqueue(session, "compile_long", source_id, {"run_id": run_id})


def stage_overview(session: Session, source_id: int) -> list[dict[str, Any]]:
    """Per-stage status for the UI: latest job + artifact."""
    rows = []
    for stage in [*ANALYSIS_CHAIN, *EXTRA_STAGES]:
        job = session.exec(
            select(Job)
            .where(Job.source_id == source_id, Job.type == stage)
            .order_by(col(Job.id).desc())
        ).first()
        art = session.exec(
            select(StageArtifact).where(
                StageArtifact.source_id == source_id, StageArtifact.stage == stage
            )
        ).first()
        elapsed = None
        if job and job.started_at and job.finished_at:
            elapsed = (job.finished_at - job.started_at).total_seconds()
        rows.append(
            {
                "stage": stage,
                "label": STAGE_LABELS.get(stage) or EXTRA_STAGES[stage],
                "job": job,
                "artifact": art,
                "elapsed": elapsed,
                "active": bool(job and job.status in ACTIVE_JOB_STATUSES),
            }
        )
    return rows


def runs_for(session: Session, source_id: int) -> list[AnalysisRun]:
    return list(
        session.exec(
            select(AnalysisRun)
            .where(AnalysisRun.source_id == source_id)
            .order_by(col(AnalysisRun.id).desc())
        ).all()
    )


def candidates_for(
    session: Session, run_id: int
) -> list[tuple[Candidate, CandidateFeedback | None]]:
    cands = session.exec(
        select(Candidate).where(Candidate.analysis_run_id == run_id).order_by(Candidate.rank)
    ).all()
    fb = {
        f.candidate_id: f
        for f in session.exec(
            select(CandidateFeedback).where(
                col(CandidateFeedback.candidate_id).in_([c.id for c in cands])
            )
        ).all()
    }
    return [(c, fb.get(c.id)) for c in cands]


def load_features(engine: Engine, settings: Settings, source_id: int) -> Features | None:
    with Session(engine) as s:
        art = s.exec(
            select(StageArtifact).where(
                StageArtifact.source_id == source_id, StageArtifact.stage == "features"
            )
        ).first()
    if art is None:
        return None
    path = settings.abs_path(art.files.get("features", ""))
    return Features.load(path) if path.is_file() else None


def timeline(feats: Features, run: AnalysisRun, max_points: int = 2000) -> dict[str, Any]:
    params = DetectParams.model_validate(
        {k: v for k, v in run.params.items() if k != "effective_weights"}
    )
    curve, _eff = interest_curve(feats, params.candidates.weights)
    step = max(1, int(np.ceil(curve.size / max_points)))

    def down(x: np.ndarray) -> list[float]:
        n = (x.size // step) * step
        head = x[:n].reshape(-1, step).max(axis=1) if n else np.zeros(0)
        tail = [float(x[n:].max())] if x.size > n else []
        return [round(float(v), 3) for v in head] + tail

    t = [round(feats.t0 + i * step, 1) for i in range(int(np.ceil(curve.size / step)))]
    data = {"t": t, "interest": down(curve), "audio": down(feats.signals["audio"])}
    if feats.has_chat:
        data["chat"] = down(feats.signals["chat"])
    return data


def curve_value_at(feats: Features, weights: dict[str, float], t: float) -> float | None:
    curve, _ = interest_curve(feats, weights)
    i = int(np.floor(t - feats.t0))
    if 0 <= i < curve.size:
        return round(100 * float(curve[i]), 1)
    return None
