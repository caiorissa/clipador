"""Automatic faithful previews (540x960) for the best candidates of an analysis (step 5E).

After an analysis (VOD or live) the top N candidates by final score get their faithful preview
queued in the background, so the review screen already has real renders to watch. These jobs
run BELOW renders asked for by hand (priority 0) and above the review prefetch (-10); a live
candidate that crossed the auto-clip threshold is queued ABOVE everything (it's urgent).
"""

from __future__ import annotations

from sqlmodel import Session, col, select

from clipador.config import Settings
from clipador.errors import ClipadorError
from clipador.models import (
    Candidate,
    CandidateStatus,
    Job,
    LiveSegment,
    LiveSession,
    Render,
    RenderKind,
    RenderStatus,
    Source,
)

AUTO_PREVIEW_PRIORITY = -5
AUTO_CLIP_PRIORITY = 20


def _score(c: Candidate) -> float:
    return c.final_score if c.final_score is not None else c.score


def top_candidates(session: Session, run_id: int, n: int) -> list[Candidate]:
    """The best `n` candidates of a run by final score (rejected ones never)."""
    if n <= 0:
        return []
    cands = session.exec(
        select(Candidate).where(
            Candidate.analysis_run_id == run_id, Candidate.status != CandidateStatus.REJECTED
        )
    ).all()
    return sorted(cands, key=lambda c: (-_score(c), c.rank, c.id))[:n]


def has_video(session: Session, settings: Settings, source_id: int) -> bool:
    """True when renders of this source can be planned: the downloaded file, or (live) at least
    one captured segment still on disk."""
    src = session.get(Source, source_id)
    if src is None:
        return False
    if src.file_path and settings.abs_path(src.file_path).is_file():
        return True
    seg = session.exec(
        select(LiveSegment)
        .join(LiveSession, LiveSession.id == LiveSegment.session_id)
        .where(LiveSession.source_id == source_id, LiveSegment.path != None)  # noqa: E711
    ).first()
    return seg is not None and settings.abs_path(seg.path).is_file()


def without_preview(session: Session, candidates: list[Candidate]) -> list[Candidate]:
    """Candidates with no preview done or on its way (any params: a preview made with older
    options still counts, the user can ask for a new one)."""
    if not candidates:
        return []
    rows = session.exec(
        select(Render).where(
            col(Render.candidate_id).in_([c.id for c in candidates]),
            Render.kind == RenderKind.PREVIEW,
            col(Render.status).in_([RenderStatus.DONE, RenderStatus.QUEUED, RenderStatus.RUNNING]),
        )
    ).all()
    covered = {r.candidate_id for r in rows}
    return [c for c in candidates if c.id not in covered]


def enqueue_previews(
    session: Session, settings: Settings, candidates: list[Candidate], priority: int
) -> tuple[Job | None, int]:
    """Queue faithful previews for `candidates` (all of the same source) in ONE job.
    Cache-aware (an identical preview is reused, a queued one is bumped to `priority`).
    Best effort: returns (None, 0) when the source video is gone. Does not commit."""
    from clipador.services import analysis

    if not candidates or not has_video(session, settings, candidates[0].source_id):
        return None, 0
    try:
        job, todo, _reused = analysis.enqueue_render(
            session, settings, candidates[0].source_id, [c.id for c in candidates], "preview",
            priority=priority,
        )  # fmt: skip
    except ClipadorError:
        return None, 0
    return job, len(todo)


def auto_previews(
    session: Session, settings: Settings, run_id: int, n: int | None = None
) -> tuple[Job | None, int]:
    """Previews of the top N (`auto_preview_top_n`, 0 = off) of a run, low priority."""
    n = settings.auto_preview_top_n if n is None else n
    todo = without_preview(session, top_candidates(session, run_id, n))
    return enqueue_previews(session, settings, todo, AUTO_PREVIEW_PRIORITY)
