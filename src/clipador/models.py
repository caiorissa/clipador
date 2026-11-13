"""Database tables. All state lives here; big files live on disk (paths relative to data_dir)."""

# NOTE: no `from __future__ import annotations` here; SQLModel relationships need real annotations.
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, Column, Text, UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel

from clipador.utils import utcnow


class Platform(StrEnum):
    TWITCH = "twitch"
    YOUTUBE = "youtube"
    KICK = "kick"


PLATFORM_LABELS = {Platform.TWITCH: "Twitch", Platform.YOUTUBE: "YouTube", Platform.KICK: "Kick"}


class SourceKind(StrEnum):
    VOD = "vod"
    CLIP = "clip"


class Layout(StrEnum):
    BLUR = "blur"
    FACECAM = "facecam_embaixo"
    FULL = "jogo_cheio"


class SourceStatus(StrEnum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    READY = "ready"
    FAILED = "failed"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELED = "canceled"


ACTIVE_JOB_STATUSES = (JobStatus.QUEUED, JobStatus.RUNNING)


class Streamer(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    display_name: str = Field(index=True)
    permission_confirmed: bool = False
    permission_date: date | None = None
    permission_note: str = ""
    credit_text: str = ""
    default_layout: str = Field(default=Layout.BLUR)  # blur | facecam_embaixo | jogo_cheio
    default_caption_style: str | None = None  # key in config/caption_styles.toml
    # Facecam rectangle NORMALIZED to the frame (x, y, w, h in 0..1): valid at any resolution.
    facecam_box: list[float] | None = Field(default=None, sa_column=Column(JSON))
    show_credit: bool = True  # "via @handle" on screen
    show_opening_text: bool = True
    active: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    channels: list["StreamerChannel"] = Relationship(
        back_populates="streamer",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "order_by": "StreamerChannel.id"},
    )

    @property
    def facecam_rect(self) -> tuple[float, float, float, float] | None:
        b = self.facecam_box
        if not b or len(b) != 4 or b[2] <= 0 or b[3] <= 0:
            return None
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


class StreamerChannel(SQLModel, table=True):
    __tablename__ = "streamer_channel"
    __table_args__ = (
        UniqueConstraint("platform", "channel_handle", name="uq_channel_platform_handle"),
    )

    id: int | None = Field(default=None, primary_key=True)
    streamer_id: int = Field(foreign_key="streamer.id", index=True)
    platform: str
    # Normalized: lowercase, no leading '@' (YouTube channel ids "UC..." keep their case).
    channel_handle: str
    created_at: datetime = Field(default_factory=utcnow)

    streamer: Streamer | None = Relationship(back_populates="channels")


class Source(SQLModel, table=True):
    """A VOD or clip to analyze."""

    __table_args__ = (
        UniqueConstraint("platform", "kind", "external_id", name="uq_source_external"),
    )

    id: int | None = Field(default=None, primary_key=True)
    streamer_id: int = Field(foreign_key="streamer.id", index=True)
    platform: str
    kind: str
    external_id: str
    url: str
    title: str | None = None
    channel_name: str | None = None
    published_at: datetime | None = None
    duration_seconds: float | None = None
    thumbnail_url: str | None = None
    thumbnail_path: str | None = None  # relative to data_dir
    file_path: str | None = None  # relative to data_dir
    media_info: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    status: str = Field(default=SourceStatus.PENDING, index=True)
    # Optional range to analyze (seconds). The file is NOT cut; later stages honor it.
    start_offset: float | None = None
    end_offset: float | None = None
    game: str | None = None  # e.g. "GTA 6" (None = settings.default_game); used by the LLM
    # Non-blocking notice (e.g. channel does not match the streamer's registered channels).
    warning: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    streamer: Streamer | None = Relationship()


class Job(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    type: str = Field(index=True)
    source_id: int | None = Field(default=None, foreign_key="source.id", index=True)
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    status: str = Field(default=JobStatus.QUEUED, index=True)
    progress: float = 0.0
    attempts: int = 0
    max_attempts: int = 3
    error: str | None = Field(default=None, sa_column=Column(Text))
    log: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    needs_gpu: bool = False
    # Higher runs first (ties: oldest first). Review prefetch uses a negative priority so
    # renders asked for by hand always go ahead of it.
    priority: int = Field(default=0, sa_column_kwargs={"server_default": "0"})
    locked_at: datetime | None = None  # also used as heartbeat while running
    locked_by: str | None = None
    run_after: datetime = Field(default_factory=utcnow, index=True)  # retry backoff
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    source: Source | None = Relationship()


# --- Analysis (step 2) -----------------------------------------------------------------------


class ArtifactStatus(StrEnum):
    OK = "ok"
    SKIPPED = "skipped"  # e.g. no chat available: not an error, later stages run without it


class StageArtifact(SQLModel, table=True):
    """Cache record of a stage's output for a source: valid while `params_hash` matches."""

    __tablename__ = "stage_artifact"
    __table_args__ = (UniqueConstraint("source_id", "stage", name="uq_artifact_source_stage"),)

    id: int | None = Field(default=None, primary_key=True)
    source_id: int = Field(foreign_key="source.id", index=True)
    stage: str
    params_hash: str
    status: str = Field(default=ArtifactStatus.OK)
    message: str | None = None  # e.g. reason for skipped
    files: dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    summary: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    elapsed_seconds: float | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class AnalysisRun(SQLModel, table=True):
    __tablename__ = "analysis_run"

    id: int | None = Field(default=None, primary_key=True)
    source_id: int = Field(foreign_key="source.id", index=True)
    params_hash: str = Field(index=True)
    params: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    candidate_count: int = 0
    elapsed_seconds: float | None = None
    has_chat: bool = False
    created_at: datetime = Field(default_factory=utcnow)


class CandidateStatus(StrEnum):
    SUGGESTED = "suggested"
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"  # review: decide later (comes back at the end of the queue)
    RENDERED = "rendered"  # final render done


RENDERABLE_STATUSES = (CandidateStatus.APPROVED, CandidateStatus.RENDERED)
PENDING_STATUSES = (CandidateStatus.SUGGESTED, CandidateStatus.SKIPPED)


class Candidate(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    source_id: int = Field(foreign_key="source.id", index=True)
    analysis_run_id: int = Field(foreign_key="analysis_run.id", index=True)
    rank: int = 0
    start: float  # seconds, relative to the original video
    end: float
    peak_t: float
    score: float  # 0-100
    breakdown: dict[str, float] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    transcript_snippet: str = Field(
        default="", sa_column=Column(Text, nullable=False, server_default="")
    )
    status: str = Field(default=CandidateStatus.SUGGESTED)
    user_start: float | None = None
    user_end: float | None = None
    # Detection score combined with the LLM score (step 3); None until ranked.
    final_score: float | None = None
    final_rank: int | None = None
    # Step 4 (render): per-candidate choices (None = streamer default).
    caption_overrides: dict[str, str] | None = Field(default=None, sa_column=Column(JSON))
    overlay_title: str | None = None  # opening text (None = suggestion from the LLM/draft)
    render_layout: str | None = None
    render_style: str | None = None  # caption style key
    # {"opening": bool, "credit": bool, "zoom": float, "foco_x": float, "peak_zoom": bool}
    render_options: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    # Step 5 (review): when it was approved/rejected/skipped; pre-publish checks (5B).
    reviewed_at: datetime | None = None
    checks: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @property
    def eff_start(self) -> float:
        return self.user_start if self.user_start is not None else self.start

    @property
    def eff_end(self) -> float:
        return self.user_end if self.user_end is not None else self.end


class Verdict(StrEnum):
    GOOD = "bom"
    BAD = "ruim"


class CandidateFeedback(SQLModel, table=True):
    __tablename__ = "candidate_feedback"

    id: int | None = Field(default=None, primary_key=True)
    candidate_id: int = Field(foreign_key="candidate.id", unique=True)
    verdict: str
    note: str = ""
    reason_tag: str | None = None  # rejection reason (services.review.REJECT_REASONS)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class DecisionAction(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    SKIP = "skip"
    UNDO = "undo"
    EDIT_RANGE = "edit_range"
    REOPEN = "reopen"


class Decision(SQLModel, table=True):
    """Review history (a global, persisted undo stack). `prev_state` is the candidate state
    right before the action, so undo can restore it exactly."""

    id: int | None = Field(default=None, primary_key=True)
    candidate_id: int = Field(foreign_key="candidate.id", index=True)
    action: str
    reason_tag: str | None = None
    note: str = ""
    prev_state: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utcnow, index=True)


# --- LLM enrichment (step 3) -----------------------------------------------------------------


class EnrichStatus(StrEnum):
    OK = "ok"
    LLM_FAILED = "llm_failed"
    SKIPPED = "skipped"


class CandidateEnrichment(SQLModel, table=True):
    """LLM judgement + texts for one candidate (latest wins; `input_hash` is the cache key)."""

    __tablename__ = "candidate_enrichment"

    id: int | None = Field(default=None, primary_key=True)
    candidate_id: int = Field(foreign_key="candidate.id", unique=True)
    source_id: int = Field(foreign_key="source.id", index=True)
    prompt_version: str
    model: str
    input_hash: str = Field(index=True)
    instruction: str = ""
    scores: dict[str, float] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    motivo: str = ""
    precisa_de_contexto: bool = False
    alertas: list[dict[str, str]] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    titulos: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    texto_de_abertura: str | None = None
    descricao: str = ""
    hashtags: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    status: str = Field(default=EnrichStatus.OK)
    error: str | None = Field(default=None, sa_column=Column(Text))
    latency_ms: int | None = None
    tokens: int | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @property
    def nota_geral(self) -> float | None:
        if self.status != EnrichStatus.OK:
            return None
        v = self.scores.get("nota_geral")
        return float(v) if v is not None else None


class PostDraft(SQLModel, table=True):
    __tablename__ = "post_draft"
    __table_args__ = (UniqueConstraint("candidate_id", "platform", name="uq_draft_platform"),)

    id: int | None = Field(default=None, primary_key=True)
    candidate_id: int = Field(foreign_key="candidate.id", index=True)
    platform: str  # youtube | tiktok | instagram
    title: str = ""
    description: str = Field(default="", sa_column=Column(Text, nullable=False, server_default=""))
    hashtags: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    edited_by_user: bool = False
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# --- Rendering (step 4) -------------------------------------------------------------------------


class RenderKind(StrEnum):
    PREVIEW = "preview"
    FINAL = "final"


class RenderStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELED = "canceled"


class Render(SQLModel, table=True):
    """One preview or final render of a candidate (cached by params_hash)."""

    id: int | None = Field(default=None, primary_key=True)
    candidate_id: int = Field(foreign_key="candidate.id", index=True)
    source_id: int = Field(foreign_key="source.id", index=True)
    job_id: int | None = Field(default=None, foreign_key="job.id")
    kind: str = Field(default=RenderKind.FINAL)
    params_hash: str = Field(index=True)
    params: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    path: str | None = None  # relative to data_dir
    package_path: str | None = None
    thumbnail_path: str | None = None
    duration: float | None = None
    size_bytes: int | None = None
    status: str = Field(default=RenderStatus.QUEUED, index=True)
    error: str | None = Field(default=None, sa_column=Column(Text))
    warning: str | None = None
    encoder: str | None = None
    render_seconds: float | None = None
    loudness_lufs: float | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
