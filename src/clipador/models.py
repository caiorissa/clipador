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
    FACECAM = "facecam"


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
    default_layout: str = Field(default=Layout.BLUR)
    # Optional facecam rectangle in source-video pixels.
    facecam_x: int | None = None
    facecam_y: int | None = None
    facecam_w: int | None = None
    facecam_h: int | None = None
    active: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    channels: list["StreamerChannel"] = Relationship(
        back_populates="streamer",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "order_by": "StreamerChannel.id"},
    )

    @property
    def facecam_rect(self) -> tuple[int, int, int, int] | None:
        vals = (self.facecam_x, self.facecam_y, self.facecam_w, self.facecam_h)
        if any(v is None for v in vals):
            return None
        return vals  # type: ignore[return-value]


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
    # Step 4 (render): user choices for the vertical clip.
    subtitle_text: str | None = Field(default=None, sa_column=Column(Text))  # corrected words
    overlay_title: str | None = None  # text on top of the clip (None = suggestion)
    render_layout: str | None = None  # blur | facecam (None = streamer default)
    render_style: str | None = None  # subtitle style key
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
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


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


class RenderedClip(SQLModel, table=True):
    """Latest rendered vertical clip of a candidate (re-rendering replaces it)."""

    __tablename__ = "rendered_clip"

    id: int | None = Field(default=None, primary_key=True)
    candidate_id: int = Field(foreign_key="candidate.id", unique=True)
    source_id: int = Field(foreign_key="source.id", index=True)
    params_hash: str
    layout: str
    style: str
    file_path: str  # relative to data_dir
    package_path: str | None = None  # .txt with title/description/hashtags per platform
    thumbnail_path: str | None = None
    duration_seconds: float
    encoder: str
    render_seconds: float | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
