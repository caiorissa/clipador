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
