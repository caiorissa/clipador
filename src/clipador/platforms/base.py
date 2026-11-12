"""Common interface for source platforms (Twitch, YouTube, Kick)."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import SplitResult, urlsplit

from clipador.errors import ValidationError
from clipador.models import Platform, Source, SourceKind
from clipador.services.downloader import (
    DownloadError,
    DownloadErrorKind,
    DownloadResult,
    ProgressCallback,
    YtDlp,
)


class InvalidUrlError(ValidationError):
    """URL is not a supported VOD/clip link (PT-BR message)."""


@dataclass(frozen=True)
class ParsedUrl:
    platform: Platform
    kind: SourceKind
    external_id: str
    url: str  # canonical URL handed to yt-dlp
    channel_handle: str | None = None  # when the URL itself names the channel


@dataclass
class VideoMetadata:
    external_id: str
    title: str | None
    duration_seconds: float | None
    thumbnail_url: str | None
    channel_name: str | None
    published_at: datetime | None
    # Normalized identifiers of the uploading channel (handle, id, name) for matching.
    channel_candidates: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChatMessage:
    t_seconds: float  # offset from the start of the VOD
    author: str
    text: str


class ChatProvider(ABC):
    """Fetches chat replay for a source. Implementations arrive on day 2."""

    @abstractmethod
    def fetch(self, source: Source) -> list[ChatMessage]: ...


# --- URL helpers ------------------------------------------------------------------------------


def split_url(url: str) -> tuple[SplitResult, str]:
    """Split a user-pasted URL. Returns (parts, bare_host) with 'www.'/'m.' removed."""
    text = url.strip()
    if not text:
        raise InvalidUrlError("Cole uma URL.")
    if "://" not in text:
        text = "https://" + text
    parts = urlsplit(text)
    host = (parts.hostname or "").lower()
    for prefix in ("www.", "m."):
        if host.startswith(prefix):
            host = host[len(prefix) :]
    return parts, host


def path_segments(parts: SplitResult) -> list[str]:
    return [seg for seg in parts.path.split("/") if seg]


_YT_CHANNEL_ID_RE = re.compile(r"^UC[\w-]{22}$")


def normalize_handle(value: str) -> str:
    """Normalize a channel handle/URL typed by the user: 'https://twitch.tv/Fulano' -> 'fulano',
    '@Fulano' -> 'fulano'. YouTube channel ids (UC...) keep their case."""
    v = value.strip()
    if "/" in v or v.lower().startswith(("http", "www.")):
        try:
            parts, _host = split_url(v)
        except InvalidUrlError:
            return ""
        segs = path_segments(parts)
        if segs and segs[0] in ("channel", "c", "user") and len(segs) > 1:
            v = segs[1]
        elif segs:
            v = segs[0]
        else:
            v = ""
    v = v.lstrip("@").strip()
    return v if _YT_CHANNEL_ID_RE.match(v) else v.lower()


def parse_published(info: dict[str, Any]) -> datetime | None:
    ts = info.get("timestamp") or info.get("release_timestamp")
    if ts:
        try:
            return datetime.fromtimestamp(float(ts), UTC)
        except (OverflowError, OSError, ValueError):
            pass
    d = info.get("upload_date") or info.get("release_date")
    if d and len(str(d)) == 8:
        try:
            return datetime.strptime(str(d), "%Y%m%d").replace(tzinfo=UTC)
        except ValueError:
            return None
    return None


# --- adapter ----------------------------------------------------------------------------------


class PlatformAdapter(ABC):
    platform: Platform
    hosts: tuple[str, ...]  # bare hosts (no www./m.)

    def __init__(self, ytdlp: YtDlp | None = None):
        self._ytdlp = ytdlp

    @property
    def ytdlp(self) -> YtDlp:
        if self._ytdlp is None:
            from clipador.config import get_settings

            self._ytdlp = YtDlp(get_settings())
        return self._ytdlp

    @classmethod
    def detect(cls, url: str) -> bool:
        try:
            _parts, host = split_url(url)
        except InvalidUrlError:
            return False
        return host in cls.hosts

    @classmethod
    @abstractmethod
    def parse(cls, url: str) -> ParsedUrl:
        """Validate and parse. Raises InvalidUrlError with a PT-BR message."""

    def fetch_metadata(self, url: str) -> VideoMetadata:
        info = self.ytdlp.fetch_info(url)
        if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
            raise DownloadError(
                DownloadErrorKind.LIVE,
                "Esta URL é de uma live em andamento ou agendada. Use o VOD depois que terminar.",
                detail=f"live_status={info.get('live_status')}",
                retryable=False,
            )
        return self.metadata_from_info(info)

    def metadata_from_info(self, info: dict[str, Any]) -> VideoMetadata:
        raw = [
            info.get(k)
            for k in ("uploader_id", "channel_id", "channel", "uploader", "uploader_url",
                      "channel_url")
        ]  # fmt: skip
        candidates = sorted({normalize_handle(str(v)) for v in raw if v} - {""})
        duration = info.get("duration")
        return VideoMetadata(
            external_id=str(info.get("id") or ""),
            title=info.get("title") or info.get("fulltitle"),
            duration_seconds=float(duration) if duration else None,
            thumbnail_url=info.get("thumbnail"),
            channel_name=info.get("channel") or info.get("uploader") or info.get("uploader_id"),
            published_at=parse_published(info),
            channel_candidates=candidates,
        )

    def download(
        self,
        url: str,
        dest,
        progress_cb: ProgressCallback,
        log_cb=None,
    ) -> DownloadResult:
        return self.ytdlp.download(url, dest, progress_cb, log_cb)

    def chat_provider(self) -> ChatProvider | None:
        return None  # day 2

    @staticmethod
    def channel_matches(candidates: list[str], registered_handles: list[str]) -> bool:
        cands = {c.casefold() for c in candidates}
        return any(normalize_handle(h).casefold() in cands for h in registered_handles)
