"""Exceptions carrying a user-facing PT-BR message plus technical detail for the log."""

from __future__ import annotations


class ClipadorError(Exception):
    """Base error. `message` is shown to the user (PT-BR); `detail` goes to logs only."""

    def __init__(self, message: str, *, detail: str | None = None, retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.retryable = retryable


class ValidationError(ClipadorError):
    """Invalid user input or a violated business rule."""


class ToolError(ClipadorError):
    """An external tool (ffmpeg, ffprobe, yt-dlp) failed or is missing."""


class JobCanceled(Exception):
    """Raised inside a running stage when the user canceled the job."""
