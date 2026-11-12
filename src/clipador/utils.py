"""Small helpers shared across modules."""

from __future__ import annotations

import re
from datetime import UTC, datetime


def utcnow() -> datetime:
    """Timezone-aware UTC now (SQLModel's UTCDateTime columns require tz info)."""
    return datetime.now(UTC)


_TIMECODE_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")
_HMS_RE = re.compile(r"^(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+(?:\.\d+)?)s)?$", re.IGNORECASE)


def parse_timecode(value: str | None) -> float | None:
    """Parse '1:02:03', '62:03', '3723', '1h2m3s' into seconds. Empty -> None.

    Raises ValueError with a PT-BR message for anything else.
    """
    if value is None:
        return None
    text = value.strip().replace(",", ".")
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        pass
    else:
        if seconds < 0:
            raise ValueError("O tempo não pode ser negativo.")
        return seconds
    m = _TIMECODE_RE.match(text)
    if m:
        h, mnt, s = m.groups()
        return int(h or 0) * 3600 + int(mnt) * 60 + float(s)
    m = _HMS_RE.match(text)
    if m and any(m.groups()):
        h, mnt, s = m.groups()
        return int(h or 0) * 3600 + int(mnt or 0) * 60 + float(s or 0)
    raise ValueError(f"Tempo inválido: '{value}'. Use 1:02:03, 62:03, 3723 ou 1h2m3s.")


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    total = round(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def format_hms(seconds: float | None) -> str:
    """Always HH:MM:SS (timestamps in a long VOD)."""
    if seconds is None:
        return "—"
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def tail_lines(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])
