"""Jump cuts: remove dead air inside a clip (the standard Shorts/TikTok edit that keeps people
watching), conservatively.

A stretch is cut only when ALL of this holds:
- no transcribed word in it, for at least `min_gap` seconds;
- its audio stays clearly quieter than the streamer's voice in the same clip (so gameplay
  action - a crash, an explosion, a chase - is never cut just because nobody talks);
- it is not near the peak of the moment.
`pad` seconds are kept on each side so cuts don't clip breaths/word tails. Captions, the zoom
peak and the opening overlay are remapped to the shortened timeline.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np

FPS = 20  # audio analysis frames per second
SR = 16000


@dataclass(frozen=True)
class CutParams:
    min_gap: float = 1.0  # shortest silence worth cutting (s)
    pad: float = 0.2  # kept on each side of a cut (s)
    quiet_margin_db: float = 8.0  # gap must stay this much below the median voice level
    peak_guard: float = 2.0  # never cut within this many seconds of the peak
    min_saving: float = 0.4  # ignore cuts shorter than this (s)


def clip_db(video, seek: float, duration: float, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """dBFS per 1/FPS s of the clip's audio (one fast ffmpeg pass, mono 16 kHz)."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{seek:.3f}",
           "-t", f"{duration:.3f}", "-i", str(video), "-vn", "-ac", "1", "-ar", str(SR),
           "-f", "s16le", "-"]  # fmt: skip
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=120).stdout
    except (OSError, subprocess.TimeoutExpired):
        return np.zeros(0)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    hop = SR // FPS
    n = x.size // hop
    if n == 0:
        return np.zeros(0)
    ms = np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)
    return 10 * np.log10(ms + 1e-12)


def keep_intervals(
    words: list[tuple[float, float]],
    db: np.ndarray,
    duration: float,
    peak: float | None = None,
    p: CutParams | None = None,
) -> list[tuple[float, float]]:
    """Clip-local [a, b) spans to KEEP, in order. One span [0, duration] = nothing to cut.
    `words` are (start, end) in clip-local seconds."""
    p = p or CutParams()
    whole = [(0.0, duration)]
    if not words or db.size == 0:
        return whole
    words = sorted((max(0.0, a), min(duration, b)) for a, b in words if b > 0 and a < duration)
    if not words:
        return whole
    idx = [(int(a * FPS), max(int(a * FPS) + 1, int(np.ceil(b * FPS)))) for a, b in words]
    voiced = np.concatenate([db[i:j] for i, j in idx if i < db.size]) if idx else np.zeros(0)
    if voiced.size < FPS // 2:
        return whole
    quiet_level = float(np.median(voiced)) - p.quiet_margin_db

    gaps = []  # silent stretches between/around words
    edges = [(0.0, words[0][0])]
    edges += [(words[i][1], words[i + 1][0]) for i in range(len(words) - 1)]
    edges.append((words[-1][1], duration))
    for a, b in edges:
        if b - a < p.min_gap:
            continue
        ca, cb = a + p.pad if a > 0 else a, b - p.pad if b < duration else b
        if peak is not None and ca < peak + p.peak_guard and cb > peak - p.peak_guard:
            continue
        if cb - ca < p.min_saving:
            continue
        seg = db[int(ca * FPS) : int(np.ceil(cb * FPS))]
        if seg.size and float(seg.max()) < quiet_level:
            gaps.append((ca, cb))
    if not gaps:
        return whole
    keeps, t = [], 0.0
    for a, b in gaps:
        if a > t:
            keeps.append((round(t, 3), round(a, 3)))
        t = b
    if t < duration:
        keeps.append((round(t, 3), round(duration, 3)))
    return [(a, b) for a, b in keeps if b - a > 0.05] or whole


def is_trivial(keeps: list[tuple[float, float]], duration: float) -> bool:
    return len(keeps) == 1 and keeps[0][0] <= 0.01 and keeps[0][1] >= duration - 0.01


def total(keeps: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in keeps)


def remap(t: float, keeps: list[tuple[float, float]]) -> float:
    """Clip-local time -> time in the cut clip (a time inside a removed span snaps to the
    next kept frame)."""
    out = 0.0
    for a, b in keeps:
        if t < a:
            return out
        if t <= b:
            return out + (t - a)
        out += b - a
    return out


def cut_filters(keeps: list[tuple[float, float]]) -> tuple[list[str], str, str]:
    """filter_complex parts that trim the input to `keeps` and join them; returns (parts,
    video label, audio label) to use instead of [0:v] / [0:a]."""
    k = len(keeps)
    parts = [f"[0:v]split={k}" + "".join(f"[jv{i}]" for i in range(k)),
             f"[0:a]asplit={k}" + "".join(f"[ja{i}]" for i in range(k))]  # fmt: skip
    pairs = []
    for i, (a, b) in enumerate(keeps):
        parts.append(f"[jv{i}]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[jvt{i}]")
        parts.append(f"[ja{i}]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[jat{i}]")
        pairs.append(f"[jvt{i}][jat{i}]")
    parts.append("".join(pairs) + f"concat=n={k}:v=1:a=1[jcv][jca]")
    return parts, "[jcv]", "[jca]"
