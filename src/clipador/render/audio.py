"""Loudness for clips: measured linear gain + peak limiter (final) / loudnorm (preview)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from clipador.errors import ToolError

LRA = 11.0
# Mono sources are upmixed first: loudness meters disagree by ~6 dB on mono, and the platforms
# expect stereo anyway.
STEREO = "aformat=channel_layouts=stereo"
MAX_GAIN_DB = 40.0


def gain_chain(gain_db: float, true_peak: float) -> str:
    limit = 10 ** (true_peak / 20)
    return (
        f"{STEREO},volume={gain_db:.2f}dB,"
        f"alimiter=limit={limit:.4f}:attack=5:release=60:level=false"
    )


def loudnorm_filter(
    target: float, true_peak: float, gain_db: float | None, duration: float, fade: float
) -> str:
    """Audio chain with short fades against clicks at the cut points, 48 kHz.

    Final (gain_db given, see `final_gain`): exact linear gain + peak limiter.
    Preview (gain_db None): single-pass dynamic loudnorm (fast, approximate).
    """
    st = max(0.0, duration - fade)
    fades = f"aresample=48000,afade=t=in:d={fade},afade=t=out:st={st:.3f}:d={fade}"
    if gain_db is not None:
        return f"{gain_chain(gain_db, true_peak)},{fades}"
    return f"{STEREO},loudnorm=I={target}:TP={true_peak}:LRA={LRA},{fades}"


_I_RE = re.compile(r"Integrated loudness:\s+I:\s+(-?[\d.]+|-inf)\s+LUFS", re.S)


def parse_integrated(stderr: str) -> float | None:
    m = _I_RE.search(stderr)
    if not m or m.group(1) == "-inf":
        return None
    return float(m.group(1))


def _measure(video: Path, start: float, duration: float, chain: str, ffmpeg: str) -> float | None:
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
           "-i", str(video), "-vn", "-af", f"{chain},ebur128", "-f", "null", "-"]  # fmt: skip
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise ToolError("Falha ao medir o volume do trecho.", detail=res.stderr[-2000:])
    return parse_integrated(res.stderr)


def final_gain(
    video: Path, start: float, duration: float, target: float, true_peak: float, ffmpeg: str,
    tolerance: float = 0.5, rounds: int = 3,
) -> float:  # fmt: skip
    """Gain (dB) so that gain + limiter lands on `target` LUFS. Audio-only passes (fast).

    A single linear step is not enough when loud peaks get limited (a scream next to quiet
    talk carries most of the energy), so the result is re-measured and corrected."""
    first = _measure(video, start, duration, STEREO, ffmpeg)
    if first is None:  # silence
        return 0.0
    gain = max(-30.0, min(MAX_GAIN_DB, target - first))
    for _ in range(rounds):
        got = _measure(video, start, duration, gain_chain(gain, true_peak), ffmpeg)
        if got is None or abs(got - target) <= tolerance:
            break
        gain = max(-30.0, min(MAX_GAIN_DB, gain + (target - got)))
    return round(gain, 2)


def integrated_loudness(path: Path, ffmpeg: str = "ffmpeg") -> float | None:
    """Integrated loudness (LUFS) of a file, via ebur128."""
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-i", str(path), "-vn", "-af", "ebur128",
           "-f", "null", "-"]  # fmt: skip
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return parse_integrated(res.stderr)
