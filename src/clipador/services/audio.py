"""Audio extraction (ffmpeg -> 16 kHz mono s16 WAV on disk) and block-wise WAV reading.

A 6 h VOD is ~700 MB as s16 and would be ~1.4 GB as float32, so nothing reads the whole file:
features and transcription stream it in blocks.
"""

from __future__ import annotations

import contextlib
import subprocess
import wave
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from clipador.errors import ToolError

SAMPLE_RATE = 16000


def extract_wav(
    video: Path,
    out: Path,
    *,
    start: float | None,
    end: float | None,
    duration_hint: float | None,
    progress_cb: Callable[[float], None],
    ffmpeg: str = "ffmpeg",
) -> None:
    """Write `out` (16 kHz mono PCM s16) for [start, end) of `video`. Atomic (tmp + rename)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.wav")
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video)]
    if end is not None:
        cmd += ["-t", f"{end - (start or 0):.3f}"]
    cmd += [
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
        "-progress", "pipe:1", "-nostats", str(tmp),
    ]  # fmt: skip
    total = (end - (start or 0)) if end is not None else (duration_hint or 0) - (start or 0)
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace",
        )  # fmt: skip
    except FileNotFoundError as e:
        raise ToolError("ffmpeg não encontrado (rode `clipador doctor`).", detail=str(e)) from e
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            if key == "out_time_us" and total > 0 and value.isdigit():
                progress_cb(min(99.0, int(value) / 1e6 / total * 100))
        stderr = proc.stderr.read() if proc.stderr else ""
        proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    if proc.returncode != 0 or not tmp.is_file():
        raise ToolError(
            "O ffmpeg não conseguiu extrair o áudio do vídeo (sem trilha de áudio ou arquivo "
            "corrompido).",
            detail=stderr[-3000:],
        )
    tmp.replace(out)


class WavReader:
    """Read a mono s16 WAV in blocks without loading it all."""

    def __init__(self, path: Path):
        self.path = path
        with wave.open(str(path), "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise ToolError("WAV inesperado (precisa ser mono 16 bits).", detail=str(path))
            self.sample_rate = w.getframerate()
            self.n_frames = w.getnframes()

    @property
    def duration(self) -> float:
        return self.n_frames / self.sample_rate

    def read(self, t0: float, t1: float) -> np.ndarray:
        """float32 samples in [-1, 1) for [t0, t1) seconds."""
        a = max(0, int(t0 * self.sample_rate))
        b = min(self.n_frames, int(t1 * self.sample_rate))
        if b <= a:
            return np.zeros(0, dtype=np.float32)
        with wave.open(str(self.path), "rb") as w:
            w.setpos(a)
            raw = w.readframes(b - a)
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    def blocks(self, block_s: float = 60.0) -> Iterator[tuple[float, np.ndarray]]:
        t = 0.0
        while t < self.duration:
            yield t, self.read(t, t + block_s)
            t += block_s

    def write_slice(self, t0: float, t1: float, out: Path) -> None:
        """Copy [t0, t1) into a new WAV (used to feed transcription chunks)."""
        a = max(0, int(t0 * self.sample_rate))
        b = min(self.n_frames, int(t1 * self.sample_rate))
        with wave.open(str(self.path), "rb") as src, wave.open(str(out), "wb") as dst:
            dst.setnchannels(1)
            dst.setsampwidth(2)
            dst.setframerate(self.sample_rate)
            src.setpos(a)
            remaining = b - a
            while remaining > 0:
                n = min(remaining, self.sample_rate * 30)
                dst.writeframes(src.readframes(n))
                remaining -= n
