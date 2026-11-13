"""Amplitude peaks of work/<id>/audio.wav for the review timeline.

The WAV is memory-mapped (np.memmap over the 'data' chunk): only the samples of the requested
window are touched, never the whole 6 h file. Peaks are computed in fixed 60 s tiles and cached
as small .npy files, so moving the clip edges (a new window) reuses almost everything.

Times in and out are relative to the ORIGINAL video: the WAV starts at the source's start_offset
(stored as `offset` in the extract_audio artifact).
"""

from __future__ import annotations

import contextlib
import math
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import Engine

from clipador.config import Settings
from clipador.errors import ToolError
from clipador.services.artifacts import get_artifact

TILE_SECONDS = 60
BINS_PER_SECOND = 20  # 50 ms per bar
PAD_SECONDS = 60.0


def wav_layout(path: Path) -> tuple[int, int, int]:
    """(byte offset of the samples, number of samples, sample rate) of a mono s16 WAV."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ToolError("Arquivo de áudio inválido (não é WAV).", detail=str(path))
        sr = None
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                raise ToolError("WAV sem bloco de dados.", detail=str(path))
            cid, n = hdr[:4], struct.unpack("<I", hdr[4:])[0]
            if cid == b"fmt ":
                fmt = f.read(n)
                channels, sr = struct.unpack("<HI", fmt[2:8])
                bits = struct.unpack("<H", fmt[14:16])[0]
                if channels != 1 or bits != 16:
                    raise ToolError("WAV inesperado (precisa ser mono 16 bits).", detail=str(path))
                if n & 1:
                    f.seek(1, 1)
            elif cid == b"data":
                if sr is None:
                    raise ToolError("WAV sem cabeçalho de formato.", detail=str(path))
                start = f.tell()
                return start, min(n, size - start) // 2, sr
            else:
                f.seek(n + (n & 1), 1)


def compute_tile(path: Path, k: int, bps: int = BINS_PER_SECOND,
                 tile_s: int = TILE_SECONDS) -> np.ndarray:  # fmt: skip
    """Peak |amplitude| (0-255) per bin for WAV seconds [k*tile_s, (k+1)*tile_s)."""
    offset, n, sr = wav_layout(path)
    per_bin = max(1, sr // bps)
    a = k * tile_s * sr
    b = min(n, (k + 1) * tile_s * sr)
    if b <= a:
        return np.zeros(0, dtype=np.uint8)
    mm = np.memmap(path, dtype="<i2", mode="r", offset=offset, shape=(n,))
    try:
        seg = np.abs(mm[a:b].astype(np.int32))
    finally:
        del mm
    bins = math.ceil(seg.size / per_bin)
    seg = np.pad(seg, (0, bins * per_bin - seg.size))
    peaks = seg.reshape(bins, per_bin).max(axis=1)
    return np.minimum(255, np.round(peaks / 32768 * 255)).astype(np.uint8)


def _tile(cache_dir: Path, key: str, path: Path, k: int, bps: int) -> np.ndarray:
    f = cache_dir / f"{key}_{bps}_{k}.npy"
    if f.is_file():
        with contextlib.suppress(OSError, ValueError):
            return np.load(f)
    peaks = compute_tile(path, k, bps)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, peaks)
    tmp.replace(f)
    return peaks


def waveform(
    engine: Engine,
    settings: Settings,
    source_id: int,
    start: float,
    end: float,
    *,
    pad: float = PAD_SECONDS,
    bps: int = BINS_PER_SECOND,
) -> dict[str, Any]:
    """Peaks for [start - pad, end + pad] (original-video seconds). `t0` is the time of the
    first bar, `step` the bar width; `peaks` are 0-255."""
    empty = {"t0": max(0.0, start - pad), "step": 1 / bps, "peaks": [], "scale": 255}
    art = get_artifact(engine, source_id, "extract_audio")
    if art is None or "audio" not in art.files:
        return {**empty, "error": "Áudio ainda não extraído (analise a fonte)."}
    path = settings.abs_path(art.files["audio"])
    if not path.is_file():
        return {**empty, "error": "Arquivo de áudio não encontrado."}
    offset = float((art.summary or {}).get("offset") or 0.0)
    _off, n, sr = wav_layout(path)
    dur = n / sr
    wa = max(0.0, start - pad - offset)
    wb = min(dur, end + pad - offset)
    if wb <= wa:
        return {**empty, "offset": offset}
    k0, k1 = int(wa // TILE_SECONDS), math.ceil(wb / TILE_SECONDS)
    key = art.params_hash  # a new extraction (other offsets/file) never reuses old tiles
    cache_dir = settings.source_work_dir(source_id) / "waveform"
    tiles = [_tile(cache_dir, key, path, k, bps) for k in range(k0, k1)]
    allp = np.concatenate(tiles) if tiles else np.zeros(0, dtype=np.uint8)
    i0 = round((wa - k0 * TILE_SECONDS) * bps)
    i1 = round((wb - k0 * TILE_SECONDS) * bps)
    peaks = allp[i0:i1]
    return {
        "t0": round(offset + k0 * TILE_SECONDS + i0 / bps, 3),
        "step": 1 / bps,
        "peaks": peaks.tolist(),
        "scale": 255,
        "offset": offset,
    }
