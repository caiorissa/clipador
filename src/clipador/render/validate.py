"""Post-render checks with ffprobe (+ faststart by reading the MP4 atom order)."""

from __future__ import annotations

import json
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProbeResult:
    width: int
    height: int
    vcodec: str
    acodec: str | None
    duration: float
    size: int
    faststart: bool


def moov_before_mdat(path: Path) -> bool:
    """True when the 'moov' atom comes before 'mdat' (playback can start before download ends)."""
    with open(path, "rb") as f:
        while True:
            head = f.read(8)
            if len(head) < 8:
                return False
            size, kind = struct.unpack(">I4s", head)
            if kind == b"moov":
                return True
            if kind == b"mdat":
                return False
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0]
                f.seek(size - 16, 1)
            elif size < 8:
                return False
            else:
                f.seek(size - 8, 1)


def probe_output(path: Path, ffprobe: str = "ffprobe") -> ProbeResult:
    cmd = [ffprobe, "-v", "error", "-show_entries",
           "stream=codec_type,codec_name,width,height:format=duration,size", "-of", "json",
           str(path)]  # fmt: skip
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise ValueError(f"ffprobe não conseguiu ler o arquivo: {res.stderr.strip()[-300:]}")
    data = json.loads(res.stdout or "{}")
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v is None:
        raise ValueError("o arquivo não tem vídeo")
    fmt = data.get("format", {})
    return ProbeResult(
        int(v.get("width", 0)), int(v.get("height", 0)), v.get("codec_name", ""),
        a.get("codec_name") if a else None, float(fmt.get("duration") or 0),
        int(fmt.get("size") or 0), moov_before_mdat(path),
    )  # fmt: skip


def validate_render(
    path: Path, width: int, height: int, duration: float, tolerance: float = 0.15,
    ffprobe: str = "ffprobe",
) -> tuple[ProbeResult, list[str]]:  # fmt: skip
    """Returns (probe, problems in PT-BR). Empty list = valid."""
    if not path.is_file() or path.stat().st_size == 0:
        return ProbeResult(0, 0, "", None, 0, 0, False), ["o arquivo final está vazio"]
    try:
        p = probe_output(path, ffprobe)
    except (ValueError, json.JSONDecodeError) as e:
        return ProbeResult(0, 0, "", None, 0, 0, False), [str(e)]
    problems = []
    if (p.width, p.height) != (width, height):
        problems.append(f"resolução {p.width}x{p.height}, esperado {width}x{height}")
    if p.vcodec != "h264":
        problems.append(f"codec de vídeo {p.vcodec}, esperado H.264")
    if p.acodec != "aac":
        problems.append("sem trilha de áudio AAC" if p.acodec is None else f"áudio {p.acodec}")
    if abs(p.duration - duration) > tolerance:
        problems.append(f"duração {p.duration:.2f}s, esperado {duration:.2f}s (±{tolerance}s)")
    if not p.faststart:
        problems.append("sem faststart (moov depois do mdat)")
    return p, problems
