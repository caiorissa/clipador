"""ffmpeg / ffprobe helpers: probe, encoder detection, thumbnails."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from clipador.errors import ToolError

log = logging.getLogger(__name__)

NVENC = "h264_nvenc"
X264 = "libx264"


@dataclass
class MediaInfo:
    duration: float
    width: int | None
    height: int | None
    fps: float | None
    video_codec: str | None
    audio_codec: str | None
    size_bytes: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
        )
    except FileNotFoundError as e:
        raise ToolError(
            f"'{cmd[0]}' nÃ£o foi encontrado. Instale o ffmpeg e coloque-o no PATH "
            "(rode `clipador doctor`).",
            detail=str(e),
        ) from e
    except subprocess.TimeoutExpired as e:
        raise ToolError(f"'{cmd[0]}' demorou demais para responder.", detail=str(e)) from e


def _parse_fps(rate: str | None) -> float | None:
    if not rate or rate in ("0/0", "0"):
        return None
    try:
        num, _, den = rate.partition("/")
        return round(float(num) / float(den or 1), 3)
    except (ValueError, ZeroDivisionError):
        return None


def probe(path: Path, ffprobe: str = "ffprobe") -> MediaInfo:
    """Read duration/resolution/codecs. Raises ToolError if the file is missing or unreadable."""
    if not path.is_file():
        raise ToolError(f"Arquivo nÃ£o encontrado: {path.name}", detail=str(path))
    cmd = [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams"]
    res = _run([*cmd, str(path)], timeout=60)
    if res.returncode != 0:
        raise ToolError(
            "O ffprobe nÃ£o conseguiu ler o vÃ­deo (arquivo corrompido ou incompleto).",
            detail=res.stderr.strip()[-2000:],
        )
    data = json.loads(res.stdout or "{}")
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    a = next((s for s in streams if s.get("codec_type") == "audio"), {})
    duration = float(fmt.get("duration") or v.get("duration") or 0)
    if duration <= 0:
        raise ToolError("O vÃ­deo nÃ£o tem duraÃ§Ã£o vÃ¡lida.", detail=json.dumps(fmt)[:2000])
    return MediaInfo(
        duration=duration,
        width=v.get("width"),
        height=v.get("height"),
        fps=_parse_fps(v.get("avg_frame_rate") or v.get("r_frame_rate")),
        video_codec=v.get("codec_name"),
        audio_codec=a.get("codec_name"),
        size_bytes=int(fmt["size"]) if fmt.get("size") else None,
    )


def try_encoder(encoder: str, ffmpeg: str = "ffmpeg") -> tuple[bool, str]:
    """Actually encode 0.2 s with `encoder`. Listing in `ffmpeg -encoders` is not enough:
    without a usable NVIDIA GPU/driver, h264_nvenc is listed but fails to open."""
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "lavfi", "-i", "testsrc2=size=256x256:rate=30", "-t", "0.2",
        "-c:v", encoder, "-f", "null", "-",
    ]  # fmt: skip
    try:
        res = _run(cmd, timeout=30)
    except ToolError as e:
        return False, e.message
    if res.returncode == 0:
        return True, ""
    return False, (res.stderr.strip().splitlines() or ["erro desconhecido"])[-1]


def pick_encoder(ffmpeg: str = "ffmpeg") -> str:
    """h264_nvenc if it really works on this machine, else libx264."""
    ok, err = try_encoder(NVENC, ffmpeg)
    if ok:
        return NVENC
    log.info("NVENC indisponÃ­vel (%s); usando libx264", err)
    return X264


def make_thumbnail(video: Path, out: Path, at_seconds: float, ffmpeg: str = "ffmpeg") -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-ss", f"{max(0.0, at_seconds):.2f}", "-i", str(video),
        "-frames:v", "1", "-vf", "scale=480:-2", str(out),
    ]  # fmt: skip
    res = _run(cmd, timeout=60)
    if res.returncode != 0 or not out.is_file():
        raise ToolError("NÃ£o foi possÃ­vel gerar a miniatura.", detail=res.stderr.strip()[-2000:])
    return out


def tool_version(binary: str) -> str | None:
    """First line of `<binary> -version`, or None if not runnable."""
    try:
        res = _run([binary, "-version"], timeout=15)
    except ToolError:
        return None
    if res.returncode != 0:
        return None
    return (res.stdout.splitlines() or [""])[0].strip()
