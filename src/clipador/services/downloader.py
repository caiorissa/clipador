"""yt-dlp wrapper: metadata, download with progress, and error classification (PT-BR)."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from clipador.config import Settings
from clipador.errors import ClipadorError

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str | None], None]
"""(percent 0-100, optional status message). May raise JobCanceled to abort."""

# --------------------------------------------------------------------------------------------
# Progress parsing
# --------------------------------------------------------------------------------------------

_PCT_RE = re.compile(r"^\[download\]\s+(\d{1,3}(?:\.\d+)?)%")
_DEST_RE = re.compile(r"^\[download\] Destination: (.+)$")
_ALREADY_RE = re.compile(r"^\[download\] (.+) has already been downloaded")


def parse_progress_line(line: str) -> float | None:
    """Return the percentage from a yt-dlp '[download]  42.0% of ...' line, else None."""
    m = _PCT_RE.match(line.strip())
    if not m:
        return None
    return min(100.0, float(m.group(1)))


class ProgressTracker:
    """Turns yt-dlp's per-file percentages into one monotonic overall percentage.

    yt-dlp downloads video and audio as separate files (each 0-100%) and then merges them.
    The first file (video, the bulk) maps to 0-85%, later files to 85-95%, merge/post-processing
    to 95-99%. The caller sets 100% once the whole stage finishes.
    """

    def __init__(self) -> None:
        self.file_index = -1
        self.overall = 0.0

    def feed(self, line: str) -> tuple[float, str | None] | None:
        text = line.strip()
        if _DEST_RE.match(text) or _ALREADY_RE.match(text):
            if not text.lower().endswith((".jpg", ".webp", ".png")):
                self.file_index += 1
            return None
        if text.startswith(("[Merger]", "[VideoRemuxer]", "[FixupM3u8]", "[FixupM4a]")):
            return self._set(97.0, "Finalizando arquivo (merge/remux)…")
        pct = parse_progress_line(text)
        if pct is None:
            return None
        idx = max(self.file_index, 0)
        value = pct * 0.85 if idx == 0 else 85.0 + min(idx, 2) * 5.0 * (pct / 100.0)
        return self._set(value, None)

    def _set(self, value: float, msg: str | None) -> tuple[float, str | None] | None:
        value = min(99.0, value)
        if value < self.overall:
            return None
        self.overall = value
        return value, msg


# --------------------------------------------------------------------------------------------
# Error classification
# --------------------------------------------------------------------------------------------


class DownloadErrorKind(StrEnum):
    INVALID_URL = "invalid_url"
    NOT_FOUND = "not_found"
    PRIVATE = "private"
    ANTI_BOT = "anti_bot"
    NETWORK = "network"
    EXTRACTOR = "extractor"
    LIVE = "live"
    DISK = "disk"
    UNKNOWN = "unknown"


class DownloadError(ClipadorError):
    def __init__(
        self, kind: DownloadErrorKind, message: str, *, detail: str | None, retryable: bool
    ):
        super().__init__(message, detail=detail, retryable=retryable)
        self.kind = kind


_UPDATE_HINT = "Atualize o yt-dlp: `uv lock --upgrade-package yt-dlp` e depois `uv sync`."
_COOKIES_HINT = (
    "Configure cookies do navegador no .env (CLIPADOR_COOKIES_FROM_BROWSER=firefox ou "
    "CLIPADOR_COOKIES_FILE=caminho\\cookies.txt) e tente de novo."
)

# Order matters: the first matching rule wins.
_RULES: list[tuple[DownloadErrorKind, tuple[str, ...], str, bool]] = [
    (
        DownloadErrorKind.DISK,
        ("no space left on device", "errno 28", "não há espaço"),
        "Sem espaço em disco na pasta de dados. Libere espaço e tente de novo.",
        False,
    ),
    (
        DownloadErrorKind.LIVE,
        ("is currently live", "this live event will begin", "premieres in", "is not live yet"),
        "Esta URL é de uma live em andamento ou ainda não começou. O Clipador só baixa VODs e "
        "clips já publicados; aguarde a live terminar e use o VOD.",
        False,
    ),
    (
        DownloadErrorKind.ANTI_BOT,
        (
            "not a bot", "cloudflare", "captcha", "http error 403", "403: forbidden",
            "http error 429", "too many requests", "cf_clearance",
        ),
        "A plataforma bloqueou o download (proteção anti-bot ou limite de acessos). "
        + _COOKIES_HINT,
        True,
    ),
    (
        DownloadErrorKind.PRIVATE,
        (
            "private video", "this video is private", "subscriber", "sub-only", "members-only",
            "members only", "join this channel", "requires authentication", "login required",
            "you must be logged in", "sign in to confirm your age", "age-restricted",
            "only available to", "only available for registered users",
        ),
        "Conteúdo privado, restrito a inscritos/membros ou com restrição de idade. Só é possível "
        "baixar com cookies de uma conta que tenha acesso. " + _COOKIES_HINT,
        False,
    ),
    (
        DownloadErrorKind.NOT_FOUND,
        (
            "video unavailable", "does not exist", "http error 404", "404: not found",
            "has been removed", "no longer available", "has been deleted", "vod not found",
        ),
        "Conteúdo não encontrado: o VOD/clip foi removido, expirou ou a URL está errada.",
        False,
    ),
    (
        DownloadErrorKind.INVALID_URL,
        ("unsupported url", "is not a valid url", "no video formats found"),
        "URL inválida ou não suportada. Cole o link direto de um VOD ou clip.",
        False,
    ),
    (
        DownloadErrorKind.NETWORK,
        (
            "unable to download webpage", "getaddrinfo failed", "name or service not known",
            "temporary failure in name resolution", "timed out", "connection reset",
            "connection refused", "connection aborted", "network is unreachable",
            "errno 11001", "remote end closed", "incompleteread", "urlopen error", "ssl:",
            "unable to download video data", "giving up after",
        ),
        "Falha de rede durante o download. Verifique sua conexão; o Clipador vai tentar de novo.",
        True,
    ),
    (
        DownloadErrorKind.EXTRACTOR,
        (
            "unable to extract", "please report this issue", "confirm you are on the latest",
            "nsig extraction failed", "unable to parse", "keyerror", "extractor error",
        ),
        "O yt-dlp não conseguiu ler esta página (o site mudou e o extrator está desatualizado). "
        + _UPDATE_HINT,
        False,
    ),
]  # fmt: skip


def classify_ytdlp_error(output: str) -> DownloadError:
    """Map yt-dlp output to a DownloadError with a clear PT-BR message."""
    error_lines = [ln for ln in output.splitlines() if "ERROR" in ln]
    haystack = ("\n".join(error_lines) if error_lines else output).lower()
    detail = output.strip()[-4000:] or None
    for kind, needles, message, retryable in _RULES:
        if any(n in haystack for n in needles):
            return DownloadError(kind, message, detail=detail, retryable=retryable)
    return DownloadError(
        DownloadErrorKind.UNKNOWN,
        "O download falhou por um motivo não identificado. Veja o log do job. Se persistir, "
        + _UPDATE_HINT,
        detail=detail,
        retryable=True,
    )


# --------------------------------------------------------------------------------------------
# yt-dlp runner
# --------------------------------------------------------------------------------------------

VIDEO_BASENAME = "video"
THUMB_BASENAME = "thumb"


@dataclass
class DownloadResult:
    video_path: Path
    thumbnail_path: Path | None


class YtDlp:
    def __init__(self, settings: Settings):
        self.settings = settings

    # --- command building -----------------------------------------------------------------
    def base_cmd(self) -> list[str]:
        if self.settings.ytdlp_command.strip():
            return [self.settings.ytdlp_command.strip()]
        return [sys.executable, "-m", "yt_dlp"]

    def common_args(self) -> list[str]:
        args = ["--ignore-config", "--no-playlist", "--no-colors", "--encoding", "utf-8"]
        if self.settings.cookies_file:
            args += ["--cookies", str(self.settings.cookies_file)]
        elif self.settings.cookies_from_browser:
            args += ["--cookies-from-browser", self.settings.cookies_from_browser]
        if self.settings.ffmpeg_path not in ("", "ffmpeg"):
            args += ["--ffmpeg-location", self.settings.ffmpeg_path]
        return args

    def download_args(self, url: str, dest_dir: Path) -> list[str]:
        h = self.settings.max_download_height
        return [
            *self.base_cmd(),
            *self.common_args(),
            "--newline",
            "--continue",  # resume partial downloads after a crash
            "-f", "bv*+ba/b",
            "-S", f"res:{h},vcodec:h264,acodec:m4a",
            "--merge-output-format", "mp4",
            "--remux-video", "mp4",
            "--write-thumbnail", "--convert-thumbnails", "jpg",
            "-o", str(dest_dir / f"{VIDEO_BASENAME}.%(ext)s"),
            "-o", "thumbnail:" + str(dest_dir / f"{THUMB_BASENAME}.%(ext)s"),
            "--",
            url,
        ]  # fmt: skip

    @staticmethod
    def _env() -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        return env

    # --- operations -------------------------------------------------------------------------
    def version(self) -> str | None:
        try:
            res = subprocess.run(
                [*self.base_cmd(), "--version"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=60, env=self._env(),
            )  # fmt: skip
        except (OSError, subprocess.TimeoutExpired):
            return None
        return res.stdout.strip() if res.returncode == 0 else None

    def fetch_info(self, url: str) -> dict[str, Any]:
        cmd = [*self.base_cmd(), *self.common_args(), "-J", "--skip-download", "--", url]
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=180, env=self._env(),
            )  # fmt: skip
        except FileNotFoundError as e:
            raise DownloadError(
                DownloadErrorKind.UNKNOWN,
                "yt-dlp não encontrado. Rode `uv sync` ou ajuste CLIPADOR_YTDLP_COMMAND.",
                detail=str(e),
                retryable=False,
            ) from e
        except subprocess.TimeoutExpired as e:
            raise DownloadError(
                DownloadErrorKind.NETWORK,
                "Tempo esgotado ao buscar os dados do vídeo.",
                detail=str(e),
                retryable=True,
            ) from e
        if res.returncode != 0:
            raise classify_ytdlp_error(res.stderr + "\n" + res.stdout)
        try:
            return json.loads(res.stdout)
        except json.JSONDecodeError as e:
            raise DownloadError(
                DownloadErrorKind.UNKNOWN,
                "Resposta inesperada do yt-dlp ao buscar os dados do vídeo.",
                detail=res.stdout[-2000:],
                retryable=True,
            ) from e

    def download(
        self,
        url: str,
        dest_dir: Path,
        progress_cb: ProgressCallback,
        log_cb: Callable[[str], None] | None = None,
    ) -> DownloadResult:
        """Download best <= max height as mp4 into dest_dir/video.mp4 (+ thumb.jpg)."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.download_args(url, dest_dir)
        log.info("yt-dlp: %s", " ".join(cmd))
        tail: deque[str] = deque(maxlen=60)
        tracker = ProgressTracker()
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace", bufsize=1, env=self._env(),
            )  # fmt: skip
        except FileNotFoundError as e:
            raise DownloadError(
                DownloadErrorKind.UNKNOWN,
                "yt-dlp não encontrado. Rode `uv sync` ou ajuste CLIPADOR_YTDLP_COMMAND.",
                detail=str(e),
                retryable=False,
            ) from e
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                tail.append(line)
                update = tracker.feed(line)
                if update is not None:
                    progress_cb(*update)
                elif log_cb and not line.startswith("[download]"):
                    log_cb(line)
            proc.wait()
        except BaseException:
            _terminate(proc)
            raise
        output = "\n".join(tail)
        if proc.returncode != 0:
            if log_cb:
                for ln in list(tail)[-10:]:
                    log_cb(ln)
            raise classify_ytdlp_error(output)
        video = find_video_file(dest_dir)
        if video is None:
            raise DownloadError(
                DownloadErrorKind.UNKNOWN,
                "O yt-dlp terminou, mas o arquivo de vídeo não foi encontrado.",
                detail=output,
                retryable=True,
            )
        thumb = dest_dir / f"{THUMB_BASENAME}.jpg"
        return DownloadResult(video, thumb if thumb.is_file() else None)


def find_video_file(dest_dir: Path) -> Path | None:
    mp4 = dest_dir / f"{VIDEO_BASENAME}.mp4"
    if mp4.is_file():
        return mp4
    for p in sorted(dest_dir.glob(f"{VIDEO_BASENAME}.*")):
        if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov", ".ts", ".flv"):
            return p
    return None


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
