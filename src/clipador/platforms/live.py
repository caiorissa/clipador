"""Live URLs (channel pages / live videos) and the yt-dlp calls used by the live capture:
is the channel live now, and which stream URL should ffmpeg record (step 5E)."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from urllib.parse import parse_qs

from clipador.models import Platform
from clipador.platforms.base import InvalidUrlError, path_segments, split_url
from clipador.services.downloader import (
    DownloadError,
    DownloadErrorKind,
    YtDlp,
    classify_ytdlp_error,
)

_TWITCH_CHANNEL = re.compile(r"^[A-Za-z0-9_]{2,25}$")
_KICK_CHANNEL = re.compile(r"^[A-Za-z0-9_-]{2,}$")
_YT_VIDEO = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_CHANNEL_ID = re.compile(r"^UC[\w-]{22}$")
_TWITCH_RESERVED = {"videos", "directory", "settings", "downloads", "jobs", "p", "search"}


@dataclass(frozen=True)
class LiveTarget:
    platform: Platform
    url: str  # canonical URL handed to yt-dlp
    channel_handle: str | None  # normalized, when the URL names the channel
    external_id: str  # channel handle or video id (the session adds a timestamp)

    @property
    def supports_from_start(self) -> bool:
        """yt-dlp's --live-from-start only works on YouTube (DVR window of the live)."""
        return self.platform == Platform.YOUTUBE


def parse_live_url(url: str) -> LiveTarget:
    """Channel page (Twitch/Kick), or a YouTube channel '/live' page or live video.
    Raises InvalidUrlError (PT-BR) for VOD/clip links and anything else."""
    parts, host = split_url(url)
    segs = path_segments(parts)
    if host in ("twitch.tv", "m.twitch.tv"):
        if len(segs) == 1 and _TWITCH_CHANNEL.match(segs[0]) and segs[0] not in _TWITCH_RESERVED:
            ch = segs[0].lower()
            return LiveTarget(Platform.TWITCH, f"https://www.twitch.tv/{ch}", ch, ch)
        raise InvalidUrlError(
            "Para gravar uma live da Twitch, cole o link do canal (twitch.tv/fulano). Links de "
            "VOD ou clip vão em Fontes."
        )
    if host == "kick.com":
        if len(segs) == 1 and _KICK_CHANNEL.match(segs[0]):
            ch = segs[0].lower()
            return LiveTarget(Platform.KICK, f"https://kick.com/{ch}", ch, ch)
        raise InvalidUrlError(
            "Para gravar uma live do Kick, cole o link do canal (kick.com/fulano). Links de VOD "
            "ou clip vão em Fontes."
        )
    if host in ("youtube.com", "youtu.be"):
        vid = None
        if host == "youtu.be" and segs and _YT_VIDEO.match(segs[0]):
            vid = segs[0]
        elif segs[:1] == ["watch"]:
            v = parse_qs(parts.query).get("v", [""])[0]
            vid = v if _YT_VIDEO.match(v) else None
        elif len(segs) >= 2 and segs[0] == "live" and _YT_VIDEO.match(segs[1]):
            vid = segs[1]
        if vid:
            return LiveTarget(Platform.YOUTUBE, f"https://www.youtube.com/watch?v={vid}", None, vid)
        if segs and segs[0].startswith("@") and len(segs[0]) > 1 and segs[1:] in ([], ["live"]):
            handle = segs[0][1:].lower()
            return LiveTarget(
                Platform.YOUTUBE, f"https://www.youtube.com/@{handle}/live", handle, handle
            )
        if len(segs) >= 2 and segs[0] == "channel" and _YT_CHANNEL_ID.match(segs[1]):
            cid = segs[1]
            return LiveTarget(
                Platform.YOUTUBE, f"https://www.youtube.com/channel/{cid}/live", cid, cid
            )
        raise InvalidUrlError(
            "Para gravar uma live do YouTube, cole o link da live (youtube.com/watch?v=…) ou do "
            "canal (youtube.com/@fulano/live)."
        )
    raise InvalidUrlError("URL não reconhecida. Lives: Twitch, Kick ou YouTube (link do canal).")


def channel_live_url(platform: str, handle: str) -> str | None:
    """The page yt-dlp checks for a registered channel being live now (auto_watch)."""
    if platform == Platform.TWITCH:
        return f"https://www.twitch.tv/{handle}"
    if platform == Platform.KICK:
        return f"https://kick.com/{handle}"
    if platform == Platform.YOUTUBE:
        if _YT_CHANNEL_ID.match(handle):
            return f"https://www.youtube.com/channel/{handle}/live"
        return f"https://www.youtube.com/@{handle}/live"
    return None


# --- yt-dlp -------------------------------------------------------------------------------

_OFFLINE = (
    "is offline", "not currently live", "is not live", "not live", "no live stream",
    "this live event will begin", "is not currently live", "the channel is not currently live",
    "does not exist", "http error 404", "video unavailable",
)  # fmt: skip


@dataclass
class LiveStatusInfo:
    live: bool
    title: str | None = None
    video_id: str | None = None
    reason: str | None = None  # why not live (PT-BR), when known


def parse_status_output(returncode: int, stdout: str, stderr: str) -> LiveStatusInfo:
    """Output of `yt-dlp --simulate --print "%(is_live)s|%(live_status)s|%(id)s|%(title)s"`."""
    if returncode != 0:
        text = (stderr + "\n" + stdout).lower()
        if any(k in text for k in _OFFLINE):
            return LiveStatusInfo(False, reason="O canal não está ao vivo agora.")
        err = classify_ytdlp_error(stderr + "\n" + stdout)
        raise err
    line = next((ln for ln in stdout.splitlines() if "|" in ln), "")
    is_live, live_status, vid, title = [*line.split("|", 3), "", "", "", ""][:4]
    live = is_live.strip() == "True" or live_status.strip() == "is_live"
    return LiveStatusInfo(
        live, title.strip() or None, vid.strip() or None,
        None if live else "O canal não está ao vivo agora.",
    )  # fmt: skip


def check_live(ytdlp: YtDlp, url: str, timeout: float = 90) -> LiveStatusInfo:
    """Light check (no download): is this channel/video live right now?"""
    cmd = [*ytdlp.base_cmd(), *ytdlp.common_args(), "--simulate", "--print",
           "%(is_live)s|%(live_status)s|%(id)s|%(title)s", "--", url]  # fmt: skip
    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, env=ytdlp._env(),
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired) as e:
        raise DownloadError(
            DownloadErrorKind.NETWORK, "Não foi possível consultar a live (yt-dlp).",
            detail=str(e), retryable=True,
        ) from e  # fmt: skip
    return parse_status_output(res.returncode, res.stdout, res.stderr)


def stream_url_args(ytdlp: YtDlp, url: str, max_height: int) -> list[str]:
    fmt = f"best[height<={max_height}]/best"
    return [*ytdlp.base_cmd(), *ytdlp.common_args(), "-g", "-f", fmt, "--", url]


def resolve_stream_url(ytdlp: YtDlp, url: str, max_height: int, timeout: float = 90) -> str:
    """Direct media URL (usually an HLS .m3u8) for ffmpeg to record. Raises DownloadError
    (kind LIVE when the channel is offline)."""
    try:
        res = subprocess.run(
            stream_url_args(ytdlp, url, max_height), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, env=ytdlp._env(),
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired) as e:
        raise DownloadError(
            DownloadErrorKind.NETWORK, "Não foi possível obter o endereço do stream (yt-dlp).",
            detail=str(e), retryable=True,
        ) from e  # fmt: skip
    if res.returncode != 0:
        text = (res.stderr + res.stdout).lower()
        if any(k in text for k in _OFFLINE):
            raise DownloadError(DownloadErrorKind.LIVE, "O canal não está ao vivo agora.",
                                detail=res.stderr[-2000:], retryable=False)  # fmt: skip
        raise classify_ytdlp_error(res.stderr + "\n" + res.stdout)
    urls = [ln.strip() for ln in res.stdout.splitlines() if ln.strip().startswith("http")]
    if not urls:
        raise DownloadError(
            DownloadErrorKind.UNKNOWN, "O yt-dlp não devolveu o endereço do stream.",
            detail=res.stdout[-2000:], retryable=True,
        )  # fmt: skip
    return urls[0]


def pipe_download_args(ytdlp: YtDlp, url: str, max_height: int, from_start: bool) -> list[str]:
    """yt-dlp writing the live to stdout (used with --live-from-start, which can't be done
    with a plain URL): ffmpeg reads it from its stdin and segments it."""
    args = [*ytdlp.base_cmd(), *ytdlp.common_args(), "--no-part", "--quiet", "--no-warnings",
            "-f", f"best[height<={max_height}]/best", "-o", "-"]  # fmt: skip
    if from_start:
        args.append("--live-from-start")
    return [*args, "--", url]
