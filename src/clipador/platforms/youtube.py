"""YouTube: videos/live replays (watch, youtu.be, /live/, /embed/) and clips (/clip/, /shorts/)."""

from __future__ import annotations

import re
from urllib.parse import parse_qs

from clipador.models import Platform, SourceKind
from clipador.platforms.base import (
    InvalidUrlError,
    ParsedUrl,
    PlatformAdapter,
    path_segments,
    split_url,
)

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_CLIP_ID = re.compile(r"^[A-Za-z0-9_-]{20,}$")


def _video(video_id: str) -> ParsedUrl:
    return ParsedUrl(
        Platform.YOUTUBE,
        SourceKind.VOD,
        video_id,
        f"https://www.youtube.com/watch?v={video_id}",
    )


class YouTubeAdapter(PlatformAdapter):
    platform = Platform.YOUTUBE
    hosts = ("youtube.com", "youtu.be", "youtube-nocookie.com")

    @classmethod
    def parse(cls, url: str) -> ParsedUrl:
        parts, host = split_url(url)
        if host not in cls.hosts:
            raise InvalidUrlError("Esta URL não é do YouTube.")
        segs = path_segments(parts)

        if host == "youtu.be":
            if segs and _VIDEO_ID.match(segs[0]):
                return _video(segs[0])
            raise InvalidUrlError("Link curto do YouTube inválido.")

        if segs[:1] == ["watch"] or not segs:
            vid = parse_qs(parts.query).get("v", [""])[0]
            if _VIDEO_ID.match(vid):
                return _video(vid)
            raise InvalidUrlError("Link do YouTube sem o ID do vídeo (?v=…).")
        if len(segs) >= 2 and segs[0] in ("live", "embed", "v") and _VIDEO_ID.match(segs[1]):
            return _video(segs[1])
        if len(segs) >= 2 and segs[0] == "shorts" and _VIDEO_ID.match(segs[1]):
            return ParsedUrl(
                Platform.YOUTUBE,
                SourceKind.CLIP,
                segs[1],
                f"https://www.youtube.com/shorts/{segs[1]}",
            )
        if len(segs) >= 2 and segs[0] == "clip" and _CLIP_ID.match(segs[1]):
            return ParsedUrl(
                Platform.YOUTUBE,
                SourceKind.CLIP,
                segs[1],
                f"https://www.youtube.com/clip/{segs[1]}",
            )
        if segs[0].startswith("@") or segs[0] in ("channel", "c", "user", "playlist"):
            raise InvalidUrlError(
                "Este é o link de um canal ou playlist do YouTube. Cole o link de um vídeo "
                "(youtube.com/watch?v=…, youtu.be/…) ou de um clip (youtube.com/clip/…)."
            )
        raise InvalidUrlError("Link do YouTube não reconhecido como vídeo ou clip.")
