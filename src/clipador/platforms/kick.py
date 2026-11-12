"""Kick: VODs (kick.com/<canal>/videos/<uuid>, kick.com/video/<uuid>) and clips
(kick.com/<canal>/clips/clip_…, kick.com/<canal>?clip=clip_…).

Kick has aggressive anti-bot protection; downloads may need cookies (see README). Its chat API is
unofficial, so the day-2 chat provider will be best effort.
"""

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
from clipador.platforms.chat import KickChatProvider

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_CLIP_ID = re.compile(r"^clip_[A-Za-z0-9_-]+$")
_CHANNEL = re.compile(r"^[A-Za-z0-9_-]{2,}$")


class KickAdapter(PlatformAdapter):
    platform = Platform.KICK
    hosts = ("kick.com",)
    chat_provider_cls = KickChatProvider

    @classmethod
    def parse(cls, url: str) -> ParsedUrl:
        parts, host = split_url(url)
        if host not in cls.hosts:
            raise InvalidUrlError("Esta URL não é do Kick.")
        segs = path_segments(parts)
        clip_q = parse_qs(parts.query).get("clip", [""])[0]

        if len(segs) >= 2 and segs[0] == "video" and _UUID.match(segs[1]):
            vid = segs[1].lower()
            return ParsedUrl(Platform.KICK, SourceKind.VOD, vid, f"https://kick.com/video/{vid}")
        if len(segs) >= 3 and segs[1] == "videos" and _UUID.match(segs[2]):
            channel, vid = segs[0].lower(), segs[2].lower()
            return ParsedUrl(
                Platform.KICK,
                SourceKind.VOD,
                vid,
                f"https://kick.com/{channel}/videos/{vid}",
                channel,
            )
        if len(segs) >= 3 and segs[1] == "clips" and _CLIP_ID.match(segs[2]):
            channel = segs[0].lower()
            return ParsedUrl(
                Platform.KICK,
                SourceKind.CLIP,
                segs[2],
                f"https://kick.com/{channel}/clips/{segs[2]}",
                channel,
            )
        if len(segs) == 1 and _CLIP_ID.match(clip_q) and _CHANNEL.match(segs[0]):
            channel = segs[0].lower()
            return ParsedUrl(
                Platform.KICK,
                SourceKind.CLIP,
                clip_q,
                f"https://kick.com/{channel}/clips/{clip_q}",
                channel,
            )
        if len(segs) == 1 and _CHANNEL.match(segs[0]):
            raise InvalidUrlError(
                "Este é o link de um canal ou live do Kick. Cole o link de um VOD "
                "(kick.com/canal/videos/…) ou de um clip. Lives em andamento não são suportadas."
            )
        raise InvalidUrlError("Link do Kick não reconhecido como VOD ou clip.")
