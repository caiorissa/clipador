"""Twitch: VODs (twitch.tv/videos/123) and clips (clips.twitch.tv/Slug, twitch.tv/x/clip/Slug)."""

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
from clipador.platforms.chat import TwitchChatProvider

_VOD_ID = re.compile(r"^v?(\d{5,})$")
_CLIP_SLUG = re.compile(r"^[A-Za-z0-9_-]{4,}$")
_CHANNEL = re.compile(r"^[A-Za-z0-9_]{2,25}$")


def _vod(vod_id: str, channel: str | None = None) -> ParsedUrl:
    return ParsedUrl(
        Platform.TWITCH,
        SourceKind.VOD,
        vod_id,
        f"https://www.twitch.tv/videos/{vod_id}",
        channel.lower() if channel else None,
    )


def _clip(slug: str, channel: str | None = None) -> ParsedUrl:
    return ParsedUrl(
        Platform.TWITCH,
        SourceKind.CLIP,
        slug,
        f"https://clips.twitch.tv/{slug}",
        channel.lower() if channel else None,
    )


class TwitchAdapter(PlatformAdapter):
    platform = Platform.TWITCH
    hosts = ("twitch.tv", "clips.twitch.tv", "player.twitch.tv")
    chat_provider_cls = TwitchChatProvider

    @classmethod
    def parse(cls, url: str) -> ParsedUrl:
        parts, host = split_url(url)
        if host not in cls.hosts:
            raise InvalidUrlError("Esta URL não é da Twitch.")
        segs = path_segments(parts)
        query = parse_qs(parts.query)

        if host == "clips.twitch.tv":
            slug = query.get("clip", [None])[0] if segs[:1] == ["embed"] else (segs or [None])[0]
            if slug and _CLIP_SLUG.match(slug):
                return _clip(slug)
            raise InvalidUrlError("Link de clip da Twitch inválido.")

        if host == "player.twitch.tv":
            m = _VOD_ID.match(query.get("video", [""])[0])
            if m:
                return _vod(m.group(1))
            raise InvalidUrlError("Link do player da Twitch sem ?video=. Use o link do VOD.")

        # twitch.tv
        if len(segs) >= 2 and segs[0] == "videos":
            m = _VOD_ID.match(segs[1])
            if m:
                return _vod(m.group(1))
        if len(segs) >= 3 and segs[1] in ("v", "video", "videos"):
            m = _VOD_ID.match(segs[2])
            if m:
                return _vod(m.group(1), segs[0])
        if len(segs) >= 3 and segs[1] == "clip" and _CLIP_SLUG.match(segs[2]):
            return _clip(segs[2], segs[0])
        if len(segs) >= 2 and segs[0] == "clip" and _CLIP_SLUG.match(segs[1]):
            return _clip(segs[1])
        is_channel_page = len(segs) == 1 and _CHANNEL.match(segs[0])
        if is_channel_page or (len(segs) == 2 and segs[1] == "videos"):
            raise InvalidUrlError(
                "Este é o link de um canal ou live da Twitch. Cole o link de um VOD "
                "(twitch.tv/videos/123…) ou de um clip. Lives em andamento não são suportadas."
            )
        raise InvalidUrlError("Link da Twitch não reconhecido como VOD ou clip.")
