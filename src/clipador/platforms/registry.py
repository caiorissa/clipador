"""Find the platform adapter for a URL."""

from __future__ import annotations

from clipador.models import Platform
from clipador.platforms.base import InvalidUrlError, ParsedUrl, PlatformAdapter
from clipador.platforms.kick import KickAdapter
from clipador.platforms.twitch import TwitchAdapter
from clipador.platforms.youtube import YouTubeAdapter
from clipador.services.downloader import YtDlp

ADAPTERS: dict[Platform, type[PlatformAdapter]] = {
    Platform.TWITCH: TwitchAdapter,
    Platform.YOUTUBE: YouTubeAdapter,
    Platform.KICK: KickAdapter,
}


def detect_platform(url: str) -> Platform | None:
    for platform, cls in ADAPTERS.items():
        if cls.detect(url):
            return platform
    return None


def parse_url(url: str) -> ParsedUrl:
    platform = detect_platform(url)
    if platform is None:
        raise InvalidUrlError(
            "URL não reconhecida. O Clipador aceita links de VODs e clips da Twitch, "
            "do YouTube e do Kick."
        )
    return ADAPTERS[platform].parse(url)


def get_adapter(platform: Platform | str, ytdlp: YtDlp | None = None) -> PlatformAdapter:
    return ADAPTERS[Platform(platform)](ytdlp)
