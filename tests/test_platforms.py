import pytest

from clipador.models import Platform, SourceKind
from clipador.platforms.base import InvalidUrlError, PlatformAdapter, normalize_handle
from clipador.platforms.registry import detect_platform, get_adapter, parse_url

T, Y, K = Platform.TWITCH, Platform.YOUTUBE, Platform.KICK
VOD, CLIP = SourceKind.VOD, SourceKind.CLIP


@pytest.mark.parametrize(
    ("url", "platform", "kind", "external_id", "channel"),
    [
        # Twitch
        ("https://www.twitch.tv/videos/2283746281", T, VOD, "2283746281", None),
        ("twitch.tv/videos/2283746281?t=1h2m3s", T, VOD, "2283746281", None),
        ("https://m.twitch.tv/videos/2283746281", T, VOD, "2283746281", None),
        ("https://www.twitch.tv/Fulano/v/2283746281", T, VOD, "2283746281", "fulano"),
        ("https://player.twitch.tv/?video=v2283746281&parent=x", T, VOD, "2283746281", None),
        ("https://clips.twitch.tv/FunnyClipSlug-AbC123_x", T, CLIP, "FunnyClipSlug-AbC123_x", None),
        ("https://clips.twitch.tv/embed?clip=FunnySlug&parent=x", T, CLIP, "FunnySlug", None),
        ("https://www.twitch.tv/fulano/clip/FunnySlug-xyz", T, CLIP, "FunnySlug-xyz", "fulano"),
        ("https://m.twitch.tv/fulano/clip/FunnySlug-xyz", T, CLIP, "FunnySlug-xyz", "fulano"),
        # YouTube
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123&t=42s", Y, VOD, "dQw4w9WgXcQ", None),
        ("https://youtu.be/dQw4w9WgXcQ?t=42", Y, VOD, "dQw4w9WgXcQ", None),
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", Y, VOD, "dQw4w9WgXcQ", None),
        ("https://www.youtube.com/live/dQw4w9WgXcQ?si=abc", Y, VOD, "dQw4w9WgXcQ", None),
        ("https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ", Y, VOD, "dQw4w9WgXcQ", None),
        ("https://youtube.com/shorts/dQw4w9WgXcQ", Y, CLIP, "dQw4w9WgXcQ", None),
        (
            "https://www.youtube.com/clip/UgkxU2HSeGL_NvmDJ-nQJrlLwllwMDBdGZFs",
            Y, CLIP, "UgkxU2HSeGL_NvmDJ-nQJrlLwllwMDBdGZFs", None,
        ),
        # Kick
        (
            "https://kick.com/fulano/videos/0b4e8a3c-1d2e-4f5a-9b6c-7d8e9f0a1b2c",
            K, VOD, "0b4e8a3c-1d2e-4f5a-9b6c-7d8e9f0a1b2c", "fulano",
        ),
        (
            "kick.com/video/0B4E8A3C-1D2E-4F5A-9B6C-7D8E9F0A1B2C",
            K, VOD, "0b4e8a3c-1d2e-4f5a-9b6c-7d8e9f0a1b2c", None,
        ),
        ("https://kick.com/Fulano/clips/clip_01HQXYZABC123", K, CLIP, "clip_01HQXYZABC123", "fulano"),
        ("https://kick.com/fulano?clip=clip_01HQXYZABC123", K, CLIP, "clip_01HQXYZABC123", "fulano"),
    ],
)  # fmt: skip
def test_parse_valid_urls(url, platform, kind, external_id, channel):
    assert detect_platform(url) == platform
    parsed = parse_url(url)
    assert (parsed.platform, parsed.kind, parsed.external_id) == (platform, kind, external_id)
    assert parsed.channel_handle == channel
    assert parsed.url.startswith("https://")
    # canonical URL parses to the same thing
    assert parse_url(parsed.url).external_id == external_id


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not a url",
        "https://example.com/videos/123",
        "https://vimeo.com/123456",
        "https://www.twitch.tv/fulano",  # live/channel
        "https://www.twitch.tv/fulano/videos",
        "https://www.twitch.tv/videos/abc",
        "https://www.youtube.com/@fulano",
        "https://www.youtube.com/channel/UC1234567890123456789012",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/playlist?list=PL123",
        "https://youtu.be/",
        "https://kick.com/fulano",  # live/channel
        "https://kick.com/fulano/videos/not-a-uuid",
        "https://kick.com/fulano/clips/notaclip",
    ],
)
def test_parse_invalid_urls(url):
    with pytest.raises(InvalidUrlError) as exc:
        parse_url(url)
    assert exc.value.message  # PT-BR message for the user


def test_live_channel_message_mentions_vod():
    with pytest.raises(InvalidUrlError) as exc:
        parse_url("https://www.twitch.tv/fulano")
    assert "VOD" in exc.value.message


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Fulano", "fulano"),
        ("@Fulano", "fulano"),
        ("https://www.twitch.tv/Fulano", "fulano"),
        ("https://kick.com/Fulano", "fulano"),
        ("https://www.youtube.com/@Fulano", "fulano"),
        ("https://www.youtube.com/channel/UCabcdefghijklmnopqrstuv", "UCabcdefghijklmnopqrstuv"),
        ("UCabcdefghijklmnopqrstuv", "UCabcdefghijklmnopqrstuv"),
    ],
)
def test_normalize_handle(value, expected):
    assert normalize_handle(value) == expected


def test_metadata_from_info_and_channel_match():
    adapter = get_adapter("youtube")
    info = {
        "id": "dQw4w9WgXcQ",
        "title": "GTA 6 — dia 1",
        "duration": 7200.5,
        "thumbnail": "https://i.ytimg.com/x.jpg",
        "channel": "Fulano Gamer",
        "channel_id": "UCabcdefghijklmnopqrstuv",
        "uploader_id": "@FulanoGamer",
        "timestamp": 1762905600,
    }
    meta = adapter.metadata_from_info(info)
    assert meta.title == "GTA 6 — dia 1"
    assert meta.duration_seconds == 7200.5
    assert meta.published_at.year == 2025
    assert PlatformAdapter.channel_matches(
        meta.channel_candidates, ["https://youtube.com/@fulanogamer"]
    )
    assert PlatformAdapter.channel_matches(meta.channel_candidates, ["UCabcdefghijklmnopqrstuv"])
    assert not PlatformAdapter.channel_matches(meta.channel_candidates, ["outro"])


def test_every_platform_has_a_chat_provider():
    from clipador.platforms.chat import KickChatProvider, TwitchChatProvider, YouTubeChatProvider

    expected = {T: TwitchChatProvider, Y: YouTubeChatProvider, K: KickChatProvider}
    for p, cls in expected.items():
        assert isinstance(get_adapter(p).chat_provider(), cls)


def test_fetch_metadata_rejects_live(monkeypatch):
    from clipador.services.downloader import DownloadError, DownloadErrorKind

    adapter = get_adapter("twitch")
    monkeypatch.setattr(adapter.ytdlp, "fetch_info", lambda url: {"id": "1", "is_live": True})
    with pytest.raises(DownloadError) as exc:
        adapter.fetch_metadata("https://www.twitch.tv/videos/12345")
    assert exc.value.kind == DownloadErrorKind.LIVE
