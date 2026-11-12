import json
from pathlib import Path

import pytest

from clipador.models import Source
from clipador.platforms.base import ChatLog, ChatMessage, ChatUnavailable
from clipador.platforms.chat import (
    KickChatProvider,
    TwitchChatProvider,
    load_chat,
    parse_twitch_chat_json,
    parse_youtube_live_chat,
    save_chat,
)

FIX = Path(__file__).parent / "fixtures"


def test_parse_twitch_downloader_json():
    msgs = parse_twitch_chat_json(json.loads((FIX / "twitch_chat.json").read_text("utf-8")))
    assert [m.t for m in msgs] == [5.2, 100.75, 101.0]
    assert msgs[1].author == "Viewer2" and msgs[1].text == "LUL LUL"
    assert msgs[1].emotes == ("LUL", "LUL")
    assert msgs[2].emotes == ()  # third-party emote names come as plain text


def test_parse_youtube_live_chat_lines():
    with open(FIX / "youtube_live_chat.json", encoding="utf-8") as f:
        msgs = parse_youtube_live_chat(f)
    assert [round(m.t, 2) for m in msgs] == [5.3, 100.25, 100.9]
    assert msgs[0].author == "Fulana"
    assert msgs[0].emotes == (":face_with_tears_of_joy:",)
    assert msgs[1].text == "CLIPA ISSO"  # super chat counts too


def test_chat_roundtrip(tmp_path):
    log = ChatLog([ChatMessage(1.5, "a", "oi"), ChatMessage(2.0, "b", "LUL", ("LUL",))], "twitch")
    save_chat(tmp_path / "c.jsonl.gz", log)
    back = load_chat(tmp_path / "c.jsonl.gz")
    assert back.provider == "twitch" and back.messages == log.messages


def test_twitch_provider_without_tool_explains_where_to_get_it(settings, tmp_path):
    settings.twitch_downloader_path = str(tmp_path / "nope.exe")
    src = Source(streamer_id=1, platform="twitch", kind="vod", external_id="1", url="u")
    with pytest.raises(ChatUnavailable) as exc:
        TwitchChatProvider(settings).fetch(src, tmp_path, lambda f: None)
    assert "github.com/lay295/TwitchDownloader" in exc.value.reason


def test_twitch_provider_with_fake_cli(settings, tmp_path):
    """A fake TwitchDownloaderCLI (python script wrapped in .cmd/.sh) writes the fixture."""
    import sys

    script = tmp_path / "fake_tdcli.py"
    script.write_text(
        "import sys, shutil\n"
        "out = sys.argv[sys.argv.index('-o') + 1]\n"
        "print('[STATUS] - Downloading 50%', flush=True)\n"
        f"shutil.copy(r'{FIX / 'twitch_chat.json'}', out)\n",
        encoding="utf-8",
    )
    if sys.platform == "win32":
        exe = tmp_path / "tdcli.cmd"
        exe.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        exe = tmp_path / "tdcli.sh"
        exe.write_text(f'#!/bin/sh\n"{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        exe.chmod(0o755)
    settings.twitch_downloader_path = str(exe)
    src = Source(streamer_id=1, platform="twitch", kind="vod", external_id="2283746281", url="u")
    progress = []
    log = TwitchChatProvider(settings).fetch(src, tmp_path / "work", progress.append)
    assert len(log.messages) == 3 and log.provider == "twitch"
    assert progress == [0.5]


def test_kick_chat_is_unavailable(settings, tmp_path):
    src = Source(streamer_id=1, platform="kick", kind="vod", external_id="x", url="u")
    with pytest.raises(ChatUnavailable, match="Kick"):
        KickChatProvider(settings).fetch(src, tmp_path, lambda f: None)
