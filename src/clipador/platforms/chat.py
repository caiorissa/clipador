"""Chat replay providers and parsers.

- Twitch: TwitchDownloaderCLI (`chatdownload`) -> JSON -> parse_twitch_chat_json.
- YouTube: yt-dlp `live_chat` subtitle (JSON lines) -> parse_youtube_live_chat.
- Kick: no reliable/official way to get VOD chat -> ChatUnavailable.

Parsers are pure functions over the tools' file formats so they can be tested with fixtures.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from clipador.config import Settings
from clipador.models import Source
from clipador.platforms.base import ChatLog, ChatMessage, ChatProvider, ChatUnavailable

# --- storage ----------------------------------------------------------------------------------


def save_chat(path: Path, log: ChatLog) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"provider": log.provider, "count": len(log.messages)}) + "\n")
        for m in log.messages:
            row = {"t": round(m.t, 3), "a": m.author, "x": m.text}
            if m.emotes:
                row["e"] = list(m.emotes)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_chat(path: Path) -> ChatLog:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = json.loads(f.readline())
        msgs = []
        for line in f:
            r = json.loads(line)
            msgs.append(ChatMessage(r["t"], r.get("a", ""), r.get("x", ""), tuple(r.get("e", ()))))
    return ChatLog(msgs, header.get("provider", "?"))


# --- Twitch (TwitchDownloaderCLI) -------------------------------------------------------------


def parse_twitch_chat_json(data: dict[str, Any]) -> list[ChatMessage]:
    """TwitchDownloaderCLI `chatdownload` JSON: {"comments": [{"content_offset_seconds": 12.3,
    "commenter": {"display_name": ...}, "message": {"body": ..., "fragments": [{"text": "LUL",
    "emoticon": {"emoticon_id": ...}}]}}]}. Third-party emotes (BTTV/FFZ/7TV) arrive as plain
    text; they are recognized later by name."""
    out = []
    for c in data.get("comments") or []:
        t = c.get("content_offset_seconds")
        if t is None:
            continue
        commenter = c.get("commenter") or {}
        msg = c.get("message") or {}
        body = msg.get("body")
        fragments = msg.get("fragments") or []
        if body is None:
            body = "".join(f.get("text", "") for f in fragments)
        emotes = tuple(
            f.get("text", "").strip() for f in fragments if f.get("emoticon") and f.get("text")
        )
        author = commenter.get("display_name") or commenter.get("name") or ""
        out.append(ChatMessage(float(t), author, body, emotes))
    out.sort(key=lambda m: m.t)
    return out


class TwitchChatProvider(ChatProvider):
    version = "twitchdownloader-1"

    def __init__(self, settings: Settings):
        self.settings = settings

    def _exe(self) -> str:
        configured = self.settings.twitch_downloader_path
        exe = shutil.which(configured) or (configured if Path(configured).is_file() else None)
        if not exe:
            raise ChatUnavailable(
                "TwitchDownloaderCLI não encontrado, então o chat da Twitch não foi baixado. "
                "Baixe em https://github.com/lay295/TwitchDownloader/releases (arquivo "
                "TwitchDownloaderCLI-…-Windows-x64.zip), extraia e defina "
                "CLIPADOR_TWITCH_DOWNLOADER_PATH no .env com o caminho do .exe."
            )
        return exe

    def fetch(self, source: Source, work_dir: Path, progress_cb: Callable[[float], None]):
        exe = self._exe()
        work_dir.mkdir(parents=True, exist_ok=True)
        out = work_dir / "twitch_chat.json"
        with contextlib.suppress(OSError):
            out.unlink()
        cmd = [exe, "chatdownload", "--id", source.external_id, "-o", str(out)]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace", cwd=str(work_dir),
            )  # fmt: skip
        except OSError as e:
            raise ChatUnavailable("Não foi possível executar o TwitchDownloaderCLI.", str(e)) from e
        tail: list[str] = []
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                tail = [*tail[-29:], line.rstrip()]
                m = re.search(r"(\d{1,3})\s*%", line)
                if m:
                    progress_cb(min(1.0, int(m.group(1)) / 100))
            proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        if proc.returncode != 0 or not out.is_file():
            raise ChatUnavailable(
                "O TwitchDownloaderCLI não conseguiu baixar o chat deste VOD (VOD expirado, "
                "sub-only ou ferramenta desatualizada).",
                "\n".join(tail),
            )
        data = json.loads(out.read_text("utf-8"))
        return ChatLog(parse_twitch_chat_json(data), "twitch")


# --- YouTube (yt-dlp live_chat) -----------------------------------------------------------------


def _runs_text(runs: Iterable[dict]) -> tuple[str, tuple[str, ...]]:
    parts, emotes = [], []
    for r in runs:
        if "text" in r:
            parts.append(r["text"])
        elif "emoji" in r:
            e = r["emoji"]
            shortcuts = e.get("shortcuts") or []
            name = shortcuts[0] if shortcuts else e.get("emojiId", "")
            emotes.append(name)
            parts.append(name)
    return "".join(parts), tuple(emotes)


def parse_youtube_live_chat(lines: Iterable[str]) -> list[ChatMessage]:
    """yt-dlp `--sub-langs live_chat` output: one JSON object per line, each with
    replayChatItemAction.{videoOffsetTimeMsec, actions[].addChatItemAction.item.<renderer>}."""
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        replay = obj.get("replayChatItemAction") or {}
        offset = replay.get("videoOffsetTimeMsec") or obj.get("videoOffsetTimeMsec")
        if offset is None:
            continue
        t = int(offset) / 1000.0
        for action in replay.get("actions") or []:
            item = (action.get("addChatItemAction") or {}).get("item") or {}
            r = item.get("liveChatTextMessageRenderer") or item.get("liveChatPaidMessageRenderer")
            if not r:
                continue
            text, emotes = _runs_text((r.get("message") or {}).get("runs") or [])
            author = (r.get("authorName") or {}).get("simpleText", "")
            out.append(ChatMessage(t, author, text, emotes))
    out.sort(key=lambda m: m.t)
    return out


class YouTubeChatProvider(ChatProvider):
    version = "ytdlp-live_chat-1"

    def __init__(self, settings: Settings):
        self.settings = settings

    def fetch(self, source: Source, work_dir: Path, progress_cb: Callable[[float], None]):
        from clipador.services.downloader import YtDlp, classify_ytdlp_error

        yt = YtDlp(self.settings)
        work_dir.mkdir(parents=True, exist_ok=True)
        stem = work_dir / "yt_chat"
        for old in work_dir.glob("yt_chat*"):
            old.unlink()
        cmd = [
            *yt.base_cmd(), *yt.common_args(), "--skip-download", "--write-subs",
            "--sub-langs", "live_chat", "-o", str(stem) + ".%(ext)s", "--", source.url,
        ]  # fmt: skip
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=6 * 3600, env=yt._env(),
            )  # fmt: skip
        except (OSError, subprocess.TimeoutExpired) as e:
            raise ChatUnavailable("Falha ao baixar o chat do YouTube.", str(e)) from e
        files = sorted(work_dir.glob("yt_chat*.live_chat.json"))
        if res.returncode != 0 and not files:
            err = classify_ytdlp_error(res.stderr + res.stdout)
            raise ChatUnavailable(f"Chat do YouTube não baixado: {err.message}", err.detail)
        if not files:
            return None  # regular video, no live chat replay
        progress_cb(1.0)
        with open(files[0], encoding="utf-8") as f:
            return ChatLog(parse_youtube_live_chat(f), "youtube")


# --- Kick -----------------------------------------------------------------------------------


class KickChatProvider(ChatProvider):
    version = "kick-none-1"

    def __init__(self, settings: Settings):
        self.settings = settings

    def fetch(self, source: Source, work_dir: Path, progress_cb: Callable[[float], None]):
        raise ChatUnavailable(
            "Chat do Kick indisponível: o Kick não oferece uma forma oficial e estável de baixar "
            "o chat de VODs. A nota usa só áudio e fala."
        )
