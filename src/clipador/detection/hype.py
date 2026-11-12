"""Hype words/expressions (editable TOML) and matching against transcript words and chat."""

from __future__ import annotations

import re
import shutil
import tomllib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_FILE = Path(__file__).with_name("default_hype_words.toml")
TEXT_SECTIONS = ("pt", "en", "gta")
LAUGH_RE = re.compile(r"^(k{3,}|(ha){2,}h?|(he){2,}|(rs){2,}|(ja){2,}|a?(hu){2,})$")
_TOKEN_RE = re.compile(r"[^\w]+")


def normalize(text: str) -> str:
    """casefold + strip accents + collapse punctuation to spaces."""
    t = unicodedata.normalize("NFKD", text.casefold())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(_TOKEN_RE.sub(" ", t).split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


@dataclass
class HypeLexicon:
    phrases: dict[tuple[str, ...], float] = field(default_factory=dict)  # spoken
    chat: dict[str, float] = field(default_factory=dict)  # whole message / emote
    laugh_weight: float = 1.0

    @property
    def max_len(self) -> int:
        return max((len(p) for p in self.phrases), default=1)

    @classmethod
    def from_toml(cls, text: str) -> HypeLexicon:
        data = tomllib.loads(text)
        lex = cls()
        ignore = {normalize(w) for w in data.get("ignorar", {}).get("lista", [])}
        for section in TEXT_SECTIONS:
            for phrase, weight in (data.get(section) or {}).items():
                key = tuple(tokens(phrase))
                if key and " ".join(key) not in ignore:
                    lex.phrases[key] = max(float(weight), lex.phrases.get(key, 0.0))
        for item, weight in (data.get("chat") or {}).items():
            # emoji / :shortcut: keep their raw form; words are normalized
            key = normalize(item) or item.strip().casefold()
            lex.chat[key] = float(weight)
        lex.laugh_weight = float((data.get("risada") or {}).get("peso", 1.0))
        return lex


def ensure_hype_file(path: Path) -> Path:
    """Copy the default list to data/ on first use so the user can edit it."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DEFAULT_FILE, path)
    return path


def load_lexicon(path: Path) -> HypeLexicon:
    return HypeLexicon.from_toml(ensure_hype_file(path).read_text("utf-8"))


def match_words(words: list[tuple[float, str]], lex: HypeLexicon) -> list[tuple[float, float, str]]:
    """words: (start_time, text) in order. Returns hits (time, weight, phrase), longest phrase
    first and without double counting a word inside a longer matched expression."""
    toks: list[tuple[float, str]] = []
    for t, w in words:
        for tok in tokens(w):
            toks.append((t, tok))
    hits = []
    i, n, maxlen = 0, len(toks), lex.max_len
    while i < n:
        matched = False
        for length in range(min(maxlen, n - i), 0, -1):
            key = tuple(tok for _, tok in toks[i : i + length])
            weight = lex.phrases.get(key)
            if weight is not None:
                hits.append((toks[i][0], weight, " ".join(key)))
                i += length
                matched = True
                break
        if not matched:
            if LAUGH_RE.match(toks[i][1]):
                hits.append((toks[i][0], lex.laugh_weight, "risada"))
            i += 1
    return hits


def chat_message_weight(text: str, emotes: tuple[str, ...], lex: HypeLexicon) -> float:
    """Hype value of one chat message: known emotes/words, laughter, or the whole message."""
    total = 0.0
    for e in emotes:
        key = normalize(e) or e.strip().casefold()
        total += lex.chat.get(key, 0.0)
    whole = normalize(text)
    if whole in lex.chat:
        return max(total, lex.chat[whole])
    for tok in whole.split():
        if tok in lex.chat:
            total += lex.chat[tok]
        elif LAUGH_RE.match(tok):
            total += lex.laugh_weight
    for ch in text:  # raw emoji characters
        if not ch.isascii() and ch in lex.chat:
            total += lex.chat[ch]
    return min(total, 3.0)
