"""Per-platform post drafts, built by code from the LLM output (or a simple fallback).

The streamer credit line is always added here, never by the LLM.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from clipador.llm.schema import EnrichOutput, truncate_words
from clipador.models import PLATFORM_LABELS, Platform
from clipador.utils import format_hms

DRAFT_PLATFORMS = ("youtube", "tiktok", "instagram")
DRAFT_LABELS = {"youtube": "YouTube Shorts", "tiktok": "TikTok", "instagram": "Instagram Reels"}

YT_TITLE_MAX = 70  # Shorts titles get cut in the feed after ~70 chars (hard limit 100)
YT_DESC_MAX = 5000
TIKTOK_CAPTION_MAX = 150
IG_CAPTION_MAX = 2200
HASHTAG_MAX_LEN = 30

_HASHTAG_BAD = re.compile(r"[^0-9A-Za-z_]")


def credit_line(credit_text: str | None, streamer: str, source_platform: str) -> str:
    if credit_text and credit_text.strip():
        return " ".join(credit_text.split())
    try:
        label = PLATFORM_LABELS[Platform(source_platform)]
    except ValueError:
        label = source_platform
    return f"Clip de {streamer} ({label})"


def sanitize_hashtag(tag: str) -> str | None:
    """'#Perseguição Policial' -> '#PerseguicaoPolicial'; returns None if nothing is left."""
    t = unicodedata.normalize("NFKD", tag)
    t = "".join(c for c in t if not unicodedata.combining(c))
    words = [w for w in re.split(r"\s+", t.replace("#", " ").strip()) if w]
    if not words:
        return None
    joined = words[0] + "".join(w[:1].upper() + w[1:] for w in words[1:])
    joined = _HASHTAG_BAD.sub("", joined)[:HASHTAG_MAX_LEN]
    if not joined or joined.isdigit():
        return None
    return "#" + joined


def clean_hashtags(
    tags: list[str], *, required: tuple[str, ...] | list[str] = (), max_n: int, min_n: int = 0,
    extra: tuple[str, ...] | list[str] = (),
) -> list[str]:  # fmt: skip
    """Sanitize, dedupe (case-insensitive), required tags first, cap at max_n; `extra` only pads
    up to min_n when the LLM gave too few."""
    out: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        h = sanitize_hashtag(raw)
        if h and h.lower() not in seen and len(out) < max_n:
            seen.add(h.lower())
            out.append(h)

    for raw in [*required, *tags]:
        add(raw)
    for raw in extra:
        if len(out) >= min_n:
            break
        add(raw)
    return out


def clean_text(text: str, one_line: bool = False) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text or "")
    text = text.replace("<", "").replace(">", "")  # YouTube rejects angle brackets
    if one_line:
        text = " ".join(text.split())
    return text.strip()


def fallback_title(snippet: str, streamer: str, start: float) -> str:
    """No LLM: first sentence of what was said (editable), or a neutral label."""
    s = " ".join((snippet or "").split())
    first = re.split(r"(?<=[.!?])\s", s, maxsplit=1)[0] if s else ""
    if len(first) >= 8:
        first = first[:1].upper() + first[1:]
        return truncate_words(clean_text(first, True), YT_TITLE_MAX)
    return truncate_words(f"Momento de {streamer} na live ({format_hms(start)})", YT_TITLE_MAX)


@dataclass
class DraftData:
    title: str
    description: str
    hashtags: list[str] = field(default_factory=list)


def _with_credit(body: str, credit: str, limit: int, sep: str = "\n\n") -> str:
    """body + credit within `limit`; the body is shortened, the credit never dropped."""
    credit = truncate_words(credit, limit)
    room = limit - len(credit) - len(sep)
    body = truncate_words(body, room) if room > 10 else ""
    return f"{body}{sep}{credit}" if body else credit


def build_drafts(
    out: EnrichOutput | None,
    *,
    streamer: str,
    credit_text: str | None,
    source_platform: str,
    source_url: str | None,
    game: str,
    snippet: str,
    start: float,
) -> dict[str, DraftData]:
    credit = credit_line(credit_text, streamer, source_platform)
    game_tag = [game] if game else []
    if out is not None:
        title = clean_text(out.titulos[0], one_line=True)
        desc = clean_text(out.descricao)
        tags = list(out.hashtags)
    else:
        title = fallback_title(snippet, streamer, start)
        desc = ""
        tags = []
    title = truncate_words(title, YT_TITLE_MAX)
    defaults = [*game_tag, "gameplay", "clips"]

    yt_lines = [desc] if desc else []
    yt_lines.append(credit)
    if source_url:
        yt_lines.append(f"Original: {source_url}")
    yt_desc = "\n\n".join(yt_lines)[:YT_DESC_MAX]  # LLM text is <= 300 chars: credit always fits

    short_caption = clean_text(out.texto_de_abertura or title, True) if out else title
    tiktok = _with_credit(short_caption, credit, TIKTOK_CAPTION_MAX, sep=" | ")
    insta = _with_credit(desc or title, credit, IG_CAPTION_MAX)

    return {
        "youtube": DraftData(
            title,
            yt_desc,
            clean_hashtags(tags, required=["shorts"], max_n=6, min_n=3, extra=defaults),
        ),
        "tiktok": DraftData("", tiktok, clean_hashtags(tags, max_n=5, min_n=3, extra=defaults)),
        "instagram": DraftData("", insta, clean_hashtags(tags, max_n=5, min_n=3, extra=defaults)),
    }
