"""Word-by-word ASS captions + opening text + credit (pure functions).

Times are relative to the real clip start. Words keep their GLOBAL transcript index so
`Candidate.caption_overrides` ({index: text}) stays valid if the clip edges move.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from clipador.render.styles import EMBEDDED_FONT, SubtitleStyle


@dataclass(frozen=True)
class IWord:
    idx: int  # index in the full transcript
    start: float  # seconds from the clip start
    end: float
    text: str


@dataclass(frozen=True)
class SafeZone:
    width: int
    height: int
    caption_margin_v: int  # bottom margin of the captions (above the platform UI)
    top_margin: int  # opening text: below the top band covered by the platform UI
    credit_margin: int  # credit line: below the opening text
    font_scale: float = 1.0


def safe_zone(
    width: int, height: int, safe_bottom: float, safe_top: float, lift: float,
    font_scale: float | None = None,
) -> SafeZone:  # fmt: skip
    scale = font_scale if font_scale is not None else height / 1920
    top = round(height * safe_top)
    return SafeZone(
        width,
        height,
        caption_margin_v=round(height * (safe_bottom + lift)),
        top_margin=top + round(20 * scale),
        credit_margin=top + round(190 * scale),
        font_scale=scale,
    )


@dataclass(frozen=True)
class Overlays:
    opening_text: str | None = None
    opening_seconds: float = 2.0  # <= 0: whole clip
    credit: str | None = None


# --- text hygiene ----------------------------------------------------------------------------

_DROP_CATEGORIES = {"So", "Cn", "Co", "Cs", "Cc", "Cf"}


def clean_text(text: str) -> str:
    """ASS-safe and font-safe: braces would open override blocks, backslashes start escapes
    (\\N, \\h…), line breaks become spaces; emojis and symbols the font can't draw are dropped."""
    t = (text or "").replace("{", "(").replace("}", ")").replace("\\", "/")
    t = re.sub(r"[\r\n\t]+", " ", t)
    t = "".join(
        c for c in t if ord(c) <= 0xFFFF and unicodedata.category(c) not in _DROP_CATEGORIES
    )
    return " ".join(t.split())


def words_in_window(words: list[tuple[float, float, str]], start: float, end: float) -> list[IWord]:
    """Absolute transcript words -> words relative to the clip (global index kept)."""
    dur = end - start
    out = []
    for i, (s, e, w) in enumerate(words):
        if e <= start or s >= end:
            continue
        rs, re_ = max(0.0, s - start), min(dur, e - start)
        text = clean_text(w)
        if re_ - rs > 0.01 and text:
            out.append(IWord(i, round(rs, 3), round(re_, 3), text))
    return out


def apply_overrides(words: list[IWord], overrides: dict | None) -> list[IWord]:
    """Replace the text of words by global index; timing never changes. Empty text = hide."""
    if not overrides:
        return words
    ov = {int(k): v for k, v in overrides.items()}
    out = []
    for w in words:
        if w.idx in ov:
            text = clean_text(str(ov[w.idx]))
            if not text:
                continue
            w = IWord(w.idx, w.start, w.end, text)
        out.append(w)
    return out


# --- grouping --------------------------------------------------------------------------------


def chunk_words(
    words: list[IWord], max_words: int = 3, max_line_chars: int = 16, max_gap: float = 0.6
) -> list[list[IWord]]:
    """2-4 words per event, at most 2 lines; break on pauses and sentence punctuation."""
    chunks: list[list[IWord]] = []
    cur: list[IWord] = []
    for w in words:
        if cur:
            gap = w.start - cur[-1].end
            chars = sum(len(x.text) for x in cur) + len(cur) + len(w.text)
            ends_sentence = cur[-1].text[-1:] in ".!?…"
            if (len(cur) >= max_words or chars > 2 * max_line_chars or gap > max_gap
                    or ends_sentence):  # fmt: skip
                chunks.append(cur)
                cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def split_lines(texts: list[str], max_line_chars: int) -> int | None:
    """Index where to break into a 2nd line (balanced), or None if one line fits."""
    if len(texts) < 2 or sum(len(t) for t in texts) + len(texts) - 1 <= max_line_chars:
        return None
    best, best_cost = 1, 10**9
    for k in range(1, len(texts)):
        a = sum(len(t) for t in texts[:k]) + k - 1
        b = sum(len(t) for t in texts[k:]) + len(texts) - k - 1
        if max(a, b) < best_cost:
            best, best_cost = k, max(a, b)
    return best


def highlight_intervals(
    chunk: list[IWord], chunk_end: float, min_event: float
) -> list[tuple[float, float, list[int]]]:
    """(start, end, highlighted positions) per event; a word that would flash for less than
    `min_event` is merged with its neighbour, so no event is shorter than that."""
    raw = []
    for i, w in enumerate(chunk):
        end = chunk[i + 1].start if i + 1 < len(chunk) else chunk_end
        raw.append([w.start, end, [i]])
    merged: list[list] = []
    for iv in raw:
        if merged and merged[-1][1] - merged[-1][0] < min_event:
            merged[-1][1] = iv[1]
            merged[-1][2] += iv[2]
        else:
            merged.append(iv)
    if len(merged) > 1 and merged[-1][1] - merged[-1][0] < min_event:
        last = merged.pop()
        merged[-1][1] = last[1]
        merged[-1][2] += last[2]
    return [(s, e, pos) for s, e, pos in merged]


# --- ASS ---------------------------------------------------------------------------------------


def ass_time(t: float) -> str:
    cs = round(max(0.0, t) * 100)
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _color(rgb: str, alpha: int = 0) -> str:
    """'#RRGGBB' -> ASS '&HAABBGGRR'."""
    r, g, b = rgb[1:3], rgb[3:5], rgb[5:7]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def build_ass(
    words: list[IWord],
    duration: float,
    style: SubtitleStyle,
    zone: SafeZone,
    overlays: Overlays | None = None,
    min_event: float = 0.25,
    captions: bool = True,
) -> str:
    ov = overlays or Overlays()
    sc = zone.font_scale
    fs = round(style.font_size * sc)
    border_style = 3 if style.box else 1
    # with BorderStyle=3 (opaque box) libass paints the box with the OUTLINE colour
    outline_c = (
        _color(style.box_color, style.box_alpha) if style.box else _color(style.outline_color)
    )
    head = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {zone.width}",
        f"PlayResY: {zone.height}", "WrapStyle: 2", "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709", "", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Sub,{style.font},{fs},{_color(style.color)},{_color(style.highlight)},"
        f"{outline_c},&H80000000,0,0,0,0,100,100,0,0,{border_style},{round(style.outline * sc)},"
        f"{style.shadow},2,{round(60 * sc)},{round(60 * sc)},{zone.caption_margin_v},1",
        f"Style: Opening,{EMBEDDED_FONT},{round(70 * sc)},&H00FFFFFF,&H00FFFFFF,&H20000000,"
        f"&H20000000,0,0,0,0,100,100,0,0,3,{round(16 * sc)},0,8,{round(70 * sc)},"
        f"{round(70 * sc)},{zone.top_margin},1",
        f"Style: Credit,{EMBEDDED_FONT},{round(36 * sc)},&H30FFFFFF,&H30FFFFFF,&H80000000,"
        f"&H80000000,0,0,0,0,100,100,1,0,1,{max(1, round(2 * sc))},0,8,{round(60 * sc)},"
        f"{round(60 * sc)},{zone.credit_margin},1",
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]  # fmt: skip
    ev: list[str] = []

    def dialogue(layer: int, s: float, e: float, st: str, text: str) -> None:
        ev.append(f"Dialogue: {layer},{ass_time(s)},{ass_time(e)},{st},,0,0,0,,{text}")

    opening = clean_text(ov.opening_text or "")
    if opening:
        end = duration if ov.opening_seconds <= 0 else min(duration, ov.opening_seconds)
        dialogue(2, 0, end, "Opening", "{\\fad(150,250)}" + opening)
    credit = clean_text(ov.credit or "")
    if credit:
        dialogue(1, 0, duration, "Credit", credit)

    if captions:
        hl, base = _color(style.highlight), _color(style.color)
        chunks = chunk_words(words, style.max_words, style.max_line_chars)
        for ci, chunk in enumerate(chunks):
            next_start = chunks[ci + 1][0].start if ci + 1 < len(chunks) else duration
            chunk_end = min(next_start, chunk[-1].end + 0.35, duration)
            if chunk_end - chunk[0].start < min_event:
                chunk_end = min(next_start, duration, chunk[0].start + min_event)
            texts = [x.text.upper() if style.uppercase else x.text for x in chunk]
            brk = split_lines(texts, style.max_line_chars)

            def join(parts: list[str], brk=brk) -> str:
                if brk is None:
                    return " ".join(parts)
                return " ".join(parts[:brk]) + "\\N" + " ".join(parts[brk:])

            if style.mode == "classico":
                dialogue(0, chunk[0].start, chunk_end, "Sub", join(texts))
                continue
            for s, e, positions in highlight_intervals(chunk, chunk_end, min_event):
                parts = []
                for i, t in enumerate(texts):
                    if i not in positions:
                        parts.append(t)
                    elif style.mode == "pop":
                        p = style.pop
                        parts.append(
                            f"{{\\c{hl}\\t(0,90,\\fscx{p}\\fscy{p})\\t(90,180,\\fscx100\\fscy100)}}"
                            f"{t}{{\\c{base}\\fscx100\\fscy100}}"
                        )
                    else:
                        parts.append(f"{{\\c{hl}}}{t}{{\\c{base}}}")
                dialogue(0, s, e, "Sub", join(parts))
    return "\n".join([*head, *ev]) + "\n"


_HANDLE_RE = re.compile(r"@[\w.]+")


def credit_overlay(credit_text: str | None, streamer_name: str) -> str:
    """'Clip de @fulano na Twitch' -> 'via @fulano'; otherwise 'via <name>'."""
    m = _HANDLE_RE.search(credit_text or "")
    return f"via {m.group(0)}" if m else f"via {streamer_name}"
