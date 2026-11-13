"""Word-by-word ASS subtitles (pure functions).

Each chunk of 2-4 words is shown while it is being spoken; the word being said is highlighted.
Implementation: one Dialogue event per word, showing the whole chunk with that word styled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from clipador.render.styles import SubtitleStyle

SubWord = tuple[float, float, str]  # (start, end, text) relative to the clip start


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    sub_margin_v: int  # distance of the subtitle baseline from the bottom
    title_margin_v: int  # distance of the title from the top
    font_scale: float = 1.0


# ---------------------------------------------------------------------------------------------


def clean_word(text: str) -> str:
    """ASS-safe text: braces start override blocks and backslashes start escapes."""
    t = text.replace("{", "(").replace("}", ")").replace("\\", "/")
    return " ".join(t.split())


def words_in_window(words: list[SubWord], start: float, end: float) -> list[SubWord]:
    """Absolute transcript words -> words relative to the clip, clipped to [0, end-start]."""
    dur = end - start
    out = []
    for s, e, w in words:
        if e <= start or s >= end:
            continue
        rs, re_ = max(0.0, s - start), min(dur, e - start)
        if re_ - rs > 0.01 and clean_word(w):
            out.append((round(rs, 3), round(re_, 3), clean_word(w)))
    return out


_TOKEN = re.compile(r"\S+")


def align_edited_text(original: list[SubWord], edited: str) -> list[SubWord]:
    """User-corrected subtitle text -> timed words.

    Same number of words: keep each original timing. Otherwise spread the new words over the
    original speech span, proportionally to their length (good enough for short clips).
    """
    tokens = [clean_word(t) for t in _TOKEN.findall(edited or "")]
    tokens = [t for t in tokens if t]
    if not tokens:
        return []
    if not original:
        return []
    if len(tokens) == len(original):
        return [(s, e, t) for (s, e, _), t in zip(original, tokens, strict=True)]
    span_start, span_end = original[0][0], original[-1][1]
    total_chars = sum(len(t) + 1 for t in tokens)
    out, t = [], span_start
    for tok in tokens:
        d = (span_end - span_start) * (len(tok) + 1) / total_chars
        out.append((round(t, 3), round(t + d * 0.92, 3), tok))
        t += d
    return out


def chunk_words(
    words: list[SubWord], max_words: int = 3, max_chars: int = 18, max_gap: float = 0.6
) -> list[list[SubWord]]:
    chunks: list[list[SubWord]] = []
    cur: list[SubWord] = []
    for w in words:
        if cur:
            gap = w[0] - cur[-1][1]
            chars = sum(len(x[2]) for x in cur) + len(cur) + len(w[2])
            ends_sentence = cur[-1][2][-1:] in ".!?"
            if len(cur) >= max_words or chars > max_chars or gap > max_gap or ends_sentence:
                chunks.append(cur)
                cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = round(t * 100)
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _color(rgb: str, alpha: int = 0) -> str:
    """'#RRGGBB' -> ASS '&HAABBGGRR'."""
    r, g, b = rgb[1:3], rgb[3:5], rgb[5:7]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def build_ass(
    words: list[SubWord],
    duration: float,
    style: SubtitleStyle,
    layout: Layout,
    title: str | None = None,
) -> str:
    fs = round(style.font_size * layout.font_scale)
    tfs = round(style.title_size * layout.font_scale)
    # With BorderStyle=3 (opaque box) libass paints the box with the OUTLINE colour.
    border_style = 3 if style.box else 1
    outline_c = (
        _color(style.box_color, style.box_alpha) if style.box else _color(style.outline_color)
    )
    back = _color("#000000", 0x80)
    title_box = _color(style.title_box, 0x20)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {layout.width}
PlayResY: {layout.height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,{style.font},{fs},{_color(style.color)},{_color(style.highlight)},{outline_c},{back},-1,0,0,0,100,100,0,0,{border_style},{style.outline},{style.shadow},2,60,60,{layout.sub_margin_v},1
Style: Title,{style.font},{tfs},{_color(style.title_color)},{_color(style.title_color)},{title_box},{title_box},-1,0,0,0,100,100,0,0,3,14,0,8,70,70,{layout.title_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""  # noqa: E501
    lines = [header]
    if title:
        t = clean_word(title)
        lines.append(f"Dialogue: 1,{ass_time(0)},{ass_time(duration)},Title,,0,0,0,,{t}\n")

    def fmt(text: str) -> str:
        return text.upper() if style.uppercase else text

    hl = _color(style.highlight)
    base = _color(style.color)
    pop = f"\\fscx{style.pop}\\fscy{style.pop}" if style.pop != 100 else ""
    chunks = chunk_words(words, style.max_words, style.max_chars)
    for ci, chunk in enumerate(chunks):
        next_start = chunks[ci + 1][0][0] if ci + 1 < len(chunks) else duration
        chunk_end = min(next_start, chunk[-1][1] + 0.35, duration)
        for wi, (ws, _we, _w) in enumerate(chunk):
            end = chunk[wi + 1][0] if wi + 1 < len(chunk) else chunk_end
            if end - ws < 0.04:
                continue
            parts = []
            for wj, (_s, _e, w) in enumerate(chunk):
                if wj == wi:
                    parts.append(f"{{\\c{hl}{pop}}}{fmt(w)}{{\\c{base}\\fscx100\\fscy100}}")
                else:
                    parts.append(fmt(w))
            lines.append(
                f"Dialogue: 0,{ass_time(ws)},{ass_time(end)},Sub,,0,0,0,,{' '.join(parts)}\n"
            )
    return "".join(lines)
