"""Subtitle style presets (ASS). Fonts are system fonts resolved by libass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubtitleStyle:
    key: str
    label: str
    font: str = "Arial Black"
    font_size: int = 92
    color: str = "#FFFFFF"
    highlight: str = "#FFE000"  # word being spoken
    outline_color: str = "#000000"
    outline: int = 6
    shadow: int = 0
    box: bool = False
    box_color: str = "#000000"
    box_alpha: int = 0x60  # 0 = opaque, 255 = transparent
    uppercase: bool = True
    pop: int = 115  # % scale of the highlighted word (100 = no pop)
    max_words: int = 3
    max_chars: int = 18
    title_size: int = 64
    title_color: str = "#FFFFFF"
    title_box: str = "#E0245E"


STYLES: dict[str, SubtitleStyle] = {
    s.key: s
    for s in (
        SubtitleStyle("classico", "Clássico (branco, palavra amarela)"),
        SubtitleStyle(
            "caixa", "Caixa escura", box=True, outline=10, highlight="#3DF57A", pop=100,
            title_box="#111111",
        ),
        SubtitleStyle(
            "neon", "Neon", highlight="#00F0FF", outline_color="#2A0055", outline=7,
            title_box="#7C5CFF", max_words=2,
        ),
        SubtitleStyle("minimo", "Mínimo (menor, sem caixa alta)", font="Arial", font_size=72,
                      uppercase=False, outline=5, pop=100, max_words=4, max_chars=24),
    )
}  # fmt: skip
DEFAULT_STYLE = "classico"


def get_style(key: str | None) -> SubtitleStyle:
    return STYLES.get(key or DEFAULT_STYLE, STYLES[DEFAULT_STYLE])
