"""Per-account edit styles (step 6C): every social account posts its OWN version of a clip, so
the platforms don't see the same file on several profiles (duplicated content gets held back).

A style changes things a viewer barely notices but that make each file genuinely different:
layout, caption style, color grade, speed (pitch kept), crop zoom/framing, a small trim at the
start, which AI title is the opening text, credit and peak zoom.
"""

from __future__ import annotations

import random
from typing import Any

# key -> (label, ffmpeg video filters applied to the composed 9:16 frame, before captions)
GRADES: dict[str, tuple[str, str]] = {
    "nenhum": ("Original", ""),
    "vivo": ("Vivo (mais cor e contraste)", "eq=contrast=1.08:saturation=1.25"),
    "quente": ("Quente", "colorbalance=rs=.06:gs=.02:bs=-.06,eq=saturation=1.08"),
    "frio": ("Frio", "colorbalance=rs=-.05:bs=.07,eq=saturation=1.05"),
    "suave": ("Suave", "eq=contrast=.95:saturation=.9:brightness=.02"),
    "cinema": ("Cinema", "eq=contrast=1.12:saturation=.92,colorbalance=rm=.03:bm=-.03"),
}
SPEEDS = (1.0, 1.03, 1.05, 1.08, 1.1)
TRIMS = (0.0, 0.3, 0.6, 1.0)
OPENINGS = {"auto": "Título sugerido", "t1": "1º título da IA", "t2": "2º título da IA",
            "t3": "3º título da IA", "nenhum": "Sem texto de abertura"}  # fmt: skip
LAYOUT_CHOICES = {"": "Padrão do streamer", "blur": "Fundo borrado",
                  "facecam_embaixo": "Facecam embaixo", "jogo_cheio": "Jogo cheio"}  # fmt: skip

DEFAULTS: dict[str, Any] = {
    "layout": "", "caption_style": "", "grade": "nenhum", "speed": 1.0, "zoom": 1.0,
    "foco_x": 0.5, "trim_start": 0.0, "opening": "auto", "credit": True, "peak_zoom": True,
}  # fmt: skip

# Distinct looks handed to new accounts in order (the first one is the "normal" edit).
PRESETS: list[dict[str, Any]] = [
    {"grade": "nenhum", "speed": 1.0, "opening": "auto"},
    {
        "grade": "vivo",
        "speed": 1.05,
        "caption_style": "pop",
        "opening": "t2",
        "zoom": 1.08,
        "trim_start": 0.3,
    },
    {
        "grade": "quente",
        "speed": 1.03,
        "caption_style": "classico",
        "opening": "t3",
        "zoom": 1.15,
        "foco_x": 0.45,
        "trim_start": 0.6,
    },
    {
        "grade": "frio",
        "speed": 1.08,
        "caption_style": "caixa",
        "opening": "t2",
        "zoom": 1.05,
        "foco_x": 0.55,
        "trim_start": 1.0,
        "peak_zoom": False,
    },
    {
        "grade": "cinema",
        "speed": 1.05,
        "caption_style": "neon",
        "opening": "t3",
        "zoom": 1.12,
        "trim_start": 0.3,
    },
    {
        "grade": "suave",
        "speed": 1.1,
        "caption_style": "destaque_amarelo",
        "opening": "t1",
        "zoom": 1.2,
        "foco_x": 0.4,
        "trim_start": 0.6,
    },
]


def normalize(style: dict[str, Any] | None) -> dict[str, Any]:
    """Clean, bounded copy with every key (unknown keys dropped)."""
    s = {**DEFAULTS, **{k: v for k, v in (style or {}).items() if k in DEFAULTS}}
    s["layout"] = s["layout"] if s["layout"] in LAYOUT_CHOICES else ""
    s["grade"] = s["grade"] if s["grade"] in GRADES else "nenhum"
    s["opening"] = s["opening"] if s["opening"] in OPENINGS else "auto"
    s["speed"] = min(max(float(s["speed"] or 1.0), 1.0), 1.15)
    s["zoom"] = min(max(float(s["zoom"] or 1.0), 1.0), 1.5)
    s["foco_x"] = min(max(float(s["foco_x"]), 0.0), 1.0)
    s["trim_start"] = min(max(float(s["trim_start"] or 0.0), 0.0), 2.0)
    s["credit"], s["peak_zoom"] = bool(s["credit"]), bool(s["peak_zoom"])
    s["caption_style"] = str(s["caption_style"] or "")
    return s


def is_default(style: dict[str, Any] | None) -> bool:
    """True when the style changes nothing (the account can post the normal render)."""
    return not style or normalize(style) == normalize({})


def suggest(existing: list[dict[str, Any]], seed: int | None = None) -> dict[str, Any]:
    """A look different from the accounts that already exist: the next unused preset, or a
    random combination once the presets run out."""
    used = [normalize(e) for e in existing]
    for p in PRESETS:
        if normalize(p) not in used:
            return normalize(p)
    rng = random.Random(seed)
    while True:
        cand = normalize({
            "grade": rng.choice(list(GRADES)), "speed": rng.choice(SPEEDS[1:]),
            "caption_style": rng.choice(["pop", "classico", "caixa", "neon", "destaque_amarelo"]),
            "opening": rng.choice(["t1", "t2", "t3"]), "zoom": rng.choice([1.05, 1.1, 1.15, 1.2]),
            "foco_x": rng.choice([0.4, 0.45, 0.5, 0.55, 0.6]), "trim_start": rng.choice(TRIMS),
            "peak_zoom": rng.random() < 0.7,
        })  # fmt: skip
        if cand not in used:
            return cand


def describe(style: dict[str, Any] | None) -> str:
    """Short PT-BR summary for the account card."""
    s = normalize(style)
    if is_default(s):
        return "edição padrão"
    parts = []
    if s["layout"]:
        parts.append(LAYOUT_CHOICES[s["layout"]].lower())
    if s["caption_style"]:
        parts.append(f"legenda {s['caption_style']}")
    if s["grade"] != "nenhum":
        parts.append(f"cor {GRADES[s['grade']][0].split(' ')[0].lower()}")
    if s["speed"] != 1.0:
        parts.append(f"{s['speed']:g}x")
    if s["zoom"] != 1.0:
        parts.append(f"zoom {s['zoom']:g}")
    if s["trim_start"]:
        parts.append(f"começa +{s['trim_start']:g}s")
    if s["opening"] != "auto":
        parts.append(OPENINGS[s["opening"]].lower())
    return " · ".join(parts)
