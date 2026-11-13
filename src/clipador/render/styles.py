"""Caption styles, loaded from config/caption_styles.toml (editable)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STYLES_FILE = REPO_ROOT / "config" / "caption_styles.toml"
DEFAULT_FONTS_DIR = REPO_ROOT / "assets" / "fonts"
EMBEDDED_FONT = "Anton"
DEFAULT_STYLE = "destaque_amarelo"
MODES = ("classico", "destaque", "pop")


@dataclass(frozen=True)
class SubtitleStyle:
    key: str
    label: str
    mode: str = "destaque"
    font: str = EMBEDDED_FONT
    font_size: int = 96
    color: str = "#FFFFFF"
    highlight: str = "#FFE000"
    outline_color: str = "#000000"
    outline: int = 7
    shadow: int = 0
    box: bool = False
    box_color: str = "#000000"
    box_alpha: int = 0x60
    uppercase: bool = True
    pop: int = 118
    max_words: int = 3
    max_line_chars: int = 16


_KEYS = {
    "nome": "label", "modo": "mode", "fonte": "font", "tamanho": "font_size", "cor": "color",
    "cor_destaque": "highlight", "cor_contorno": "outline_color", "contorno": "outline",
    "sombra": "shadow", "caixa": "box", "cor_caixa": "box_color",
    "transparencia_caixa": "box_alpha", "caixa_alta": "uppercase", "escala_pop": "pop",
    "palavras_por_grupo": "max_words", "caracteres_por_linha": "max_line_chars",
}  # fmt: skip


def parse_styles(text: str) -> dict[str, SubtitleStyle]:
    data = tomllib.loads(text)
    out = {}
    for key, raw in data.items():
        if not isinstance(raw, dict):
            continue
        kw = {_KEYS[k]: v for k, v in raw.items() if k in _KEYS}
        kw.setdefault("label", key)
        if kw.get("mode") not in MODES:
            kw["mode"] = "destaque"
        kw["max_words"] = max(1, min(4, int(kw.get("max_words", 3))))
        out[key] = SubtitleStyle(key=key, **kw)
    if not out:
        raise ValueError("nenhum estilo de legenda no arquivo")
    return out


@lru_cache(maxsize=8)
def _load(path: str, mtime: float) -> dict[str, SubtitleStyle]:
    return parse_styles(Path(path).read_text("utf-8"))


def load_styles(path: Path | None = None) -> dict[str, SubtitleStyle]:
    p = path or DEFAULT_STYLES_FILE
    return _load(str(p), p.stat().st_mtime)


def get_style(key: str | None, path: Path | None = None) -> SubtitleStyle:
    styles = load_styles(path)
    if key in styles:
        return styles[key]
    return styles.get(DEFAULT_STYLE) or next(iter(styles.values()))


class _StylesProxy(dict):
    """`STYLES` kept for templates/older code: always reflects the TOML file."""

    def _d(self):
        return load_styles()

    def __getitem__(self, k):
        return self._d()[k]

    def __contains__(self, k):
        return k in self._d()

    def __iter__(self):
        return iter(self._d())

    def __len__(self):
        return len(self._d())

    def items(self):
        return self._d().items()

    def keys(self):
        return self._d().keys()

    def values(self):
        return self._d().values()

    def get(self, k, default=None):
        return self._d().get(k, default)


STYLES = _StylesProxy()
