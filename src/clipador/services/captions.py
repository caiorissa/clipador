"""Caption editor (5B): per-word corrections of a candidate and the streamer glossary.

Corrections live in `Candidate.caption_overrides` ({global index: text}, "" = hidden) with the
transcript text of each corrected word in `Candidate.caption_override_originals`. The saved
transcript is never modified. Functions take a Session and never commit (the caller does).
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session

from clipador.config import Settings
from clipador.errors import ValidationError
from clipador.models import Candidate, Source, Streamer
from clipador.render.subtitles import (
    IWord,
    StaleOverride,
    apply_glossary,
    caption_words,
    clean_text,
    norm_word,
    resolve_overrides,
    split_punct,
    words_in_window,
)
from clipador.services import rendering
from clipador.utils import utcnow

GLOSSARY_MAX = 500


def streamer_of(session: Session, c: Candidate) -> Streamer | None:
    src = session.get(Source, c.source_id)
    return session.get(Streamer, src.streamer_id) if src else None


def shown_words(
    words_all: list, c: Candidate, glossary: dict | None, start: float | None = None,
    end: float | None = None,
) -> tuple[list[IWord], list[StaleOverride]]:  # fmt: skip
    """Caption words of [start, end] (default: the clip range), relative to `start`."""
    a = c.eff_start if start is None else start
    b = c.eff_end if end is None else end
    return caption_words(
        words_in_window(words_all, a, b), glossary, c.caption_overrides,
        c.caption_override_originals,
    )  # fmt: skip


@dataclass(frozen=True)
class EditorWord:
    """One row of the editor: the transcript word and what the caption shows for it."""

    idx: int
    start: float  # absolute (original video)
    end: float
    raw: str  # transcript text
    glossary: str | None  # text after the glossary (None = glossary didn't touch it)
    override: str | None  # valid per-word correction ("" = hidden)
    stale: bool

    @property
    def shown(self) -> str:
        if self.override is not None:
            return clean_text(self.override)
        return self.glossary if self.glossary is not None else self.raw

    def as_json(self) -> dict:
        return {"i": self.idx, "s": self.start, "e": self.end, "raw": self.raw,
                "gloss": self.glossary, "ov": self.override, "t": self.shown,
                "hidden": self.override is not None and not self.shown,
                "stale": self.stale}  # fmt: skip


def editor_words(
    words_all: list, c: Candidate, glossary: dict | None, a: float, b: float
) -> tuple[list[EditorWord], list[StaleOverride]]:
    window = words_in_window(words_all, a, b)
    valid, stale = resolve_overrides(window, c.caption_overrides, c.caption_override_originals)
    glossed = {w.idx: w.text for w in apply_glossary(window, glossary)}
    stale_idx = {s.idx for s in stale}
    rows = [
        EditorWord(w.idx, round(a + w.start, 3), round(a + w.end, 3), w.text,
                   glossed[w.idx] if glossed[w.idx] != w.text else None,
                   valid.get(w.idx), w.idx in stale_idx)
        for w in window
    ]  # fmt: skip
    return rows, stale


def _set(c: Candidate, idx: int, text: str | None, original: str | None) -> None:
    ov = {str(k): v for k, v in (c.caption_overrides or {}).items()}
    orig = {str(k): v for k, v in (c.caption_override_originals or {}).items()}
    key = str(idx)
    if text is None:
        ov.pop(key, None)
        orig.pop(key, None)
    else:
        ov[key] = text
        if original is not None:
            orig[key] = original
    c.caption_overrides = ov or None
    c.caption_override_originals = orig or None
    flag_modified(c, "caption_overrides")
    flag_modified(c, "caption_override_originals")
    c.updated_at = utcnow()


def add_glossary_entry(streamer: Streamer, wrong: str, right: str, old: str | None = None) -> str:
    """Add/replace {wrong: right} (an entry with the same normalized key is replaced; `old`
    renames an entry). Returns the key stored. Raises ValidationError (PT-BR)."""
    wrong_core = split_punct(clean_text(wrong))[1]
    right_txt = clean_text(right)
    if not wrong_core or not norm_word(wrong_core):
        raise ValidationError("Informe a palavra errada.")
    if " " in wrong_core:
        raise ValidationError("O glossário troca uma palavra inteira por vez (sem espaços).")
    if not right_txt:
        raise ValidationError("Informe a correção (para esconder, use o editor do clip).")
    g = dict(streamer.glossary or {})
    drop = {norm_word(wrong_core)} | ({norm_word(old)} if old else set())
    g = {k: v for k, v in g.items() if norm_word(k) not in drop}
    if len(g) >= GLOSSARY_MAX:
        raise ValidationError(f"O glossário já tem {GLOSSARY_MAX} entradas.")
    g[wrong_core] = right_txt
    streamer.glossary = dict(sorted(g.items(), key=lambda kv: norm_word(kv[0])))
    flag_modified(streamer, "glossary")
    streamer.updated_at = utcnow()
    return wrong_core


def remove_glossary_entry(streamer: Streamer, wrong: str) -> bool:
    g = dict(streamer.glossary or {})
    key = norm_word(wrong)
    new = {k: v for k, v in g.items() if norm_word(k) != key}
    if len(new) == len(g):
        return False
    streamer.glossary = new or None
    flag_modified(streamer, "glossary")
    streamer.updated_at = utcnow()
    return True


def edit_word(
    session: Session,
    settings: Settings,
    c: Candidate,
    idx: int,
    text: str | None = None,
    *,
    hide: bool = False,
    undo: bool = False,
    to_glossary: bool = False,
) -> str:
    """Correct (text), hide or un-correct (undo) the word with global index `idx`; the timing
    is always the transcript's. With `to_glossary`, the correction also goes to the streamer's
    glossary (and the per-word override is dropped when the glossary already yields the same
    text). Returns a short PT-BR message."""
    words_all = rendering.transcript_words(session.get_bind(), settings, c.source_id)
    if not 0 <= idx < len(words_all):
        raise ValidationError("Palavra não encontrada na transcrição.")
    raw = clean_text(words_all[idx][2])
    st = streamer_of(session, c)
    if undo:
        _set(c, idx, None, None)
        return "Correção desfeita."
    new = "" if hide else clean_text(text or "")
    if not new:
        _set(c, idx, "", raw)
        return "Palavra oculta."
    msg = "Palavra corrigida."
    if to_glossary:
        if st is None:
            raise ValidationError("Esta fonte não tem streamer.")
        right = split_punct(new)[1] or new
        key = add_glossary_entry(st, raw, right)
        session.add(st)
        msg = f"Corrigida; glossário de {st.display_name}: “{key}” → “{right}”."
    gloss = st.glossary if st else None
    by_glossary = apply_glossary([IWord(idx, 0, 1, raw)], gloss)[0].text
    _set(c, idx, None if new == by_glossary else new, raw)
    return msg


def remove_stale(session: Session, settings: Settings, c: Candidate, idx: int | None) -> int:
    """Drop stale corrections (all of them, or only `idx`) of the words in the clip range."""
    words_all = rendering.transcript_words(session.get_bind(), settings, c.source_id)
    window = words_in_window(words_all, 0.0, float("inf"))
    _valid, stale = resolve_overrides(window, c.caption_overrides, c.caption_override_originals)
    n = 0
    for s in stale:
        if idx is None or s.idx == idx:
            _set(c, s.idx, None, None)
            n += 1
    return n
