"""Per-second features. Every signal ends up in 0..1 so they can be weighted and combined.

Index i of every array is the second [t0 + i, t0 + i + 1) of the ORIGINAL video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, median_filter, uniform_filter1d

from clipador.detection.hype import HypeLexicon, chat_message_weight, match_words, normalize
from clipador.detection.params import FeatureParams
from clipador.platforms.base import ChatMessage

SILENCE_DB = -90.0
SIGNAL_NAMES = ("audio", "speech", "keywords", "chat")


# --- generic helpers ---------------------------------------------------------------------------


def robust_local_z(x: np.ndarray, window: int, min_scale: float) -> np.ndarray:
    """(x - rolling median) / rolling robust std (MAD). A local baseline makes "loud" relative
    to the last few minutes, so slow volume changes (music, scene) don't count as peaks."""
    if x.size == 0:
        return x.astype(float)
    w = max(3, min(window, x.size) | 1)  # odd, <= length
    base = median_filter(x, size=w, mode="nearest")
    mad = median_filter(np.abs(x - base), size=w, mode="nearest")
    return (x - base) / np.maximum(1.4826 * mad, min_scale)


def squash(z: np.ndarray, scale: float) -> np.ndarray:
    """Map z >= 0 to 0..1 smoothly (negative -> 0)."""
    return 1.0 - np.exp(-np.maximum(z, 0.0) / scale)


def smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that keeps short events tall: dilate (3 s max) first, then blur."""
    if sigma <= 0 or x.size == 0:
        return x.astype(float)
    return gaussian_filter1d(maximum_filter1d(x.astype(float), size=3), sigma)


def per_second_counts(times: np.ndarray, t0: float, n: int, weights=None) -> np.ndarray:
    idx = np.floor(np.asarray(times, dtype=float) - t0).astype(int)
    ok = (idx >= 0) & (idx < n)
    return np.bincount(idx[ok], weights=None if weights is None else np.asarray(weights)[ok],
                       minlength=n).astype(float)  # fmt: skip


# --- audio -----------------------------------------------------------------------------------


def rms_db_frames(blocks, sample_rate: int, fps: int) -> np.ndarray:
    """RMS in dBFS per 1/fps-second frame, streaming over (t, samples) blocks."""
    hop = sample_rate // fps
    out = []
    carry = np.zeros(0, dtype=np.float32)
    for _t, samples in blocks:
        data = np.concatenate([carry, samples]) if carry.size else samples
        usable = (data.size // hop) * hop
        if usable:
            frames = data[:usable].reshape(-1, hop).astype(np.float64)
            out.append(np.mean(frames**2, axis=1))
        carry = data[usable:]
    if carry.size:
        out.append(np.array([np.mean(carry.astype(np.float64) ** 2)]))
    ms = np.concatenate(out) if out else np.zeros(0)
    return np.maximum(10 * np.log10(ms + 1e-12), SILENCE_DB)


def per_second_db(fine_db: np.ndarray, fps: int) -> np.ndarray:
    n = int(np.ceil(fine_db.size / fps))
    pad = np.full(n * fps - fine_db.size, SILENCE_DB)
    power = 10 ** (np.concatenate([fine_db, pad]).reshape(n, fps) / 10)
    return np.maximum(10 * np.log10(power.mean(axis=1) + 1e-12), SILENCE_DB)


def audio_signal(rms_db: np.ndarray, p: FeatureParams) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    z = robust_local_z(rms_db, p.baseline_window_s, min_scale=1.0)
    peak = squash(z, p.z_scale)
    # sudden rise vs the previous few seconds
    prev = np.concatenate([[rms_db[0]] * 3, rms_db[:-3]]) if rms_db.size > 3 else rms_db
    prev_mean = uniform_filter1d(prev, size=3, mode="nearest")
    rise = rms_db - prev_mean
    rise_scale = max(
        1.4826 * float(np.median(np.abs(rise - np.median(rise)))) if rise.size else 1.0, 1.0
    )
    rise_s = squash(rise / rise_scale, p.z_scale)
    # relative quiet right before a loud second ("silêncio dramático" -> explosão)
    lb = max(1, p.quiet_lookback_s)
    quiet_before = np.full(z.size, np.inf)
    if z.size > lb:
        cs = np.concatenate([[0.0], np.cumsum(z)])
        t = np.arange(lb, z.size)
        quiet_before[lb:] = (cs[t] - cs[t - lb]) / lb  # mean z over the lb seconds before t
    burst = ((quiet_before < p.quiet_z) & (z > 1.5)).astype(float)
    burst = maximum_filter1d(burst, size=3)
    sig = np.clip(0.75 * peak + 0.25 * np.maximum(rise_s, burst), 0, 1)
    return smooth(sig, p.smoothing_sigma_s), {"rms_z": z}


# --- speech --------------------------------------------------------------------------------


def speech_signals(
    words: list[tuple[float, float, str]], t0: float, n: int, lex: HypeLexicon, p: FeatureParams
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, str]]]:
    starts = np.array([w[0] for w in words], dtype=float)
    wps = per_second_counts(starts, t0, n)
    speech = squash(robust_local_z(smooth(wps, 1.0), p.baseline_window_s, min_scale=0.5), p.z_scale)
    hits = match_words([(w[0], w[2]) for w in words], lex)
    density = per_second_counts(
        np.array([h[0] for h in hits], dtype=float), t0, n, [h[1] for h in hits]
    )
    keywords = squash(smooth(density, p.smoothing_sigma_s * 1.5) * 2.0, 1.0)
    return smooth(speech, p.smoothing_sigma_s), keywords, hits


# --- chat ------------------------------------------------------------------------------------


def _is_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.7


def chat_signal(
    messages: list[ChatMessage], t0: float, n: int, lex: HypeLexicon, p: FeatureParams
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    # Chat reacts after the event: shift it back by the typical delay.
    times = np.array([m.t for m in messages], dtype=float) - p.chat_delay_s
    msgs = per_second_counts(times, t0, n)
    idx = np.floor(times - t0).astype(int)
    uniq = np.zeros(n)
    hype = np.zeros(n)
    caps = np.zeros(n)
    seen: dict[int, set[str]] = {}
    norm_msgs: list[tuple[int, str]] = []
    for i, m in zip(idx, messages, strict=True):
        if 0 <= i < n:
            seen.setdefault(int(i), set()).add(m.author)
            hype[i] += chat_message_weight(m.text, m.emotes, lex)
            caps[i] += _is_caps(m.text)
            norm_msgs.append((int(i), normalize(m.text) or m.text.strip()))
    for i, authors in seen.items():
        uniq[i] = len(authors)
    # repetition: share of the most repeated message in a sliding window
    rep = np.zeros(n)
    w = max(1, p.repeat_window_s)
    by_sec: dict[int, list[str]] = {}
    for i, text in norm_msgs:
        by_sec.setdefault(i, []).append(text)
    for i in by_sec:
        window: dict[str, int] = {}
        total = 0
        for j in range(max(0, i - w + 1), i + 1):
            for text in by_sec.get(j, ()):
                window[text] = window.get(text, 0) + 1
                total += 1
        if total >= 3:
            rep[i] = max(window.values()) / total

    win = p.baseline_window_s
    activity = squash(robust_local_z(smooth(msgs, 1.0), win, min_scale=1.0), p.z_scale)
    uniq_s = squash(robust_local_z(smooth(uniq, 1.0), win, min_scale=1.0), p.z_scale)
    msgs_safe = np.maximum(msgs, 1.0)
    hype_s = np.clip(smooth(hype / msgs_safe, 1.0), 0, 1) * activity
    caps_s = smooth(caps / msgs_safe, 1.0) * activity
    rep_s = smooth(rep, 1.0) * activity
    sig = np.clip(0.35 * activity + 0.25 * uniq_s + 0.2 * hype_s + 0.1 * caps_s + 0.1 * rep_s, 0, 1)
    return smooth(sig, p.smoothing_sigma_s), {"chat_msgs": msgs}


# --- bundle ----------------------------------------------------------------------------------


@dataclass
class Features:
    t0: float
    signals: dict[str, np.ndarray]  # name -> 0..1 per second
    fine_db: np.ndarray  # fps frames, for pause detection
    fine_fps: int
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(next(iter(self.signals.values())))

    @property
    def has_chat(self) -> bool:
        return "chat" in self.signals

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"sig_{k}": v for k, v in self.signals.items()}
        arrays.update({f"x_{k}": v for k, v in self.extra.items()})
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, t0=self.t0, fine_db=self.fine_db, fine_fps=self.fine_fps, **arrays)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Features:
        with np.load(path) as z:
            signals = {k[4:]: z[k] for k in z.files if k.startswith("sig_")}
            extra = {k[2:]: z[k] for k in z.files if k.startswith("x_")}
            return cls(float(z["t0"]), signals, z["fine_db"], int(z["fine_fps"]), extra)


def compute_features(
    *,
    blocks,
    sample_rate: int,
    t0: float,
    words: list[tuple[float, float, str]],
    chat: list[ChatMessage] | None,
    lex: HypeLexicon,
    p: FeatureParams,
) -> Features:
    fine = rms_db_frames(blocks, sample_rate, p.fine_fps)
    rms = per_second_db(fine, p.fine_fps)
    n = rms.size
    audio, extra = audio_signal(rms, p)
    speech, keywords, _hits = speech_signals(words, t0, n, lex, p)
    signals = {"audio": audio, "speech": speech, "keywords": keywords}
    extra["rms_db"] = rms
    if chat:
        signals["chat"], chat_extra = chat_signal(chat, t0, n, lex, p)
        extra.update(chat_extra)
    return Features(t0, signals, fine, p.fine_fps, extra)
