"""From per-second signals to ranked clip candidates (pure functions).

1. combine weighted signals into one interest curve (weights renormalized over the signals
   that exist, e.g. without chat);
2. find peaks;
3. expand each peak: context before, until the curve decays after;
4. enforce min/max duration;
5. snap edges to word boundaries, preferring pauses (never mid-word);
6. non-maximum suppression; score and rank.
All times are absolute seconds of the original video.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from clipador.detection.features import Features
from clipador.detection.params import SIGNALS, CandidateParams

Word = tuple[float, float, str]  # (start, end, text)


@dataclass
class CandidateResult:
    start: float
    end: float
    peak_t: float
    score: float
    breakdown: dict[str, float | None] = field(default_factory=dict)
    snippet: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def effective_weights(weights: dict[str, float], available: set[str]) -> dict[str, float]:
    """Keep only available signals and renormalize to sum 1."""
    w = {k: max(0.0, float(weights.get(k, 0.0))) for k in SIGNALS if k in available}
    total = sum(w.values())
    if total <= 0:
        return {k: 1.0 / len(w) for k in w} if w else {}
    return {k: v / total for k, v in w.items()}


def interest_curve(
    feats: Features, weights: dict[str, float]
) -> tuple[np.ndarray, dict[str, float]]:
    eff = effective_weights(weights, set(feats.signals))
    curve = np.zeros(feats.n)
    for k, wk in eff.items():
        curve += wk * feats.signals[k]
    return np.clip(curve, 0, 1), eff


def pick_peaks(curve: np.ndarray, p: CandidateParams, limit: int) -> list[int]:
    if curve.size == 0:
        return []
    idx, props = find_peaks(curve, height=p.peak_min_height, distance=max(1, p.peak_min_distance_s))
    order = np.argsort(-props["peak_heights"]) if idx.size else []
    return [int(idx[i]) for i in order[:limit]]


def expand_window(curve: np.ndarray, peak: int, p: CandidateParams) -> tuple[float, float]:
    """Window in curve-index seconds (relative)."""
    start = peak - p.context_before_s
    threshold = curve[peak] * p.decay_ratio
    i, below = peak, 0
    limit = min(curve.size - 1, int(peak + p.max_duration_s))
    while i < limit:
        i += 1
        below = below + 1 if curve[i] < threshold else 0
        if below >= 2:
            i -= 1
            break
    end = i + 1 + p.tail_after_s
    return start, end


def best_subwindow(curve: np.ndarray, start: float, end: float, peak: int, length: float) -> float:
    """Start of the `length`-second window inside [start, end] that contains the peak and has
    the highest total interest."""
    lo = max(start, peak + 1 - length)
    hi = min(peak, end - length)
    if hi <= lo:
        return max(start, min(peak - length / 2, end - length))
    cs = np.concatenate([[0.0], np.cumsum(curve)])
    best, best_val = lo, -1.0
    for s in np.arange(lo, hi + 1e-9, 1.0):
        a = int(max(0, np.floor(s)))
        b = int(min(curve.size, np.ceil(s + length)))
        val = cs[b] - cs[a]
        if val > best_val:
            best, best_val = float(s), val
    return best


def enforce_duration(
    curve: np.ndarray, start: float, end: float, peak: int, p: CandidateParams, n: int
) -> tuple[float, float]:
    start, end = max(0.0, start), min(float(n), end)
    dur = end - start
    if dur < p.min_duration_s:
        deficit = p.min_duration_s - dur
        back = min(start, deficit * 0.6)  # prefer more setup/context
        start -= back
        end = min(float(n), end + deficit - back)
        start = max(0.0, end - p.min_duration_s) if end - start < p.min_duration_s else start
    elif dur > p.max_duration_s:
        start = best_subwindow(curve, start, end, peak, p.max_duration_s)
        end = start + p.max_duration_s
    return start, end


# --- edge snapping ---------------------------------------------------------------------------


def _inside_word(t: float, words: list[Word]) -> Word | None:
    for w in words:
        if w[0] < t < w[1]:
            return w
    return None


def _quietest(fine_db: np.ndarray, fps: int, t0: float, a: float, b: float) -> float | None:
    ia, ib = int(max(0, (a - t0) * fps)), int(min(fine_db.size, (b - t0) * fps))
    if ib <= ia:
        return None
    return t0 + (ia + int(np.argmin(fine_db[ia:ib]))) / fps


def snap_edges(
    start: float,
    end: float,
    words: list[Word],
    p: CandidateParams,
    bounds: tuple[float, float],
    fine_db: np.ndarray | None = None,
    fine_fps: int = 10,
) -> tuple[float, float]:
    """Move edges to sentence/word boundaries, preferring pauses >= pause_min_s. Words must be
    sorted. The result never cuts a word in half."""
    lo_b, hi_b = bounds
    near = [
        w
        for w in words
        if w[1] >= start - p.snap_window_s - 5 and w[0] <= end + p.snap_window_s + 5
    ]

    # --- start: a word start preceded by a pause (sentence start), closest, earlier preferred
    best_s = None
    for i, w in enumerate(near):
        gap = w[0] - near[i - 1][1] if i > 0 else p.pause_min_s
        if gap >= p.pause_min_s and start - p.snap_window_s <= w[0] <= start + p.snap_window_s / 2:
            cost = abs(w[0] - start) + (0.5 if w[0] > start else 0.0)
            if best_s is None or cost < best_s[0]:
                best_s = (cost, w[0] - min(0.2, gap / 2))
    if best_s is not None:
        start = best_s[1]
    elif (w := _inside_word(start, near)) is not None:
        start = w[0]
    elif not near and fine_db is not None:
        start = _quietest(fine_db, fine_fps, bounds[0], start - p.snap_window_s / 2, start) or start
    start = max(lo_b, start)

    # --- end: a word end followed by a pause, within duration limits, later preferred
    min_end, max_end = start + p.min_duration_s, start + p.max_duration_s
    best_e = None
    for i, w in enumerate(near):
        gap = near[i + 1][0] - w[1] if i + 1 < len(near) else p.pause_min_s
        if gap >= p.pause_min_s and end - p.snap_window_s / 2 <= w[1] <= end + p.snap_window_s:
            if not (min_end <= w[1] <= max_end):
                continue
            cost = abs(w[1] - end) + (0.5 if w[1] < end else 0.0)
            if best_e is None or cost < best_e[0]:
                best_e = (cost, w[1] + min(0.25, gap / 2))
    if best_e is not None:
        end = best_e[1]
    elif (w := _inside_word(end, near)) is not None:
        # prefer finishing the word; if that breaks max duration, end before it
        end = w[1] if w[1] <= max_end or w[0] <= start else w[0]
    elif not near and fine_db is not None:
        end = _quietest(fine_db, fine_fps, bounds[0], end, end + p.snap_window_s / 2) or end
    end = min(hi_b, end)

    # Final guarantee: no edge inside a word.
    if (w := _inside_word(start, near)) is not None:
        start = w[0]
    if (w := _inside_word(end, near)) is not None:
        end = w[1]
    return start, end


# --- scoring / nms -----------------------------------------------------------------------------


def window_score(x: np.ndarray, a: int, b: int, peak_weight: float) -> float:
    seg = x[max(0, a) : max(a + 1, b)]
    if seg.size == 0:
        return 0.0
    return float(peak_weight * seg.max() + (1 - peak_weight) * seg.mean())


def non_max_suppression(cands: list[CandidateResult], min_gap: float) -> list[CandidateResult]:
    kept: list[CandidateResult] = []
    for c in sorted(cands, key=lambda c: -c.score):
        if all(c.end + min_gap <= k.start or c.start >= k.end + min_gap for k in kept):
            kept.append(c)
    return kept


def detect_candidates(
    feats: Features, words: list[Word], p: CandidateParams
) -> tuple[list[CandidateResult], np.ndarray, dict[str, float]]:
    curve, eff = interest_curve(feats, p.weights)
    t0, n = feats.t0, feats.n
    words = sorted(words)
    results = []
    for peak in pick_peaks(curve, p, limit=max(3 * p.num_candidates, 10)):
        s, e = expand_window(curve, peak, p)
        s, e = enforce_duration(curve, s, e, peak, p, n)
        s_abs, e_abs = snap_edges(
            t0 + s, t0 + e, words, p, (t0, t0 + n), feats.fine_db, feats.fine_fps
        )
        a, b = int(np.floor(s_abs - t0)), int(np.ceil(e_abs - t0))
        breakdown: dict[str, float | None] = {
            k: round(100 * window_score(feats.signals[k], a, b, p.score_peak_weight), 1)
            if k in feats.signals
            else None
            for k in SIGNALS
        }
        snippet = " ".join(w[2] for w in words if s_abs <= w[0] < e_abs)
        if "music" in feats.extra:  # share of the clip with background music (copyright risk)
            m = feats.extra["music"][max(0, a) : max(a + 1, b)]
            breakdown["music"] = round(100 * float(m.mean()), 1) if m.size else 0.0
        results.append(
            CandidateResult(
                start=round(s_abs, 2),
                end=round(e_abs, 2),
                peak_t=round(t0 + peak + 0.5, 2),
                score=round(100 * window_score(curve, a, b, p.score_peak_weight), 1),
                breakdown=breakdown,
                snippet=snippet[:500],
            )
        )
    ranked = non_max_suppression(results, p.min_gap_s)[: p.num_candidates]
    return ranked, curve, eff
