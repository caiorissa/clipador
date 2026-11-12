"""Calibration metrics against hand-marked "golden" moments (pure functions)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from clipador.utils import parse_timecode


def parse_golden(text: str) -> list[float]:
    """One timestamp per line (HH:MM:SS, MM:SS or seconds). '#' starts a comment."""
    out = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            t = parse_timecode(line.split()[0])
        except ValueError as e:
            raise ValueError(f"linha {n}: {e}") from None
        if t is not None:
            out.append(t)
    return sorted(out)


@dataclass
class RankedCandidate:
    rank: int  # 1-based
    start: float
    end: float
    score: float
    verdict: str | None = None  # "bom" | "ruim" | None


@dataclass
class Missed:
    t: float
    curve_score: float | None  # interest at that instant (0-100)
    nearest_rank: int | None  # rank of the closest candidate in the full list
    nearest_distance: float | None


@dataclass
class EvalResult:
    total: int
    recall: dict[int, tuple[int, int]] = field(default_factory=dict)  # N -> (hits, total)
    precision: dict[int, tuple[int, int] | None] = field(default_factory=dict)  # N -> (rel, judged)
    missed: list[Missed] = field(default_factory=list)  # w.r.t. the largest N


def _contains(c: RankedCandidate, t: float, tol: float) -> bool:
    return c.start - tol <= t <= c.end + tol


def evaluate(
    golden: Sequence[float],
    ranked: Sequence[RankedCandidate],
    *,
    ns: Sequence[int] = (5, 10, 20),
    tolerance: float = 2.0,
    curve_at: Callable[[float], float | None] | None = None,
) -> EvalResult:
    ranked = sorted(ranked, key=lambda c: c.rank)
    res = EvalResult(total=len(golden))
    for n in ns:
        top = ranked[:n]
        hits = sum(any(_contains(c, t, tolerance) for c in top) for t in golden)
        res.recall[n] = (hits, len(golden))
        relevant = judged = 0
        for c in top:
            has_golden = any(_contains(c, t, tolerance) for t in golden)
            if has_golden or c.verdict == "bom":
                relevant += 1
                judged += 1
            elif c.verdict == "ruim":
                judged += 1
        res.precision[n] = (relevant, judged) if judged else None
    top_max = ranked[: max(ns)] if ns else ranked
    for t in golden:
        if any(_contains(c, t, tolerance) for c in top_max):
            continue

        def distance(c: RankedCandidate, t: float = t) -> float:
            return 0.0 if _contains(c, t, 0) else min(abs(c.start - t), abs(c.end - t))

        nearest = min(ranked, key=distance, default=None)
        dist = distance(nearest) if nearest is not None else None
        res.missed.append(
            Missed(t, curve_at(t) if curve_at else None, nearest.rank if nearest else None, dist)
        )
    return res
