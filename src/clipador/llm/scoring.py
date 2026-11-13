"""Final score = detection (0-100) combined with the LLM's nota_geral (0-10 -> 0-100)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from clipador.config import Settings


class RankingParams(BaseModel):
    w_detect: float = 0.5
    w_llm: float = 0.5
    penalty_needs_context: float = 8.0  # points (0-100 scale)
    penalty_low_standalone: float = 6.0
    low_standalone_threshold: float = 4.0  # faz_sentido_sozinho below this is penalized

    @classmethod
    def from_settings(cls, s: Settings) -> RankingParams:
        return cls(
            w_detect=s.rank_weight_detect,
            w_llm=s.rank_weight_llm,
            penalty_needs_context=s.rank_penalty_needs_context,
            penalty_low_standalone=s.rank_penalty_low_standalone,
            low_standalone_threshold=s.rank_low_standalone_threshold,
        )

    @classmethod
    def from_run(cls, run_params: dict[str, Any], s: Settings) -> RankingParams:
        data = run_params.get("ranking")
        return cls.model_validate(data) if data else cls.from_settings(s)


@dataclass
class ScoreParts:
    detect: float
    llm: float | None  # 0-100 or None (no/failed enrichment)
    penalty: float
    final: float


def final_score(
    detect_score: float,
    llm_nota: float | None,
    needs_context: bool,
    standalone: float | None,
    p: RankingParams,
) -> ScoreParts:
    """Without an LLM score the weights renormalize to detection only (no penalties)."""
    d = max(0.0, min(100.0, detect_score))
    if llm_nota is None or (p.w_detect + p.w_llm) <= 0 or p.w_llm <= 0:
        return ScoreParts(d, None if llm_nota is None else llm_nota * 10, 0.0, round(d, 1))
    llm = max(0.0, min(10.0, llm_nota)) * 10
    base = (p.w_detect * d + p.w_llm * llm) / (p.w_detect + p.w_llm)
    penalty = 0.0
    if needs_context:
        penalty += p.penalty_needs_context
    if standalone is not None and standalone < p.low_standalone_threshold:
        penalty += p.penalty_low_standalone
    return ScoreParts(d, llm, penalty, round(max(0.0, base - penalty), 1))


def rank_order(items: list[tuple[int, float, float]]) -> dict[int, int]:
    """items: (candidate_id, final_score, detection_score) -> candidate_id: rank (1-based).
    Ties keep the detection order."""
    ordered = sorted(items, key=lambda x: (-x[1], -x[2], x[0]))
    return {cid: i for i, (cid, _f, _d) in enumerate(ordered, 1)}
