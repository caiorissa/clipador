"""DB side of step 3: final ranking, post drafts, lookups for the UI."""

from __future__ import annotations

from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, col, select

from clipador.config import Settings
from clipador.llm.drafts import DRAFT_PLATFORMS, DraftData
from clipador.llm.scoring import RankingParams, ScoreParts, final_score, rank_order
from clipador.models import AnalysisRun, Candidate, CandidateEnrichment, PostDraft
from clipador.utils import utcnow


def enrichments_for(session: Session, candidate_ids: list[int]) -> dict[int, CandidateEnrichment]:
    if not candidate_ids:
        return {}
    rows = session.exec(
        select(CandidateEnrichment).where(col(CandidateEnrichment.candidate_id).in_(candidate_ids))
    ).all()
    return {r.candidate_id: r for r in rows}


def drafts_for(session: Session, candidate_ids: list[int]) -> dict[int, dict[str, PostDraft]]:
    out: dict[int, dict[str, PostDraft]] = {}
    if not candidate_ids:
        return out
    for d in session.exec(
        select(PostDraft).where(col(PostDraft.candidate_id).in_(candidate_ids))
    ).all():
        out.setdefault(d.candidate_id, {})[d.platform] = d
    return out


def score_parts(c: Candidate, e: CandidateEnrichment | None, p: RankingParams) -> ScoreParts:
    nota = e.nota_geral if e else None
    standalone = e.scores.get("faz_sentido_sozinho") if e and nota is not None else None
    return final_score(c.score, nota, bool(e and e.precisa_de_contexto), standalone, p)


def rerank_run(session: Session, run: AnalysisRun, settings: Settings) -> RankingParams:
    """Recompute final_score/final_rank for every candidate of the run (no LLM calls)."""
    p = RankingParams.from_run(run.params, settings)
    cands = session.exec(select(Candidate).where(Candidate.analysis_run_id == run.id)).all()
    enr = enrichments_for(session, [c.id for c in cands])
    items = []
    for c in cands:
        parts = score_parts(c, enr.get(c.id), p)
        c.final_score = parts.final
        items.append((c.id, parts.final, c.score))
    ranks = rank_order(items)
    for c in cands:
        c.final_rank = ranks[c.id]
        session.add(c)
    return p


def set_ranking(session: Session, run: AnalysisRun, p: RankingParams, settings: Settings) -> None:
    run.params = {**run.params, "ranking": p.model_dump()}
    flag_modified(run, "params")
    session.add(run)
    rerank_run(session, run, settings)


def save_drafts(
    session: Session, candidate_id: int, drafts: dict[str, DraftData], overwrite_edited: bool
) -> list[str]:
    """Upsert drafts; drafts edited by the user are kept unless overwrite_edited. Returns the
    platforms that were kept because they were edited."""
    existing = drafts_for(session, [candidate_id]).get(candidate_id, {})
    kept = []
    for platform in DRAFT_PLATFORMS:
        data = drafts[platform]
        row = existing.get(platform)
        if row is not None and row.edited_by_user and not overwrite_edited:
            kept.append(platform)
            continue
        if row is None:
            row = PostDraft(candidate_id=candidate_id, platform=platform)
        row.title, row.description, row.hashtags = data.title, data.description, data.hashtags
        row.edited_by_user = False
        row.updated_at = utcnow()
        session.add(row)
    return kept
