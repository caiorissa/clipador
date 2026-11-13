"""enrich stage end-to-end with fake transcriber + fake LLM (no GPU, no Ollama)."""

import json

import pytest
from sqlmodel import Session, select

from clipador.llm.scoring import RankingParams
from clipador.models import (
    AnalysisRun,
    ArtifactStatus,
    Candidate,
    CandidateEnrichment,
    PostDraft,
    StageArtifact,
)
from clipador.services import analysis
from clipador.services.enrichment import set_ranking
from clipador.worker.runner import Worker
from tests.conftest import requires_ffmpeg
from tests.fake_llm import FakeLLM, good_answer
from tests.test_analysis import (  # noqa: F401  (fixtures)
    _analyze,
    _drain,
    _source,
    asettings,
    fake_tx,
)

pytestmark = requires_ffmpeg


@pytest.fixture
def analyzed(asettings, engine, fake_tx):  # noqa: F811
    """A source analyzed by the full chain while Ollama is 'offline' (enrich skipped)."""
    asettings.target_clips = 2  # K = 6
    sid = _source(engine, asettings)
    _analyze(engine, sid)
    w = Worker(asettings, engine)
    _drain(w)
    return sid, w


def use_llm(monkeypatch, llm):
    monkeypatch.setattr("clipador.llm.client.make_llm_client", lambda settings: llm)
    return llm


def enrich(engine, worker, sid, **kw):
    with Session(engine) as s:
        analysis.enqueue_enrich(s, sid, **kw)
        s.commit()
    _drain(worker)


def rows(engine, model):
    with Session(engine) as s:
        return s.exec(select(model)).all()


def test_offline_ollama_skips_with_fallback_drafts(analyzed, engine):
    _sid, _w = analyzed
    with Session(engine) as s:
        art = s.exec(select(StageArtifact).where(StageArtifact.stage == "enrich")).one()
        cands = s.exec(select(Candidate)).all()
    assert art.status == ArtifactStatus.SKIPPED and "ollama.com/download" in art.message
    assert rows(engine, CandidateEnrichment) == []
    drafts = rows(engine, PostDraft)
    assert drafts and {d.platform for d in drafts} == {"youtube", "tiktok", "instagram"}
    assert all("Clip de Fulano (Twitch)" in d.description for d in drafts)  # default credit
    assert all(c.final_score == c.score for c in cands)  # detection only


def test_happy_path_scores_drafts_rank_and_unload(analyzed, engine, monkeypatch):
    sid, w = analyzed
    n = len(rows(engine, Candidate))
    llm = use_llm(monkeypatch, FakeLLM(lambda msgs, i: good_answer(nota=9 if i == 1 else 3)))
    enrich(engine, w, sid)
    enr = rows(engine, CandidateEnrichment)
    assert len(enr) == min(n, 6) and all(e.status == "ok" for e in enr)
    assert llm.unloaded == ["qwen2.5:7b"]
    with Session(engine) as s:
        cands = s.exec(select(Candidate).order_by(Candidate.rank)).all()
        art = s.exec(select(StageArtifact).where(StageArtifact.stage == "enrich")).one()
    assert art.status == ArtifactStatus.OK and art.summary["ok"] == len(enr)
    # the candidate the LLM loved (2nd call = detection rank 2) is now first
    assert cands[1].final_rank == 1
    yt = [d for d in rows(engine, PostDraft) if d.platform == "youtube"]
    assert any(d.title == "Fugiu da polícia do jeito mais absurdo" for d in yt)
    assert all("#shorts" in d.hashtags for d in yt)


def test_cache_no_calls_on_rerun_and_invalidation(
    analyzed, engine, monkeypatch, asettings, tmp_path
):
    sid, w = analyzed
    llm = use_llm(monkeypatch, FakeLLM())
    enrich(engine, w, sid)
    first = len(llm.calls)
    assert first > 0
    enrich(engine, w, sid)
    assert len(llm.calls) == first  # unchanged input -> no LLM call

    # new prompt version -> everything recomputed
    pdir = tmp_path / "prompts"
    pdir.mkdir()
    src = asettings.prompts_path / "enrich_v1.md"
    (pdir / "enrich_v1.md").write_text(src.read_text("utf-8") + "\nSeja breve.", "utf-8")
    asettings.prompts_dir = pdir
    enrich(engine, w, sid)
    assert len(llm.calls) == 2 * first

    # another model -> recomputed too
    llm.models.append("llama3.1:8b")
    enrich(engine, w, sid, model="llama3.1:8b")
    assert len(llm.calls) == 3 * first
    assert {e.model for e in rows(engine, CandidateEnrichment)} == {"llama3.1:8b"}


def test_regenerate_with_instruction_and_edited_drafts_protected(analyzed, engine, monkeypatch):
    sid, w = analyzed
    llm = use_llm(monkeypatch, FakeLLM())
    enrich(engine, w, sid)
    with Session(engine) as s:
        top = s.exec(select(Candidate).order_by(Candidate.rank)).first()
        d = s.exec(select(PostDraft).where(PostDraft.candidate_id == top.id,
                                            PostDraft.platform == "youtube")).one()  # fmt: skip
        d.title, d.edited_by_user = "Meu título", True
        s.add(d)
        top_id = top.id
        s.commit()
    before = len(llm.calls)
    llm.answers = [good_answer(titulos=["Mais curto", "b", "c"])]
    enrich(engine, w, sid, candidate_ids=[top_id], instruction="mais curto")
    assert len(llm.calls) == before + 1  # only the targeted candidate
    assert "mais curto" in llm.calls[-1][1]["content"]
    with Session(engine) as s:
        drafts = {
            x.platform: x for x in s.exec(select(PostDraft).where(PostDraft.candidate_id == top_id))
        }
        e = s.exec(
            select(CandidateEnrichment).where(CandidateEnrichment.candidate_id == top_id)
        ).one()
    assert drafts["youtube"].title == "Meu título"  # edited -> kept
    assert drafts["tiktok"].description.startswith("OLHA ISSO")  # not edited -> regenerated
    assert e.instruction == "mais curto" and e.titulos[0] == "Mais curto"

    # a later full run keeps the regenerated version (instruction remembered in the cache key)
    enrich(engine, w, sid)
    assert len(llm.calls) == before + 1

    # explicit overwrite replaces the edited draft
    enrich(engine, w, sid, candidate_ids=[top_id], instruction="outra", overwrite_edited=True)
    with Session(engine) as s:
        yt = s.exec(select(PostDraft).where(PostDraft.candidate_id == top_id,
                                             PostDraft.platform == "youtube")).one()  # fmt: skip
    assert yt.title == "Mais curto" and not yt.edited_by_user


def test_invalid_answers_mark_llm_failed_and_stage_continues(analyzed, engine, monkeypatch):
    sid, w = analyzed

    def answers(msgs, i):
        return "X" if i < 2 else good_answer()  # first candidate: invalid twice

    use_llm(monkeypatch, FakeLLM(answers))
    enrich(engine, w, sid)
    enr = sorted(rows(engine, CandidateEnrichment), key=lambda e: e.id)
    assert enr[0].status == "llm_failed" and "JSON" in enr[0].error
    assert all(e.status == "ok" for e in enr[1:])
    with Session(engine) as s:
        c = s.get(Candidate, enr[0].candidate_id)
    assert c.final_score == c.score  # failed -> detection only


def test_missing_model_is_skipped_with_pull_hint(analyzed, engine, monkeypatch):
    sid, w = analyzed
    use_llm(monkeypatch, FakeLLM(models=("llama3.1:8b",)))
    enrich(engine, w, sid)
    with Session(engine) as s:
        art = s.exec(select(StageArtifact).where(StageArtifact.stage == "enrich")).one()
    assert art.status == ArtifactStatus.SKIPPED and "ollama pull qwen2.5:7b" in art.message


def test_reranking_without_llm_calls(analyzed, engine, monkeypatch, asettings):  # noqa: F811
    sid, w = analyzed
    llm = use_llm(monkeypatch, FakeLLM(lambda msgs, i: good_answer(nota=10 - i)))
    enrich(engine, w, sid)
    calls = len(llm.calls)
    with Session(engine) as s:
        run = s.exec(select(AnalysisRun)).one()
        set_ranking(s, run, RankingParams(w_detect=1, w_llm=0), asettings)
        s.commit()
        cands = s.exec(select(Candidate)).all()
        assert all(c.final_rank == c.rank for c in cands)  # pure detection order
        assert json.loads(json.dumps(run.params))["ranking"]["w_llm"] == 0
    assert len(llm.calls) == calls
