"""llm-bench, eval --with-llm and doctor checks with fake LLM clients."""

import pytest
from sqlmodel import Session, select

from clipador.cli import main
from clipador.models import Candidate, CandidateFeedback
from clipador.services import analysis, doctor, llmbench
from clipador.services.calibration import pairwise_agreement, run_eval
from clipador.services.doctor import CheckStatus
from clipador.worker.runner import Worker
from tests.conftest import requires_ffmpeg
from tests.fake_llm import TIMEOUT, FakeLLM, good_answer
from tests.test_analysis import BURSTS, _analyze, _drain, _source, asettings, fake_tx  # noqa: F401


def use(monkeypatch, llm):
    monkeypatch.setattr("clipador.llm.client.make_llm_client", lambda s: llm)
    return llm


def test_pairwise_agreement():
    assert pairwise_agreement([(9, "bom"), (2, "ruim")]) == (1.0, 1)
    assert pairwise_agreement([(2, "bom"), (9, "ruim")]) == (0.0, 1)
    assert pairwise_agreement([(5, "bom"), (5, "ruim"), (7, "bom")]) == (0.75, 2)
    assert pairwise_agreement([(5, "bom")]) == (None, 0)


def test_llm_bench_with_samples(settings, monkeypatch, capsys):
    llm = use(monkeypatch, FakeLLM(["{ruim", good_answer(), TIMEOUT, good_answer(), good_answer()],
                                   models=("qwen2.5:7b",)))  # fmt: skip
    monkeypatch.setattr("clipador.cli.get_settings", lambda: settings)
    monkeypatch.setattr("clipador.services.llmbench.vram_used_mb", lambda: None)
    monkeypatch.setattr("clipador.services.llmbench.time.sleep", lambda s: None)
    assert main(["llm-bench", "--n", "4", "--models", "qwen2.5:7b,llama3.1:8b"]) == 0
    out = capsys.readouterr().out
    assert "qwen2.5:7b" in out and "25%" in out  # 1 invalid JSON out of 4 candidates
    assert "llama3.1:8b" in out and "ollama pull llama3.1:8b" in out
    assert llm.unloaded == ["qwen2.5:7b"]
    # the built-in samples include a prompt-injection attempt, wrapped in delimiters
    users = [m[1]["content"] for m in llm.calls]
    assert any("ignore as instruções anteriores" in u for u in users)


def test_llm_bench_offline(settings, monkeypatch, capsys):
    monkeypatch.setattr("clipador.cli.get_settings", lambda: settings)
    assert main(["llm-bench", "--n", "1"]) == 1
    assert "ollama.com/download" in capsys.readouterr().err


def test_bench_contexts_from_source_require_analysis(settings, engine):
    with pytest.raises(ValueError, match="não existe"):
        llmbench.contexts_from_source(engine, settings, 99, 3)


@requires_ffmpeg
def test_eval_with_llm_shows_comparison_and_agreement(
    asettings, engine, fake_tx, monkeypatch, tmp_path
):
    asettings.target_clips = 2
    sid = _source(engine, asettings)
    _analyze(engine, sid)
    w = Worker(asettings, engine)
    _drain(w)
    # the LLM loves the candidate that does NOT contain a golden moment -> ranking should get worse
    with Session(engine) as s:
        cands = s.exec(select(Candidate).order_by(Candidate.rank)).all()
        golden_ids = {c.id for c in cands if any(c.start <= t + 1 <= c.end for t in BURSTS)}
        loser = next(c for c in cands if c.id not in golden_ids)
        loser_id, top_id = loser.id, cands[0].id

    def answers(msgs, i):
        return good_answer(nota=10 if i == [c.id for c in cands].index(loser_id) else 1)

    use(monkeypatch, FakeLLM(answers))
    with Session(engine) as s:
        analysis.enqueue_enrich(s, sid)
        s.add(CandidateFeedback(candidate_id=top_id, verdict="bom"))
        s.add(CandidateFeedback(candidate_id=loser_id, verdict="ruim"))
        s.commit()
    _drain(w)
    g = tmp_path / "golden.txt"
    g.write_text("\n".join(f"00:{int(t // 60):02d}:{int(t % 60) + 1:02d}" for t in BURSTS), "utf-8")
    _res, text = run_eval(engine, asettings, sid, g, ns=(1, 5), with_llm=True)
    assert "Com a IA" in text
    assert "recall@1   detecção 1/2 → com IA 0/2 (PIOROU)" in text
    assert "IA 0% · detecção 100%" in text  # IA disagrees with the verdicts, detection agrees


def test_doctor_ollama_checks(settings, monkeypatch):
    # offline (default fake): one warning
    (r,) = doctor.check_ollama(settings)
    assert r.status == CheckStatus.WARN and "ollama.com" in r.hint
    # model missing
    use(monkeypatch, FakeLLM(models=("llama3.1:8b",)))
    r1, r2 = doctor.check_ollama(settings)
    assert r1.status == CheckStatus.OK and "llama3.1:8b" in r1.detail
    assert r2.status == CheckStatus.WARN and "ollama pull qwen2.5:7b" in r2.hint
    # installed: test generation latency
    use(monkeypatch, FakeLLM(["ok"]))
    r1, r2 = doctor.check_ollama(settings)
    assert r2.status == CheckStatus.OK and "1.2s" in r2.detail and "42 tokens/s" in r2.detail
