"""Step 5B: calibration from review decisions (pure core with synthetic data, CLI, source-page
card, apply/revert of a calibrated preset)."""

import json
from urllib.parse import unquote

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from clipador.cli import main
from clipador.detection.calibrate import (
    Sample,
    auc,
    calibrate,
    format_text,
    precision_at,
    preset_weights,
    stratified_folds,
)
from clipador.detection.params import PRESETS, SIGNALS
from clipador.models import AnalysisRun, Candidate, CandidateEnrichment, Job
from clipador.services import presets
from clipador.web.app import create_app
from tests.review_helpers import seed_review
from tests.test_analysis import asettings, fake_tx  # noqa: F401  (fixtures)

EQ = PRESETS["equilibrado"]


def synthetic(n: int, seed: int = 1, signal: str = "chat", llm: bool = True) -> list[Sample]:
    """Label depends (noisily) on one known signal; the others are noise."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        bd = {k: float(rng.uniform(0, 100)) for k in SIGNALS}
        label = int(bd[signal] + rng.normal(0, 12) > 50)
        nota = (
            float(np.clip(5 + 2.5 * (label - 0.5) * 2 + rng.normal(0, 2), 0, 10)) if llm else None
        )
        out.append(Sample(bd, label, dict(EQ), nota, 0.5, 0.5))
    return out


def test_helpers_auc_precision_folds():
    y = np.array([1, 1, 0, 0])
    assert auc(np.array([4, 3, 2, 1]), y) == 1.0
    assert auc(np.array([1, 2, 3, 4]), y) == 0.0
    assert auc(np.array([1, 1, 1, 1]), y) == 0.5
    assert auc(np.array([1, 2]), np.array([1, 1])) is None
    assert precision_at(np.array([4, 3, 2, 1]), y, 2) == 1.0
    assert precision_at(np.array([4, 3, 2, 1]), y, 4) == 0.5
    y = np.array([1] * 6 + [0] * 9)
    folds = stratified_folds(y, 3, seed=7)
    assert sorted(np.concatenate(folds).tolist()) == list(range(15))
    assert [int(y[f].sum()) for f in folds] == [2, 2, 2]
    assert [f.tolist() for f in folds] == [f.tolist() for f in stratified_folds(y, 3, seed=7)]
    assert preset_weights({"a": 0.5, "b": 0.25, "c": 0.0}) == {"a": 2.0, "b": 1.0, "c": 0.0}


def test_known_signal_gets_the_largest_weight_and_cv_auc_improves():
    cal = calibrate(synthetic(240), SIGNALS, seed=0)
    assert cal.enough_data and cal.can_apply and not cal.warnings
    assert max(cal.weights, key=cal.weights.get) == "chat"
    assert cal.weights["chat"] > 0.6
    assert sum(cal.weights.values()) == pytest.approx(1, abs=1e-3)
    assert all(v >= 0 for v in cal.weights.values())
    assert cal.curve.auc_after > cal.curve.auc_before + 0.05
    assert cal.curve.precision_after[10] >= cal.curve.precision_before[10]
    assert cal.folds == 5
    # detection x AI: the AI nota here is informative too -> a suggestion that sums to 1
    assert cal.blend is not None and cal.ranking is not None
    assert cal.ranking["w_detect"] + cal.ranking["w_llm"] == pytest.approx(1, abs=1e-2)
    text = format_text(cal)
    assert "AUC" in text and "precisão@10" in text and "Não sugiro" not in text


def test_fixed_seed_is_deterministic():
    data = synthetic(120, seed=3)
    a = calibrate(data, SIGNALS, seed=42).to_dict()
    b = calibrate(list(data), SIGNALS, seed=42).to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_few_data_warns_and_does_not_suggest_applying():
    cal = calibrate(synthetic(30), SIGNALS)
    assert not cal.enough_data and not cal.can_apply
    assert "Poucos dados" in cal.warnings[0]
    text = format_text(cal)
    assert "Poucos dados" in text and "Não sugiro aplicar" in text
    # unbalanced: plenty of reviews but < 10 rejected
    data = [s for s in synthetic(300) if s.label == 1][:60] + [
        s for s in synthetic(300) if s.label == 0
    ][:5]
    assert not calibrate(data, SIGNALS).can_apply
    # only one class: no model at all
    only = [s for s in synthetic(80) if s.label == 1]
    cal = calibrate(only, SIGNALS)
    assert cal.weights is None and not cal.can_apply


def test_missing_signal_is_handled():
    data = [Sample({**s.breakdown, "chat": None}, s.label, s.weights) for s in synthetic(120,
            signal="audio", llm=False)]  # fmt: skip
    cal = calibrate(data, SIGNALS)
    assert cal.weights["chat"] == 0 and max(cal.weights, key=cal.weights.get) == "audio"
    assert cal.blend is None and cal.n_llm == 0


# --- DB, CLI, web -------------------------------------------------------------------------------


def _seed_reviewed(engine, source_id: int, run_id: int, n: int, seed: int = 5) -> None:
    rng = np.random.default_rng(seed)
    with Session(engine) as s:
        for i in range(n):
            bd = {k: round(float(rng.uniform(0, 100)), 1) for k in SIGNALS}
            ok = bd["speech"] + rng.normal(0, 10) > 50
            c = Candidate(source_id=source_id, analysis_run_id=run_id, rank=100 + i,
                          start=100 + 40 * i, end=130 + 40 * i, peak_t=110 + 40 * i,
                          score=50, breakdown=bd, status="approved" if ok else "rejected")  # fmt: skip
            s.add(c)
            s.flush()
            if i % 2:
                s.add(CandidateEnrichment(candidate_id=c.id, source_id=source_id,
                                          prompt_version="v1", model="fake", input_hash=str(i),
                                          scores={"nota_geral": 7 if ok else 3}))  # fmt: skip
        s.add(Candidate(source_id=source_id, analysis_run_id=run_id, start=0, end=30, peak_t=5,
                        score=99, breakdown={"speech": 100}, status="skipped"))  # fmt: skip
        s.commit()


@pytest.fixture
def client(settings):
    app = create_app(settings)
    seeded = seed_review(settings, app.state.engine)
    with TestClient(app) as c:
        c.engine, c.settings, c.seeded = app.state.engine, settings, seeded
        c.sid = seeded.sources[0]
        c.run = seeded.runs[c.sid]
        yield c


def test_cli_calibrate_few_data_then_enough(client, monkeypatch, capsys):
    monkeypatch.setattr("clipador.cli.get_settings", lambda: client.settings)
    assert main(["calibrate"]) == 0
    out = capsys.readouterr().out
    assert "0 revisados" in out and "Não sugiro aplicar" in out
    _seed_reviewed(client.engine, client.sid, client.run, 80)
    assert main(["calibrate", "--all", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["n"] == 80 and d["can_apply"] is True  # the skipped one is ignored
    assert max(d["weights"], key=d["weights"].get) == "speech"
    assert d["curve"]["auc_after"] > d["curve"]["auc_before"]
    assert d["n_llm"] == 40 and d["ranking"]["w_llm"] > 0
    assert main(["calibrate", "--source", str(client.seeded.sources[1])]) == 0
    assert "fonte #" in capsys.readouterr().out
    assert main(["calibrate", "--source", "999"]) == 1


def test_card_apply_and_revert_preset(client):
    sid = client.sid
    html = client.get(f"/fontes/{sid}/calibracao").text
    assert "Pesos sugeridos pela calibração" in html and "Poucos dados" in html
    assert "disabled" in html  # apply is off with few data
    r = client.post(f"/fontes/{sid}/calibracao/aplicar", follow_redirects=False)
    assert "Ainda n" in r.headers["location"] or "Ainda%20n" in r.headers["location"]
    assert not presets.store_path(client.settings).exists()

    _seed_reviewed(client.engine, sid, client.run, 80)
    html = client.get(f"/fontes/{sid}/calibracao").text
    assert "Poucos dados" not in html and "Aplicar como novo preset" in html
    assert "Reverter" not in html
    r = client.post(f"/fontes/{sid}/calibracao/aplicar", follow_redirects=False)
    assert r.status_code == 303 and "reanalisando" in r.headers["location"]
    store = presets.load_store(client.settings)
    ((key, preset),) = store["presets"].items()
    assert (
        key.startswith("calibrado_")
        and max(preset["weights"], key=preset["weights"].get) == "speech"
    )
    assert max(preset["weights"].values()) == 2.0 and preset["ranking"]["w_llm"] > 0
    with Session(client.engine) as s:
        job = s.exec(select(Job).where(Job.type == "detect")).one()
        dp = job.payload["detect_params"]
        assert (
            dp["candidates"]["preset"] == key and dp["candidates"]["weights"] == preset["weights"]
        )
        assert dp["ranking"]["w_detect"] == preset["ranking"]["w_detect"]
        assert job.source_id == sid
    # the new preset is offered in the tuning form; the card offers to revert
    page = client.get(f"/fontes/{sid}").text
    assert f'value="{key}"' in page and preset["label"] in page
    assert "Reverter para o preset anterior" in client.get(f"/fontes/{sid}/calibracao").text

    r = client.post(f"/fontes/{sid}/calibracao/reverter", follow_redirects=False)
    assert "Voltando ao preset anterior" in unquote(r.headers["location"])
    with Session(client.engine) as s:
        jobs = s.exec(select(Job).where(Job.type == "detect").order_by(Job.id)).all()
        run = s.get(AnalysisRun, client.run)
        # the detect job had not started: it now carries the previous params (latest wins)
        assert len(jobs) == 1
        prev = jobs[0].payload["detect_params"]
        assert prev["candidates"]["preset"] == run.params["candidates"]["preset"]
        assert prev["candidates"]["weights"] == run.params["candidates"]["weights"]
        assert prev["ranking"] is None
    assert presets.last_applied(client.settings, sid) is None
    assert key in presets.load_store(client.settings)["presets"]  # the preset stays available
    r = client.post(f"/fontes/{sid}/calibracao/reverter", follow_redirects=False)
    assert "Nenhum preset calibrado" in unquote(r.headers["location"])

    # apply while a detect is RUNNING: a new job is queued after it (the running one is kept)
    with Session(client.engine) as s:
        j = s.exec(select(Job).where(Job.type == "detect")).one()
        j.status = "running"
        s.add(j)
        s.commit()
    client.post(f"/fontes/{sid}/calibracao/aplicar")
    with Session(client.engine) as s:
        jobs = s.exec(select(Job).where(Job.type == "detect").order_by(Job.id)).all()
        assert [j.status for j in jobs] == ["running", "queued"]
        assert jobs[1].payload["detect_params"]["candidates"]["preset"].startswith("calibrado_")


def test_reanalyze_with_calibrated_preset_from_tuning_form(client):
    sid = client.sid
    _seed_reviewed(client.engine, sid, client.run, 80)
    client.post(f"/fontes/{sid}/calibracao/aplicar")
    key = next(iter(presets.load_store(client.settings)["presets"]))
    client.post(f"/fontes/{sid}/reanalisar", data={"preset": key})
    with Session(client.engine) as s:
        job = s.exec(select(Job).where(Job.type == "detect").order_by(Job.id.desc())).first()
        assert job.payload["detect_params"]["candidates"]["preset"] == key
        assert job.payload["detect_params"]["ranking"] is not None


def test_detect_run_keeps_calibrated_ranking(asettings, engine, fake_tx):
    """End to end (fake Whisper): a re-analysis with a calibrated preset stores its ranking
    weights in the new run, where RankingParams.from_run picks them up."""
    from clipador.llm.scoring import RankingParams
    from clipador.services import analysis
    from clipador.worker.runner import Worker
    from tests.test_analysis import _analyze, _drain, _source

    sid = _source(engine, asettings)
    _analyze(engine, sid)
    _drain(Worker(asettings, engine))
    with Session(engine) as s:
        first = analysis.runs_for(s, sid)[0]
        first_id, first_params = first.id, dict(first.params)
        p = presets.params_for_preset(asettings, "equilibrado")
        p.candidates.preset = "calibrado_x"
        p.candidates.weights = {"audio": 2.0, "chat": 0.0, "speech": 0.5, "keywords": 0.5}
        p.ranking = {"w_detect": 0.3, "w_llm": 0.7}
        analysis.reanalyze(s, sid, p)
        s.commit()
    _drain(Worker(asettings, engine))
    with Session(engine) as s:
        run = analysis.runs_for(s, sid)[0]
        assert run.id != first_id and run.params["candidates"]["preset"] == "calibrado_x"
        rp = RankingParams.from_run(run.params, asettings)
        assert (rp.w_detect, rp.w_llm) == (0.3, 0.7)
        assert RankingParams.from_run(first_params, asettings).w_llm == asettings.rank_weight_llm
