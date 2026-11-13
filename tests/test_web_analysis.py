import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from clipador.models import AnalysisRun, Candidate, CandidateFeedback, Job
from clipador.web.app import create_app
from clipador.worker.runner import Worker
from tests.conftest import requires_ffmpeg
from tests.test_analysis import (  # noqa: F401  (fixtures)
    _analyze,
    _drain,
    _source,
    asettings,
    fake_tx,
)

pytestmark = requires_ffmpeg


@pytest.fixture
def analyzed(asettings, fake_tx):
    app = create_app(asettings)
    engine = app.state.engine
    sid = _source(engine, asettings)
    _analyze(engine, sid)
    _drain(Worker(asettings, engine))
    with TestClient(app) as c:
        c.engine = engine
        c.worker = Worker(asettings, engine)
        c.sid = sid
        yield c


def _first_candidate(client) -> Candidate:
    with Session(client.engine) as s:
        return s.exec(select(Candidate).order_by(Candidate.rank)).first()


def test_detail_page_shows_stages_candidates_and_chat_warning(analyzed):
    r = analyzed.get(f"/fontes/{analyzed.sid}")
    assert r.status_code == 200
    html = r.text
    for label in ("Extrair áudio", "Transcrever", "Chat", "Detectar"):
        assert label in html
    assert "pulado" in html and "TwitchDownloaderCLI" in html
    assert "Sem dados de chat: nota calculada só com áudio e fala." in html
    assert 'class="cand ' in html and "#t=" in html and "Reanalisar" in html
    stages = analyzed.get(f"/fontes/{analyzed.sid}/estagios").text
    assert 'data-latest-run="1"' in stages and 'data-active="0"' in stages


def test_timeline_json(analyzed):
    d = analyzed.get(f"/fontes/{analyzed.sid}/timeline.json").json()
    assert len(d["t"]) == len(d["interest"]) == len(d["audio"]) > 100
    assert "chat" not in d
    assert d["candidates"] and {"start", "end", "peak", "score", "rank"} <= set(d["candidates"][0])


def test_video_supports_range_requests(analyzed):
    full = analyzed.get(f"/fontes/{analyzed.sid}/video")
    assert full.status_code == 200 and full.headers.get("accept-ranges") == "bytes"
    part = analyzed.get(f"/fontes/{analyzed.sid}/video", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206
    assert part.headers["content-range"].startswith("bytes 100-199/")
    assert part.content == full.content[100:200]


def test_feedback_status_and_adjust(analyzed):
    c = _first_candidate(analyzed)
    r = analyzed.post(f"/candidatos/{c.id}/feedback", data={"verdict": "bom"})
    assert r.status_code == 200 and f'id="cand-{c.id}"' in r.text and ">bom<" in r.text
    analyzed.post(f"/candidatos/{c.id}/feedback", data={"verdict": "ruim"})
    with Session(analyzed.engine) as s:
        assert s.exec(select(CandidateFeedback)).one().verdict == "ruim"
    analyzed.post(f"/candidatos/{c.id}/feedback", data={"verdict": "ruim"})  # toggle off
    with Session(analyzed.engine) as s:
        assert s.exec(select(CandidateFeedback)).first() is None

    r = analyzed.post(f"/candidatos/{c.id}/status", data={"status": "approved"})
    assert "aprovado" in r.text
    r = analyzed.post(
        f"/candidatos/{c.id}/ajuste", data={"user_start": c.start - 2, "user_end": c.end + 1}
    )
    assert "ajustado" in r.text
    with Session(analyzed.engine) as s:
        got = s.get(Candidate, c.id)
        assert got.status == "approved" and got.user_start == pytest.approx(c.start - 2)
    r = analyzed.post(f"/candidatos/{c.id}/ajuste", data={"user_start": 50, "user_end": 40})
    assert "O fim precisa ser depois do início" in r.text
    analyzed.post(f"/candidatos/{c.id}/ajuste", data={"reset": "1"})
    with Session(analyzed.engine) as s:
        assert s.get(Candidate, c.id).user_start is None


def test_reanalyze_with_custom_weights_creates_new_run(analyzed):
    r = analyzed.post(
        f"/fontes/{analyzed.sid}/reanalisar",
        data={"preset": "foco_fala", "custom_weights": "on", "w_audio": "0.2", "w_chat": "1",
              "w_speech": "2", "w_keywords": "3", "min_duration_s": "15", "max_duration_s": "30",
              "num_candidates": "5"},
        follow_redirects=False,
    )  # fmt: skip
    assert r.status_code == 303
    _drain(analyzed.worker)
    with Session(analyzed.engine) as s:
        runs = s.exec(select(AnalysisRun).order_by(AnalysisRun.id)).all()
        assert len(runs) == 2
        cp = runs[1].params["candidates"]
        assert cp["weights"]["keywords"] == 3 and cp["max_duration_s"] == 30
        assert "chat" not in runs[1].params["effective_weights"]
        cands = s.exec(select(Candidate).where(Candidate.analysis_run_id == runs[1].id)).all()
        assert 0 < len(cands) <= 5 and all(c.end - c.start <= 31 for c in cands)
    page = analyzed.get(f"/fontes/{analyzed.sid}").text
    assert "Análise #2" in page
    assert analyzed.get(f"/fontes/{analyzed.sid}?run=1").status_code == 200


def test_reanalyze_rejects_bad_durations(analyzed):
    r = analyzed.post(
        f"/fontes/{analyzed.sid}/reanalisar",
        data={"preset": "equilibrado", "min_duration_s": "40", "max_duration_s": "20"},
    )
    assert "duração máxima precisa ser maior" in r.text
    with Session(analyzed.engine) as s:
        assert s.exec(select(Job).where(Job.type == "detect")).all()[-1].status == "done"


def test_analyze_button_and_unknown_source(analyzed):
    r = analyzed.post(f"/fontes/{analyzed.sid}/analisar", follow_redirects=False)
    assert r.status_code == 303
    assert analyzed.get("/fontes/999").status_code == 404
    assert analyzed.get("/fontes/tabela").status_code == 200  # static route still wins
