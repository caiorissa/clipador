import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from clipador.models import AnalysisRun, Candidate, Job, PostDraft, Source
from clipador.web.app import create_app
from clipador.worker.runner import Worker
from tests.conftest import requires_ffmpeg
from tests.fake_llm import FakeLLM, good_answer
from tests.test_analysis import _analyze, _drain, _source, asettings, fake_tx  # noqa: F401

pytestmark = requires_ffmpeg

ALERT = {"tipo": "discurso_de_odio_ou_ofensa", "trecho": "frase ofensiva"}


@pytest.fixture
def client(asettings, fake_tx, monkeypatch):
    asettings.target_clips = 1  # K = 3
    app = create_app(asettings)
    engine = app.state.engine
    sid = _source(engine, asettings)
    _analyze(engine, sid)
    w = Worker(asettings, engine)
    _drain(w)  # Ollama "offline" during the chain
    llm = FakeLLM(lambda m, i: good_answer(
        nota=9 - 3 * i, alertas=[ALERT] if i == 0 else [], precisa_de_contexto=(i == 1)
    ))  # fmt: skip
    monkeypatch.setattr("clipador.llm.client.make_llm_client", lambda s: llm)
    with TestClient(app) as c:
        c.engine, c.worker, c.sid, c.llm = engine, w, sid, llm
        yield c


def enrich_now(client):
    r = client.post(f"/fontes/{client.sid}/enriquecer", data={"model": "", "k": ""},
                    follow_redirects=False)  # fmt: skip
    assert r.status_code == 303
    _drain(client.worker)


def test_offline_page_explains_and_still_works(client, monkeypatch):
    from tests.fake_llm import OfflineLLM

    monkeypatch.setattr("clipador.llm.client.make_llm_client", lambda s: OfflineLLM())
    page = client.get(f"/fontes/{client.sid}").text
    assert "IA (Ollama)" in page and "pulado" in page
    assert "Enriquecer com IA" in page and "música de fundo" in page
    assert "rascunho simples" in page  # fallback drafts exist
    status = client.get("/sistema/ollama").text
    assert "fora do ar" in status and "ollama.com/download" in status and "ollama pull" in status


def test_enrich_shows_scores_alerts_drafts(client):
    enrich_now(client)
    assert client.llm.calls and client.llm.unloaded == ["qwen2.5:7b"]
    page = client.get(f"/fontes/{client.sid}").text
    assert "Reação forte a uma perseguição" in page  # motivo
    assert "sev-high" in page and "frase ofensiva" in page
    assert "precisa de contexto" in page
    assert "Pacote de postagem (IA)" in page and "YouTube Shorts" in page and "TikTok" in page
    assert "Fugiu da polícia do jeito mais absurdo" in page  # title option
    assert "IA local ok" in client.get("/sistema/ollama").text


def test_sorting_and_reranking_without_llm_calls(client):
    enrich_now(client)
    calls = len(client.llm.calls)
    for sort in ("final", "detect", "ia", "time"):
        assert client.get(f"/fontes/{client.sid}?sort={sort}").status_code == 200
    r = client.post(f"/fontes/{client.sid}/ordenacao",
                    data={"w_detect": "1", "w_llm": "0", "penalty_needs_context": "0"},
                    follow_redirects=False)  # fmt: skip
    assert r.status_code == 303
    with Session(client.engine) as s:
        run = s.exec(select(AnalysisRun)).one()
        assert run.params["ranking"]["w_llm"] == 0
        assert all(c.final_rank == c.rank for c in s.exec(select(Candidate)).all())
    r = client.post(f"/fontes/{client.sid}/ordenacao", data={"w_detect": "0", "w_llm": "0"})
    assert "Pelo menos um dos pesos" in r.text
    assert len(client.llm.calls) == calls


def test_draft_autosave_marks_edited(client):
    enrich_now(client)
    with Session(client.engine) as s:
        d = s.exec(select(PostDraft).where(PostDraft.platform == "youtube")).first()
    r = client.post(f"/rascunhos/{d.id}",
                    data={"title": "Meu <título>\n", "description": "desc", "hashtags": "#GTA 6, #fuga #fuga"})  # fmt: skip
    assert r.status_code == 200 and "salvo" in r.text
    with Session(client.engine) as s:
        d2 = s.get(PostDraft, d.id)
    assert d2.edited_by_user and d2.title == "Meu título"
    assert d2.hashtags == ["#GTA", "#fuga"]  # "6" alone is not a hashtag; duplicate removed
    assert client.post("/rascunhos/9999", data={"title": "x"}).status_code == 404


def test_regenerate_and_game(client):
    enrich_now(client)
    with Session(client.engine) as s:
        cid = s.exec(select(Candidate).order_by(Candidate.rank)).first().id
    r = client.post(f"/candidatos/{cid}/regerar", data={"instruction": "mais curto"})
    assert r.status_code == 200 and "Regerando" in r.text
    with Session(client.engine) as s:
        job = s.exec(select(Job).where(Job.type == "enrich").order_by(Job.id.desc())).first()
    assert job.payload["candidate_ids"] == [cid] and job.payload["instruction"] == "mais curto"
    _drain(client.worker)
    assert "mais curto" in client.llm.calls[-1][1]["content"]

    client.post(f"/fontes/{client.sid}/jogo", data={"game": "  GTA   6 RP "})
    with Session(client.engine) as s:
        assert s.get(Source, client.sid).game == "GTA 6 RP"
