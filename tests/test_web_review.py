"""/revisao routes with TestClient (no ffmpeg: renders are queued, never executed)."""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from clipador.models import Candidate, CandidateFeedback, Decision, Job, Render
from clipador.services.artifacts import save_artifact
from clipador.web.app import create_app
from tests.review_helpers import seed_review, write_wav


@pytest.fixture
def client(settings):
    app = create_app(settings)
    seeded = seed_review(settings, app.state.engine)
    with TestClient(app) as c:
        c.engine, c.settings, c.seeded = app.state.engine, settings, seeded
        s1, s2 = seeded.sources
        c.a, c.b = seeded.cands[s1], seeded.cands[s2]
        yield c


def _get(client, cid) -> Candidate:
    with Session(client.engine) as s:
        return s.get(Candidate, cid)


def test_review_page_shows_first_candidate_and_shortcuts(client):
    r = client.get("/revisao")
    assert r.status_code == 200
    html = r.text
    b1 = client.b[0]
    assert f'data-candidate="{b1}"' in html and "1 de 5" in html
    for key in ("a", "r", "s", "z", "p", "l", "space"):
        assert f'data-shortcut="{key}"' in html
    assert 'data-reason="risco_copyright_musica"' in html and "/static/review.js" in html
    assert "APROXIMADA" in html and "prévia fiel" in html.lower()
    assert 'id="review-data"' in html
    js = client.get("/static/review.js")
    assert js.status_code == 200 and "keydown" in js.text and "palavras.json" in js.text
    # the next pending ones got a low-priority faithful preview queued (prefetch)
    with Session(client.engine) as s:
        jobs = s.exec(select(Job).where(Job.type == "render")).all()
        assert jobs and all(j.priority < 0 for j in jobs)
        assert b1 not in {r.candidate_id for r in s.exec(select(Render)).all()}
    n = len(jobs)
    client.get("/revisao")  # opening again does not queue them twice
    with Session(client.engine) as s:
        assert len(s.exec(select(Job).where(Job.type == "render")).all()) == n


def test_specific_candidate_and_alerts(client):
    a1 = client.a[0]
    html = client.get(f"/revisao?c={a1}").text
    assert f'data-candidate="{a1}"' in html
    assert "perseguição engraçada" in html and "p*rra" in html and "ELE FUGIU" in html


def test_approve_reject_skip_undo_flow(client):
    b1, a1, a2 = client.b[0], client.a[0], client.a[1]
    r = client.post(f"/revisao/{b1}/decisao", data={"action": "approve"}, follow_redirects=False)
    assert r.status_code == 303 and f"after={b1}" in r.headers["location"]
    page = client.get(r.headers["location"]).text
    assert f'data-candidate="{a1}"' in page  # moved on to the next one
    assert _get(client, b1).status == "approved"

    client.post(f"/revisao/{a1}/decisao", data={"action": "reject", "reason_tag": "audio_ruim",
                                               "note": "chiado"})  # fmt: skip
    client.post(f"/revisao/{a2}/decisao", data={"action": "skip"})
    with Session(client.engine) as s:
        fb = s.exec(select(CandidateFeedback).where(CandidateFeedback.candidate_id == a1)).one()
        assert (fb.verdict, fb.reason_tag, fb.note) == ("ruim", "audio_ruim", "chiado")
    assert _get(client, a2).status == "skipped"

    r = client.post("/revisao/desfazer", follow_redirects=False)
    assert f"c={a2}" in r.headers["location"]
    client.post("/revisao/desfazer")
    client.post("/revisao/desfazer")
    assert [_get(client, i).status for i in (b1, a1, a2)] == ["suggested"] * 3
    r = client.post("/revisao/desfazer", follow_redirects=False)
    assert "Nada" in r.headers["location"] or "Nada%20para" in r.headers["location"]

    # invalid transition is explained, nothing changes
    client.post(f"/revisao/{b1}/decisao", data={"action": "approve"})
    r = client.post(f"/revisao/{b1}/decisao", data={"action": "skip"}, follow_redirects=False)
    assert f"c={b1}" in r.headers["location"]
    assert _get(client, b1).status == "approved"


def test_filters_are_kept_across_decisions(client):
    s1 = client.seeded.sources[0]
    a1, a2 = client.a[0], client.a[1]
    html = client.get(f"/revisao?fonte={s1}").text
    assert f'data-candidate="{a1}"' in html and "1 de 3" in html
    r = client.post(f"/revisao/{a1}/decisao", data={"action": "skip", "fonte": str(s1)},
                    follow_redirects=False)  # fmt: skip
    loc = r.headers["location"]
    assert f"fonte={s1}" in loc
    assert f'data-candidate="{a2}"' in client.get(loc).text


def test_empty_queue(client):
    html = client.get("/revisao?nota_min=99").text
    assert "Nada para revisar" in html


def test_status_route_records_decision_and_renders_on_approve(client):
    a1 = client.a[0]
    r = client.post(f"/candidatos/{a1}/status", data={"status": "approved"})
    assert r.status_code == 200 and "aprovado" in r.text and "render final na fila" in r.text
    r = client.post(f"/candidatos/{a1}/status", data={"status": "approved"})  # toggle off
    assert _get(client, a1).status == "suggested"
    with Session(client.engine) as s:
        assert [d.action for d in s.exec(select(Decision).order_by(Decision.id))] == [
            "approve",
            "reopen",
        ]
        assert len(s.exec(select(Job).where(Job.type == "render")).all()) == 1


def test_words_json_with_overrides_and_global_index(client):
    a1 = client.a[0]
    with Session(client.engine) as s:
        c = s.get(Candidate, a1)
        c.caption_overrides = {"2": "VIATURA!", "4": ""}
        s.add(c)
        s.commit()
    d = client.get(f"/candidatos/{a1}/palavras.json").json()
    assert (d["start"], d["end"]) == (8.0, 22.0)
    assert [w["i"] for w in d["words"]] == [0, 1, 2, 3, 5, 6]  # #4 hidden
    assert d["words"][2] == {"i": 2, "s": 10.9, "e": 11.6, "t": "VIATURA!"}
    narrow = client.get(f"/candidatos/{a1}/palavras.json?start=12.9&end=14").json()
    assert [w["i"] for w in narrow["words"]] == [3]
    assert narrow["words"][0]["e"] == 13.5
    assert d["style"] == "destaque_amarelo"


def test_waveform_json(client):
    s1 = client.seeded.sources[0]
    path = write_wav(client.settings.work_dir / str(s1) / "audio.wav", 40, [15.0])
    save_artifact(client.engine, s1, "extract_audio", "aud", files={
        "audio": client.settings.rel_path(path)}, summary={"offset": 0.0})  # fmt: skip
    d = client.get(f"/fontes/{s1}/forma-de-onda.json?start=10&end=20").json()
    assert d["t0"] == 0 and len(d["peaks"]) == 40 * 20
    assert max(range(len(d["peaks"])), key=d["peaks"].__getitem__) == 300  # 15 s
    assert client.get("/fontes/999/forma-de-onda.json?start=0&end=1").status_code == 404
    assert client.get(f"/fontes/{s1}/forma-de-onda.json?start=5&end=1").status_code == 400


def test_range_endpoint_validates_snaps_and_warns(client):
    a1, a2 = client.a[0], client.a[1]
    r = client.post(f"/revisao/{a1}/intervalo", data={"start": "9,5", "end": "21"})
    assert r.json() == {"ok": True, "start": 9.5, "end": 21.0, "warnings": [], "adjusted": True}
    r = client.post(f"/revisao/{a1}/intervalo", data={"start": "10.1", "end": "21.1", "snap": "1"})
    assert (r.json()["start"], r.json()["end"]) == (10.0, 21.2)
    r = client.post(f"/revisao/{a1}/intervalo", data={"start": "30", "end": "10"})
    assert r.status_code == 422 and "O fim precisa" in r.json()["error"]
    r = client.post(f"/revisao/{a1}/intervalo", data={"start": "0", "end": "400"})
    assert "máx. 3 min" in r.json()["error"]
    client.post(f"/revisao/{a1}/decisao", data={"action": "approve"})
    r = client.post(f"/revisao/{a2}/intervalo", data={"start": "20", "end": "45"})
    assert r.json()["ok"] and "Sobrepõe" in r.json()["warnings"][0]
    r = client.post(f"/revisao/{a2}/intervalo", data={"reset": "1"})
    assert r.json()["adjusted"] is False and _get(client, a2).user_start is None


def test_legacy_adjust_route_uses_same_validation(client):
    a1 = client.a[0]
    r = client.post(f"/candidatos/{a1}/ajuste", data={"user_start": "9", "user_end": "10"})
    assert "Trecho curto demais" in r.text
    r = client.post(f"/candidatos/{a1}/ajuste", data={"user_start": "9", "user_end": "21"})
    assert "ajustado" in r.text


def test_layout_cycle_and_faithful_preview(client):
    a1, b1 = client.a[0], client.b[0]
    r = client.post(f"/revisao/{a1}/layout", data={"layout": "proximo"}).json()
    assert r["layout"] == "facecam_embaixo"  # streamer A has a facecam
    assert client.post(f"/revisao/{a1}/layout").json()["layout"] == "jogo_cheio"
    assert client.post(f"/revisao/{a1}/layout").json()["layout"] == "blur"
    # streamer B has no facecam: the cycle skips it
    assert client.post(f"/revisao/{b1}/layout").json()["layout"] == "jogo_cheio"

    assert client.get(f"/revisao/{a1}/previa.json").json()["status"] == "nenhuma"
    d = client.post(f"/revisao/{a1}/previa").json()
    assert d["status"] == "queued"
    with Session(client.engine) as s:
        r = s.exec(select(Render).where(Render.candidate_id == a1)).one()
        assert s.get(Job, r.job_id).priority == 0
    client.post(f"/revisao/{a1}/previa")
    with Session(client.engine) as s:
        assert len(s.exec(select(Render).where(Render.candidate_id == a1)).all()) == 1
    # changing the cut makes that preview stale
    client.post(f"/revisao/{a1}/intervalo", data={"start": "9", "end": "21"})
    assert client.get(f"/revisao/{a1}/previa.json").json()["status"] == "nenhuma"


def test_caption_font_is_served(client):
    r = client.get("/revisao/fonte.ttf")
    assert r.status_code == 200 and len(r.content) > 10000
