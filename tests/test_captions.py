"""Step 5B: caption editor (resolve_overrides), streamer glossary (apply_glossary) and their
routes. No ffmpeg: renders are only planned."""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from clipador.models import Candidate, Streamer
from clipador.render.subtitles import (
    IWord,
    StaleOverride,
    apply_glossary,
    caption_words,
    norm_word,
    resolve_overrides,
)
from clipador.services import rendering
from clipador.services.artifacts import save_artifact
from clipador.services.transcriber import Word, save_transcript
from clipador.web.app import create_app
from tests.review_helpers import WORDS, seed_review


def W(i, text, s=0.0):
    return IWord(i, s, s + 0.4, text)


# --- resolve_overrides -------------------------------------------------------------------------


def test_resolve_overrides_valid_stale_window_and_legacy():
    words = [W(0, "olha"), W(1, "a"), W(2, "polícia!"), W(3, "corre")]
    overrides = {"1": "", "2": "VIATURA!", "3": "CORRE", "9": "fora"}
    originals = {"1": "a", "2": "polícia!", "3": "corra"}  # #3 was 'corra' when corrected
    valid, stale = resolve_overrides(words, overrides, originals)
    assert valid == {1: "", 2: "VIATURA!"}
    assert stale == [StaleOverride(3, "CORRE", "corra", "corre")]
    # #9 is outside the window: neither applied nor stale (kept for when the edges move)
    assert 9 not in valid and all(s.idx != 9 for s in stale)
    # overrides saved before 5B have no originals: trusted
    valid, stale = resolve_overrides(words, {"3": "CORRE"}, None)
    assert valid == {3: "CORRE"} and stale == []
    # the original is compared after the same text cleanup as the transcript word
    valid, _ = resolve_overrides([W(5, "meu (deus)")], {"5": "x"}, {"5": "meu {deus}"})
    assert valid == {5: "x"}


def test_caption_words_order_glossary_then_overrides():
    window = [W(0, "Polícia!"), W(1, "corre"), W(2, "policia")]
    gloss = {"polícia": "viatura", "corre": "foge"}
    out, stale = caption_words(window, gloss, {"1": "VOA", "2": ""}, {"1": "corre", "2": "policia"})
    assert [w.text for w in out] == ["viatura!", "VOA"]  # override wins over the glossary
    assert stale == []
    assert [w.text for w in window] == ["Polícia!", "corre", "policia"]  # input untouched


# --- apply_glossary ----------------------------------------------------------------------------


def test_apply_glossary_whole_word_case_accents_punctuation():
    words = [W(0, "Policia,"), W(1, "POLÍCIAL"), W(2, "¿polícia?"), W(3, "kkkk"), W(4, "Jão")]
    g = {"polícia": "viatura", "joão": "Jão", "jao": "JOÃO"}
    out = apply_glossary(words, g)
    assert [w.text for w in out] == ["viatura,", "POLÍCIAL", "¿viatura?", "kkkk", "JOÃO"]
    assert [(w.idx, w.start, w.end) for w in out] == [(w.idx, w.start, w.end) for w in words]
    assert apply_glossary(words, None) is words and apply_glossary(words, {}) is words
    assert norm_word("¡Ação!!") == "acao"
    # invalid entries (empty / several words) are ignored
    assert [w.text for w in apply_glossary(words[:1], {"": "x", "po licia": "y"})] == ["Policia,"]


# --- routes ------------------------------------------------------------------------------------


@pytest.fixture
def client(settings):
    app = create_app(settings)
    seeded = seed_review(settings, app.state.engine)
    with TestClient(app) as c:
        c.engine, c.settings, c.seeded = app.state.engine, settings, seeded
        s1, s2 = seeded.sources
        c.a, c.b = seeded.cands[s1], seeded.cands[s2]
        c.st_a = seeded.streamers[0]
        yield c


def _cand(client, cid) -> Candidate:
    with Session(client.engine) as s:
        return s.get(Candidate, cid)


def _words(client, cid):
    return client.get(f"/candidatos/{cid}/palavras.json").json()


def test_edit_hide_undo_word(client):
    a1 = client.a[0]
    r = client.post(f"/candidatos/{a1}/legenda/palavra", data={"i": "2", "text": "VIATURA!"})
    assert r.json()["ok"]
    c = _cand(client, a1)
    assert c.caption_overrides == {"2": "VIATURA!"}
    assert c.caption_override_originals == {"2": "polícia!"}
    d = _words(client, a1)
    w2 = next(w for w in d["words"] if w["i"] == 2)
    assert (w2["t"], w2["s"], w2["e"]) == ("VIATURA!", 10.9, 11.6)  # timing unchanged
    ed = next(w for w in d["editor"] if w["i"] == 2)
    assert ed["raw"] == "polícia!" and ed["ov"] == "VIATURA!" and not ed["stale"]

    client.post(f"/candidatos/{a1}/legenda/palavra", data={"i": "4", "hide": "1"})
    assert 4 not in [w["i"] for w in _words(client, a1)["words"]]
    assert next(w for w in _words(client, a1)["editor"] if w["i"] == 4)["hidden"]

    client.post(f"/candidatos/{a1}/legenda/palavra", data={"i": "2", "undo": "1"})
    assert _cand(client, a1).caption_overrides == {"4": ""}
    assert next(w for w in _words(client, a1)["words"] if w["i"] == 2)["t"] == "polícia!"

    r = client.post(f"/candidatos/{a1}/legenda/palavra", data={"i": "99", "text": "x"})
    assert r.status_code == 422 and "não encontrada" in r.json()["error"]
    r = client.post(f"/candidatos/{a1}/legenda/palavra", data={"i": "x"})
    assert r.status_code == 422


def test_glossary_from_editor_applies_everywhere_without_touching_transcript(client):
    s1 = client.seeded.sources[0]
    a1, a2 = client.a[0], client.a[1]
    engine, settings = client.engine, client.settings
    with Session(engine) as s:
        before = rendering.plan_render(s, engine, settings, s.get(Candidate, a1), "preview")
        other_before = rendering.plan_render(s, engine, settings, s.get(Candidate, a2), "preview")
    r = client.post(f"/candidatos/{a1}/legenda/palavra",
                    data={"i": "3", "text": "Foge", "glossary": "1"})  # fmt: skip
    assert r.json()["ok"] and "glossário" in r.json()["message"]
    with Session(engine) as s:
        assert s.get(Streamer, client.st_a).glossary == {"corre": "Foge"}
    # the glossary covers it: no per-word override is needed
    assert not _cand(client, a1).caption_overrides
    d = _words(client, a1)
    assert [w["t"] for w in d["words"] if w["i"] in (3, 4)] == ["Foge", "Foge"]
    assert next(w for w in d["editor"] if w["i"] == 4)["gloss"] == "Foge"
    # the render (.ass words and params hash) follows; the saved transcript does not change
    with Session(engine) as s:
        after = rendering.plan_render(s, engine, settings, s.get(Candidate, a1), "preview")
        other_after = rendering.plan_render(s, engine, settings, s.get(Candidate, a2), "preview")
    assert [w.text for w in after.words if w.idx in (3, 4)] == ["Foge", "Foge"]
    assert after.phash != before.phash
    assert other_after.phash == other_before.phash  # a2 has no 'corre': its cache stays valid
    assert [w for _a, _b, w in rendering.transcript_words(engine, settings, s1)] == [
        w for _a, _b, w in WORDS
    ]
    # the glossary page lists it; source page cards too
    page = client.get(f"/streamers/{client.st_a}").text
    assert "Glossário" in page and 'value="corre"' in page and 'value="Foge"' in page
    assert "olha a polícia! Foge Foge" in client.get(f"/fontes/{s1}").text


def test_glossary_correction_plus_override_when_text_differs(client):
    a1 = client.a[0]
    client.post(f"/candidatos/{a1}/legenda/palavra",
                data={"i": "2", "text": "VIATURA!!", "glossary": "on"})  # fmt: skip
    with Session(client.engine) as s:
        assert s.get(Streamer, client.st_a).glossary == {"polícia": "VIATURA"}
    # glossary keeps the transcript punctuation ('!'), so the exact text needs an override
    assert _cand(client, a1).caption_overrides == {"2": "VIATURA!!"}


def test_stale_override_is_not_applied_and_can_be_removed(client):
    s1 = client.seeded.sources[0]
    a1 = client.a[0]
    client.post(f"/candidatos/{a1}/legenda/palavra", data={"i": "2", "text": "VIATURA!"})
    # the transcript is re-generated: word #2 is now another word
    tr = client.settings.work_dir / str(s1) / "transcript.json.gz"
    new = [Word(a, b, "pulícia" if i == 2 else w) for i, (a, b, w) in enumerate(WORDS)]
    save_transcript(tr, new, {"language": "pt"})
    save_artifact(client.engine, s1, "transcribe", "h2",
                  files={"transcript": client.settings.rel_path(tr)})  # fmt: skip
    d = _words(client, a1)
    assert next(w for w in d["words"] if w["i"] == 2)["t"] == "pulícia"  # not applied
    assert d["stale"] == [{"idx": 2, "text": "VIATURA!", "original": "polícia!",
                           "current": "pulícia"}]  # fmt: skip
    assert next(w for w in d["editor"] if w["i"] == 2)["stale"]
    with Session(client.engine) as s:
        plan = rendering.plan_render(s, client.engine, client.settings, s.get(Candidate, a1))
    assert "VIATURA!" not in [w.text for w in plan.words]
    r = client.post(f"/candidatos/{a1}/legenda/obsoletas/remover")
    assert r.json()["removed"] == 1
    assert not _cand(client, a1).caption_overrides and not _words(client, a1)["stale"]


def test_glossary_page_edit_and_remove(client):
    sid = client.st_a
    r = client.post(f"/streamers/{sid}/glossario", data={"errado": "Jão", "certo": "João"},
                    follow_redirects=False)  # fmt: skip
    assert r.status_code == 303
    client.post(f"/streamers/{sid}/glossario", data={"errado": "kkk", "certo": "KKK"})
    client.post(f"/streamers/{sid}/glossario", data={"antigo": "jao", "errado": "jaum",
                                                     "certo": "João"})  # fmt: skip
    with Session(client.engine) as s:
        assert s.get(Streamer, sid).glossary == {"jaum": "João", "kkk": "KKK"}
    r = client.post(f"/streamers/{sid}/glossario", data={"errado": "duas palavras", "certo": "x"},
                    follow_redirects=False)  # fmt: skip
    assert "sem%20espa" in r.headers["location"] or "sem espa" in r.headers["location"]
    client.post(f"/streamers/{sid}/glossario/remover", data={"errado": "KKK"})
    with Session(client.engine) as s:
        assert s.get(Streamer, sid).glossary == {"jaum": "João"}
    assert client.get(f"/streamers/{sid}").status_code == 200
    assert client.get(f"/streamers/{sid}/editar").status_code == 200


def test_review_page_has_caption_editor(client):
    html = client.get(f"/revisao?c={client.a[0]}").text
    assert 'id="caption-editor"' in html and "/static/caption_editor.js" in html
    assert "<kbd>E</kbd>" in html
    js = client.get("/static/caption_editor.js")
    assert js.status_code == 200 and "legenda/palavra" in js.text
