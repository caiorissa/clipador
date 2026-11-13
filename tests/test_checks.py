"""Step 5B: pre-publish checks, ready_to_publish, the 'prontos' filters and undo."""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from clipador.errors import ValidationError
from clipador.models import Candidate, Decision
from clipador.services import review
from clipador.services.review import ReviewFilters, ready_to_publish
from clipador.web.app import create_app
from tests.review_helpers import seed_review


def _c(status, checks=None):
    return Candidate(source_id=1, analysis_run_id=1, start=0, end=30, peak_t=5, score=50,
                     status=status, checks=checks)  # fmt: skip


def test_ready_to_publish_rules():
    both = {"musica_conferida": True, "alertas_reconhecidos": True}
    music = {"musica_conferida": True}
    assert ready_to_publish(_c("approved", music), has_alerts=False)
    assert ready_to_publish(_c("rendered", music), has_alerts=False)
    assert not ready_to_publish(_c("approved", music), has_alerts=True)  # alerts not read
    assert ready_to_publish(_c("approved", both), has_alerts=True)
    assert not ready_to_publish(_c("approved", None), has_alerts=False)
    assert not ready_to_publish(_c("approved", {"alertas_reconhecidos": True}), has_alerts=True)
    for status in ("suggested", "skipped", "rejected"):
        assert not ready_to_publish(_c(status, both), has_alerts=False)


@pytest.fixture
def client(settings):
    app = create_app(settings)
    seeded = seed_review(settings, app.state.engine)
    with TestClient(app) as c:
        c.engine, c.settings, c.seeded = app.state.engine, settings, seeded
        s1, s2 = seeded.sources
        c.a, c.b = seeded.cands[s1], seeded.cands[s2]
        yield c


def _cand(client, cid) -> Candidate:
    with Session(client.engine) as s:
        return s.get(Candidate, cid)


def test_checks_route_badge_and_filters(client):
    a1, b1 = client.a[0], client.b[0]  # a1 has AI alerts, b1 has none
    html = client.get(f"/revisao?c={a1}").text
    assert "Conferi música/copyright de fundo" in html and "Li os alertas da IA" in html
    assert "A IA não detecta música com direitos autorais" in html
    assert "Li os alertas da IA" not in client.get(f"/revisao?c={b1}").text

    # checking before approving is allowed, but not 'ready' yet; approving is never blocked
    r = client.post(f"/revisao/{b1}/verificacoes", data={"musica_conferida": "1"}).json()
    assert r == {"ok": True, "checks": {"musica_conferida": True, "alertas_reconhecidos": False},
                 "ready": False}  # fmt: skip
    client.post(f"/revisao/{b1}/decisao", data={"action": "approve"})
    client.post(f"/revisao/{a1}/decisao", data={"action": "approve"})
    assert _cand(client, a1).status == "approved"
    r = client.post(f"/revisao/{a1}/verificacoes", data={"musica_conferida": "on"}).json()
    assert r["ready"] is False  # a1 still needs "li os alertas"
    r = client.post(f"/revisao/{a1}/verificacoes", data={"alertas_reconhecidos": "1"}).json()
    assert r["ready"] is True and r["checks"]["musica_conferida"] is True

    # queue filter 'prontos'
    html = client.get("/revisao?status=prontos").text
    assert "pronto para publicar" in html and 'data-candidate="' in html
    with Session(client.engine) as s:
        f = ReviewFilters(status="prontos")
        assert review.ready_ids(s, review.scope(s, f)) == {a1, b1}
        assert review.next_pending(s, f).id in {a1, b1}
        assert review.progress(s, ReviewFilters())["ready"] == 2
    # source page filter
    s1 = client.seeded.sources[0]
    page = client.get(f"/fontes/{s1}?filtro=prontos").text
    assert f'id="cand-{a1}"' in page and f'id="cand-{client.a[1]}"' not in page
    assert "✔ pronto para publicar" in page
    assert f'id="cand-{client.a[1]}"' in client.get(f"/fontes/{s1}").text

    r = client.post(f"/revisao/{a1}/verificacoes", data={"musica_conferida": "0"}).json()
    assert r["ready"] is False


def test_checks_are_in_history_and_undoable(client):
    a1 = client.a[0]
    client.post(f"/revisao/{a1}/decisao", data={"action": "approve"})
    client.post(f"/revisao/{a1}/verificacoes", data={"musica_conferida": "1"})
    client.post(f"/revisao/{a1}/verificacoes", data={"musica_conferida": "1"})  # no change
    client.post(f"/revisao/{a1}/verificacoes", data={"alertas_reconhecidos": "1"})
    with Session(client.engine) as s:
        actions = [d.action for d in s.exec(select(Decision).order_by(Decision.id))]
    assert actions == ["approve", "edit_checks", "edit_checks"]
    assert "mudou as verificações de" in client.get("/revisao?status=todos").text

    client.post("/revisao/desfazer")
    assert _cand(client, a1).checks == {"musica_conferida": True, "alertas_reconhecidos": False}
    client.post("/revisao/desfazer")
    assert not _cand(client, a1).checks
    assert _cand(client, a1).status == "approved"  # undoing checks doesn't touch the decision
    client.post("/revisao/desfazer")
    assert _cand(client, a1).status == "suggested"


def test_undo_of_old_decision_keeps_checks(client):
    """Decisions recorded before 5B have no 'checks' in prev_state: undo leaves them alone."""
    a1 = client.a[0]
    with Session(client.engine) as s:
        c = s.get(Candidate, a1)
        c.status, c.checks = "approved", {"musica_conferida": True}
        s.add(c)
        s.add(Decision(candidate_id=a1, action="approve", prev_state={"status": "suggested",
                       "reviewed_at": None, "user_start": None, "user_end": None,
                       "feedback": None}))  # fmt: skip
        s.commit()
    client.post("/revisao/desfazer")
    c = _cand(client, a1)
    assert c.status == "suggested" and c.checks == {"musica_conferida": True}


def test_set_checks_rejects_unknown_key(client):
    with Session(client.engine) as s, pytest.raises(ValidationError):
        review.set_checks(s, client.a[0], foo=True)
