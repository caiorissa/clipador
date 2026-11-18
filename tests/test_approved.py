"""/aprovados: approved clips -> render what's missing -> post (now / next slot / date)."""

from datetime import timedelta

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from clipador.models import Job, Publication, PublicationStatus, Render
from clipador.services import approved
from clipador.utils import utcnow
from tests.test_publish import _approved_with_final, _connect, ysettings  # noqa: F401


def test_states_and_counts(ysettings, engine, tmp_path):
    cid = _approved_with_final(ysettings, engine, tmp_path)
    with Session(engine) as s:
        items = approved.items(s, ysettings)
        assert [(i.c.id, i.state) for i in items] == [(cid, "prontos")]
        assert items[0].title == "Capotou o carro"
        assert approved.counts(s, ysettings)["prontos"] == 1
        for r in s.exec(select(Render)).all():  # no final -> "falta renderizar"
            s.delete(r)
        s.commit()
        assert approved.items(s, ysettings)[0].state == "sem_render"


def test_post_on_the_grid_now_and_at_a_date(ysettings, engine, tmp_path):
    cid = _approved_with_final(ysettings, engine, tmp_path)
    acc1 = _connect(engine, ext="UC1", name="Geral")
    acc2 = _connect(engine, ext="UC2", name="Lacy")
    acc3 = _connect(engine, ext="UC3", name="Outro")
    from clipador.models import SocialAccount

    with Session(engine) as s:  # normal edit here (per-account styles: tests/test_variants.py)
        for a in s.exec(select(SocialAccount)).all():
            a.render_style = {}
            s.add(a)
        s.commit()
    with Session(engine, expire_on_commit=False) as s:
        p1 = approved.post(s, ysettings, cid, acc1, when="grade", privacy="public")
        p2 = approved.post(s, ysettings, cid, acc2, when="agora", privacy="public")
        p3 = approved.post(s, ysettings, cid, acc3, when="data", privacy="public",
                           at=utcnow() + timedelta(days=1))  # fmt: skip
        s.commit()
        assert p1.status == PublicationStatus.SCHEDULED and p1.scheduled_for is not None
        assert p2.status == PublicationStatus.QUEUED and s.get(Job, p2.job_id).type == "publish"
        assert p3.publish_at is not None
        assert approved.items(s, ysettings)[0].state == "postados"
        try:
            approved.post(s, ysettings, cid, acc1, when="grade", privacy="public")
            raise AssertionError("posted twice")
        except Exception as e:
            assert "já foi" in str(e)


def test_page_card_and_post_form(ysettings, engine, tmp_path):
    from clipador.web.app import create_app

    cid = _approved_with_final(ysettings, engine, tmp_path)
    acc = _connect(engine)
    client = TestClient(create_app(ysettings))
    page = client.get("/aprovados")
    assert page.status_code == 200 and "Capotou o carro" in page.text and "Postar" in page.text
    assert "Aprovados" in page.text
    r = client.post(f"/aprovados/{cid}/postar", data={"account_id": str(acc), "when": "grade"})
    assert "Marque que conferiu" in r.text
    r = client.post(f"/aprovados/{cid}/postar",
                    data={"account_id": str(acc), "when": "grade", "confirm": "on"})  # fmt: skip
    assert r.status_code == 200 and "Agendado para" in r.text
    with Session(engine) as s:
        assert s.exec(select(Publication)).one().status == PublicationStatus.SCHEDULED
    assert client.get("/aprovados?filtro=postados").text.count('class="ap-card') == 1
    assert client.get("/aprovados?filtro=sem_render").text.count('class="ap-card') == 0
