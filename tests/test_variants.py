"""Per-account edit styles (step 6C): each account posts its own render of a clip."""

import time

import pytest
from sqlmodel import Session, select

from clipador.models import (
    Candidate,
    Job,
    Publication,
    PublicationStatus,
    Render,
    RenderStatus,
    SocialAccount,
)
from clipador.render import variants
from clipador.services import accounts, rendering
from tests.conftest import requires_ffmpeg
from tests.render_helpers import ffprobe_json, make_video, seed
from tests.test_publish import _approved_with_final, ysettings  # noqa: F401


def test_normalize_suggest_describe():
    assert variants.is_default({}) and variants.is_default({"speed": 1.0, "bogus": 3})
    s = variants.normalize({"speed": 9, "zoom": 0.2, "grade": "xx", "opening": "t2"})
    assert (s["speed"], s["zoom"], s["grade"], s["opening"]) == (1.15, 1.0, "nenhum", "t2")
    used = []
    for _ in range(len(variants.PRESETS) + 3):  # always a look nobody else has
        nxt = variants.suggest(used, seed=len(used))
        assert nxt not in used
        used.append(nxt)
    assert variants.describe({}) == "edição padrão"
    assert "1.05x" in variants.describe({"speed": 1.05, "grade": "vivo"})


def test_new_accounts_get_distinct_styles(engine):
    with Session(engine) as s:
        styles = []
        for i in range(4):
            a = accounts.upsert_account(s, "youtube", f"UC{i}", f"c{i}", {"refresh_token": "r"})
            styles.append(variants.normalize(a.render_style))
        s.commit()
    assert len({str(sorted(x.items())) for x in styles}) == 4
    assert variants.is_default(styles[0])  # the first account keeps the normal edit


def test_styled_account_posts_its_own_render_when_ready(ysettings, engine, tmp_path, monkeypatch):
    _approved_with_final(ysettings, engine, tmp_path)
    style = variants.normalize({"grade": "vivo", "speed": 1.05})
    queued = []

    def fake_enqueue(session, settings, source_id, cids, kind, priority=0, variant=None,
                     account_id=None):  # fmt: skip
        r = Render(candidate_id=cids[0], source_id=source_id, kind=kind, params_hash="v",
                   params={"variant": variant}, account_id=account_id, variant=variant)  # fmt: skip
        session.add(r)
        session.flush()
        queued.append(r.id)
        return None, [r], []

    from clipador.services import analysis

    monkeypatch.setattr(analysis, "enqueue_render", fake_enqueue)
    with Session(engine, expire_on_commit=False) as s:
        acc = accounts.upsert_account(
            s,
            "youtube",
            "UCx",
            "Perfil",
            {"refresh_token": "r", "access_token": "a", "expires_at": time.time() + 999},
        )
        acc.render_style, acc.autopost = style, True
        acc.autopost_since = None
        s.add(acc)
        s.commit()
        accounts.autopost_tick(s, ysettings)
        s.commit()
        pub = s.exec(select(Publication)).one()
        r = s.get(Render, pub.render_id)
    assert r.account_id == acc.id and r.variant == style and queued == [r.id]
    assert pub.status == PublicationStatus.SCHEDULED
    with Session(engine) as s:  # due, but its render isn't done: nothing is sent
        p = s.get(Publication, pub.id)
        p.scheduled_for = p.created_at
        s.add(p)
        s.commit()
        accounts.autopost_tick(s, ysettings)
        s.commit()
        assert s.get(Publication, pub.id).status == PublicationStatus.SCHEDULED
        row = s.get(Render, r.id)
        row.status, row.path = RenderStatus.DONE, s.get(Render, 1).path  # rendered now
        s.add(row)
        s.commit()
        accounts.autopost_tick(s, ysettings)
        s.commit()
        assert s.get(Publication, pub.id).status == PublicationStatus.QUEUED
        assert s.exec(select(Job).where(Job.type == "publish")).first() is not None


@requires_ffmpeg
def test_variant_render_is_a_different_file(ysettings, engine, tmp_path):
    """Real ffmpeg: grade + speed + trim + zoom produce a valid, shorter, different clip."""
    video = make_video(ysettings.vods_dir / "1" / "video.mp4")
    ysettings.render_encoder = "x264"
    ysettings.render_cut_silences = False
    _sid, (cid,), _ = seed(ysettings, engine, video, candidates=((0.5, 10.5, 2.0),))
    style = {"grade": "quente", "speed": 1.1, "trim_start": 1.0, "zoom": 1.15,
             "caption_style": "pop", "opening": "nenhum"}  # fmt: skip
    with Session(engine) as s:
        c = s.get(Candidate, cid)
        normal = rendering.plan_render(s, engine, ysettings, c, "preview")
        styled = rendering.plan_render(s, engine, ysettings, c, "preview", style)
        assert normal.phash != styled.phash and styled.start == pytest.approx(1.5)
        todo, _ = rendering.request_renders(s, engine, ysettings, [cid], "preview", style, 7)
        s.commit()
        row = todo[0]
        assert row.account_id == 7 and row.variant["speed"] == 1.1
        plan = rendering.plan_render(s, engine, ysettings, c, "preview", row.variant)
        rendering.execute_render(s, ysettings, row, plan, lambda f: None, lambda m: None)
        out = ysettings.abs_path(row.path)
    dur = float(ffprobe_json(out, "format=duration")["format"]["duration"])
    assert dur == pytest.approx((10.5 - 1.5) / 1.1, abs=0.2)


def test_account_style_form(ysettings, engine):
    from fastapi.testclient import TestClient

    from clipador.web.app import create_app

    with Session(engine) as s:
        acc = accounts.upsert_account(s, "tiktok", "t1", "Perfil", {"refresh_token": "r"})
        s.commit()
        acc_id = acc.id
    client = TestClient(create_app(ysettings))
    page = client.get("/publicar")
    assert "Estilo de edição desta conta" in page.text and "Sortear outro estilo" in page.text
    client.post(
        f"/publicar/contas/{acc_id}",
        data={
            "slots": "12:00",
            "max_per_day": "3",
            "privacy": "public",
            "style_form": "1",
            "st_grade": "frio",
            "st_speed": "1.08",
            "st_zoom": "1.1",
            "st_foco": "0.4",
            "st_trim": "0.6",
            "st_opening": "t3",
            "st_credit": "on",
        },
    )
    with Session(engine) as s:
        st = s.get(SocialAccount, acc_id).render_style
    assert (st["grade"], st["speed"], st["trim_start"], st["opening"]) == ("frio", 1.08, 0.6, "t3")
    assert st["credit"] and not st["peak_zoom"]
    r = client.post(f"/publicar/contas/{acc_id}/sortear-estilo", follow_redirects=False)
    assert "novo%20estilo" in r.headers["location"] and "padr" not in r.headers["location"]
    with Session(engine) as s:
        assert s.get(SocialAccount, acc_id).render_style != st
