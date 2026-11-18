"""Auto-post (step 6B): the time grid, scheduling only APPROVED clips, holds, pause, the
YouTube upload lead, the queue actions, the legacy token import, and TikTok (fake API)."""

import json
import time
from datetime import UTC, datetime, timedelta, timezone

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from clipador.models import (
    Candidate,
    CandidateStatus,
    Job,
    PostDraft,
    Publication,
    PublicationStatus,
    SocialAccount,
)
from clipador.pipeline.base import StageContext
from clipador.pipeline.stages.publish import PublishStage
from clipador.services import accounts, tiktok, youtube
from clipador.utils import utcnow
from tests.test_publish import _approved_with_final, _Rep, ysettings  # noqa: F401

BRT = timezone(timedelta(hours=-3))


def _acc(**kw) -> SocialAccount:
    base = {"platform": "youtube", "name": "Canal", "slots": ["12:00", "18:00", "21:00"],
            "max_per_day": 3, "min_gap_minutes": 60}  # fmt: skip
    return SocialAccount(**{**base, **kw})


def _at(day, hh, mm=0):
    return datetime(2026, 11, day, hh, mm, tzinfo=BRT).astimezone(UTC)


# --- the grid ----------------------------------------------------------------------------------


def test_next_slot_grid_cap_and_gap():
    a = _acc()
    now = _at(18, 13)  # 13:00 local: the 12:00 slot is gone
    assert accounts.next_slot(now, a, [], BRT) == _at(18, 18)
    assert accounts.next_slot(now, a, [_at(18, 18)], BRT) == _at(18, 21)
    # day full (3 already that day) -> tomorrow's first slot
    full = [_at(18, 9), _at(18, 18), _at(18, 21)]
    assert accounts.next_slot(now, a, full, BRT) == _at(19, 12)
    a2 = _acc(max_per_day=1)
    assert accounts.next_slot(now, a2, [_at(18, 9)], BRT) == _at(19, 12)
    # min gap: a manual post at 17:30 blocks the 18:00 slot (60 min gap)
    assert accounts.next_slot(now, a, [_at(18, 17, 30)], BRT) == _at(18, 21)


def test_next_slot_post_now_mode():
    a = _acc(post_now=True, max_per_day=2, min_gap_minutes=30)
    now = _at(18, 10)
    assert accounts.next_slot(now, a, [], BRT) == now
    assert accounts.next_slot(now, a, [_at(18, 9, 50)], BRT) == _at(18, 10, 20)
    nxt = accounts.next_slot(now, a, [_at(18, 8), _at(18, 9)], BRT)
    assert nxt == datetime(2026, 11, 19, 0, 0, tzinfo=BRT).astimezone(UTC)


def test_parse_slots():
    assert accounts.parse_slots("21:30, 12h; 18") == ["12:00", "18:00", "21:30"]
    with pytest.raises(Exception, match="inválido"):
        accounts.parse_slots("25:00")


# --- the tick --------------------------------------------------------------------------------------


def _account(engine, **kw) -> int:
    with Session(engine) as s:
        a = accounts.upsert_account(s, kw.pop("platform", "youtube"), kw.pop("ext", "UC1"),
                                    "Canal", {"refresh_token": "rt", "access_token": "at",
                                              "expires_at": time.time() + 3000})  # fmt: skip
        a.autopost, a.autopost_since = True, utcnow() - timedelta(minutes=5)
        for k, v in kw.items():
            setattr(a, k, v)
        s.add(a)
        s.commit()
        return a.id


def _pubs(engine):
    with Session(engine) as s:
        return s.exec(select(Publication).order_by(Publication.id)).all()


def test_tick_schedules_only_approved_and_respects_holds(ysettings, engine, tmp_path):
    cid = _approved_with_final(ysettings, engine, tmp_path)
    acc = _account(engine)
    with Session(engine) as s:  # an older approval (before auto-post) and a suggestion
        base = s.get(Candidate, cid)
        old = Candidate(source_id=base.source_id, analysis_run_id=base.analysis_run_id, rank=2,
                        start=20, end=50, peak_t=30, score=90, status=CandidateStatus.APPROVED,
                        reviewed_at=utcnow() - timedelta(days=1))  # fmt: skip
        sug = Candidate(source_id=base.source_id, analysis_run_id=base.analysis_run_id, rank=3,
                        start=60, end=90, peak_t=70, score=95)  # fmt: skip
        s.add_all([old, sug])
        base.reviewed_at = utcnow()
        s.add(base)
        s.commit()
    now = _at(18, 13)
    ysettings.youtube_upload_lead_hours = 0  # keep it scheduled (no early upload) here
    with Session(engine) as s:
        notes = accounts.autopost_tick(s, ysettings, now=now, tz=BRT)
        s.commit()
    pubs = _pubs(engine)
    assert len(pubs) == 1 and pubs[0].candidate_id == cid and pubs[0].auto
    assert pubs[0].status == PublicationStatus.SCHEDULED and pubs[0].account_id == acc
    assert accounts.aware(pubs[0].scheduled_for) == _at(18, 18) and "agendado" in notes[0]
    with Session(engine) as s:  # idempotent: the same clip is never scheduled twice
        accounts.autopost_tick(s, ysettings, now=now, tz=BRT)
        s.commit()
    assert len(_pubs(engine)) == 1


def test_tick_holds_music_and_queues_due_uploads(ysettings, engine, tmp_path):
    cid = _approved_with_final(ysettings, engine, tmp_path)
    with Session(engine) as s:
        c = s.get(Candidate, cid)
        c.breakdown = {"music": 85.0}
        s.add(c)
        s.commit()
    _account(engine)
    with Session(engine) as s:
        accounts.autopost_tick(s, ysettings, now=_at(18, 13), tz=BRT)
        s.commit()
    p = _pubs(engine)[0]
    assert p.status == PublicationStatus.HELD and "música" in p.hold_reason
    with Session(engine) as s:
        accounts.release(s, ysettings, p.id, now=_at(18, 13))
        s.commit()
    p = _pubs(engine)[0]
    assert p.status == PublicationStatus.SCHEDULED
    # YouTube: due 12 h before its slot (uploaded private, public at the slot)
    with Session(engine) as s:
        accounts.autopost_tick(s, ysettings, now=accounts.aware(p.scheduled_for)
                               - timedelta(hours=11), tz=BRT)  # fmt: skip
        s.commit()
        job = s.exec(select(Job).where(Job.type == "publish")).one()
    assert _pubs(engine)[0].status == PublicationStatus.QUEUED and job.priority > 0


def test_paused_account_and_tiktok_waits_for_its_slot(ysettings, engine, tmp_path):
    cid = _approved_with_final(ysettings, engine, tmp_path)
    acc = _account(engine, platform="tiktok", ext="tt1", paused=True)
    with Session(engine) as s:
        s.add(PostDraft(candidate_id=cid, platform="tiktok", description="Que isso",
                        hashtags=["gta6"]))  # fmt: skip
        s.commit()
        accounts.autopost_tick(s, ysettings, now=_at(18, 13), tz=BRT)
        s.commit()
    p = _pubs(engine)[0]
    assert p.platform == "tiktok" and p.description == "Que isso #gta6" and p.privacy == "private"
    at_slot = accounts.aware(p.scheduled_for)
    with Session(engine) as s:  # paused: due but not sent
        accounts.autopost_tick(s, ysettings, now=at_slot + timedelta(minutes=1), tz=BRT)
        s.commit()
    assert _pubs(engine)[0].status == PublicationStatus.SCHEDULED
    with Session(engine) as s:
        a = s.get(SocialAccount, acc)
        a.paused = False
        s.add(a)
        s.commit()
        accounts.autopost_tick(s, ysettings, now=at_slot - timedelta(hours=2), tz=BRT)
        s.commit()
    assert _pubs(engine)[0].status == PublicationStatus.SCHEDULED  # no lead for TikTok
    with Session(engine) as s:
        accounts.post_now(s, p.id)
        s.commit()
        accounts.autopost_tick(s, ysettings, tz=BRT)
        s.commit()
    assert _pubs(engine)[0].status == PublicationStatus.QUEUED


def test_cancel_and_legacy_token_import(ysettings, engine, tmp_path):
    youtube.save_token(
        ysettings, {"refresh_token": "old", "channel": {"id": "UC9", "title": "Antigo"}}
    )
    with Session(engine) as s:
        acc = accounts.import_legacy_youtube(s, ysettings)
        s.commit()
        assert acc.name == "Antigo" and acc.credentials == {"refresh_token": "old"}
    assert not youtube.token_path(ysettings).exists()
    with Session(engine) as s, pytest.raises(Exception, match="cancelar"):
        accounts.cancel(s, 999)


# --- TikTok ------------------------------------------------------------------------------------------


class FakeTikTok:
    def __init__(self, options=("SELF_ONLY",)):
        self.options = list(options)
        self.uploaded = b""
        self.init = None
        self.polls = 0

    def handler(self, req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if url.startswith(tiktok.TOKEN_URL):
            form = dict(x.split("=", 1) for x in req.content.decode().split("&"))
            body = {"access_token": "tt-at", "expires_in": 86400, "refresh_token": "tt-rt",
                    "open_id": "o1"}  # fmt: skip
            assert form["grant_type"] in ("authorization_code", "refresh_token")
            return httpx.Response(200, json=body)
        ok = {"error": {"code": "ok", "message": ""}}
        if "/user/info/" in url:
            return httpx.Response(200, json={"data": {"user": {"open_id": "o1",
                                                               "display_name": "fulano.tt"}}, **ok})  # fmt: skip
        if "/creator_info/query/" in url:
            return httpx.Response(200, json={"data": {"privacy_level_options": self.options}, **ok})
        if "/video/init/" in url:
            self.init = json.loads(req.content)
            return httpx.Response(200, json={"data": {"publish_id": "p1",
                                                      "upload_url": "https://up.tt/1"}, **ok})  # fmt: skip
        if url.startswith("https://up.tt/1"):
            self.uploaded += req.content
            return httpx.Response(201)
        if "/status/fetch/" in url:
            self.polls += 1
            if self.polls < 2:
                return httpx.Response(200, json={"data": {"status": "PROCESSING_UPLOAD"}, **ok})
            return httpx.Response(200, json={"data": {"status": "PUBLISH_COMPLETE",
                                                      "publicaly_available_post_id": [777]}, **ok})  # fmt: skip
        if "/video/query/" in url:
            return httpx.Response(200, json={"data": {"videos": [
                {"id": "777", "view_count": 42, "like_count": 5, "comment_count": 1}]}, **ok})  # fmt: skip
        return httpx.Response(404)


@pytest.fixture
def fake_tt(monkeypatch, ysettings):
    ysettings.tiktok_client_key, ysettings.tiktok_client_secret = "ck", "cs"
    fake = FakeTikTok(options=["SELF_ONLY", "PUBLIC_TO_EVERYONE"])
    real = httpx.Client
    monkeypatch.setattr(httpx, "Client",
                        lambda *a, **k: real(transport=httpx.MockTransport(fake.handler)))  # fmt: skip
    monkeypatch.setattr(time, "sleep", lambda s: None)
    return fake


def test_tiktok_oauth_upload_status_stats(ysettings, engine, tmp_path, fake_tt):
    v, ch = tiktok.new_pkce()
    assert len(ch) == 64 and int(ch, 16) >= 0  # hex SHA-256
    url = tiktok.auth_url(tiktok.load_client(ysettings), "http://127.0.0.1:8000/cb", "s", ch)
    assert "client_key=ck" in url and "video.publish" in url
    tok = tiktok.exchange_code(ysettings, "c", "http://127.0.0.1:8000/cb", v)
    assert tok["user"] == {"id": "o1", "title": "fulano.tt"}

    cid = _approved_with_final(ysettings, engine, tmp_path)
    with Session(engine) as s:
        acc = accounts.upsert_account(s, "tiktok", "o1", "fulano.tt",
                                      {k: v for k, v in tok.items() if k != "user"})  # fmt: skip
        s.commit()
        pub = accounts.new_publication(s, acc, s.get(Candidate, cid), 1, privacy="public")
        pub.status = PublicationStatus.QUEUED
        s.add(pub)
        s.commit()
        pid = pub.id
    ctx = StageContext(1, "publish", None, {"publication_id": pid}, 1, ysettings, engine, _Rep())
    PublishStage().run(ctx)
    with Session(engine) as s:
        p = s.get(Publication, pid)
    assert p.status == PublicationStatus.PUBLISHED and p.external_id == "777"
    assert p.url.endswith("/video/777") and p.privacy == "public"
    assert fake_tt.init["post_info"]["privacy_level"] == "PUBLIC_TO_EVERYONE"
    assert fake_tt.init["source_info"]["total_chunk_count"] == 1
    assert fake_tt.uploaded == b"\x00" * 3000

    from clipador.services import publishing

    with Session(engine) as s:
        assert publishing.refresh_metrics(s, ysettings, engine) == 1
        s.commit()
        assert s.get(Publication, pid).views == 42


def test_tiktok_unaudited_falls_back_to_private(ysettings, fake_tt, tmp_path):
    fake_tt.options = ["SELF_ONLY"]
    video = tmp_path / "v.mp4"
    video.write_bytes(b"1" * 100)
    store = youtube.MemoryStore({"access_token": "a", "refresh_token": "r",
                                 "expires_at": time.time() + 999})  # fmt: skip
    pid, used = tiktok.TikTok(ysettings, store=store).upload(video, "oi", "PUBLIC_TO_EVERYONE")
    assert (pid, used) == ("p1", "SELF_ONLY")


# --- web ------------------------------------------------------------------------------------------


def test_account_settings_and_queue_pages(ysettings, engine, tmp_path):
    from clipador.web.app import create_app

    acc = _account(engine)
    with Session(engine) as s:
        a = s.get(SocialAccount, acc)
        a.autopost = False
        s.add(a)
        s.commit()
    client = TestClient(create_app(ysettings))
    page = client.get("/publicar")
    assert "Canal" in page.text and "Postagem automática" in page.text
    r = client.post(f"/publicar/contas/{acc}", data={
        "autopost": "on", "slots": "21h, 12:00", "max_per_day": "2", "min_gap_minutes": "90",
        "privacy": "public", "hold_on_alerts": "on", "streamer_ids": ["1"]},
        follow_redirects=False)  # fmt: skip
    assert r.status_code == 303 and "ligada" in r.headers["location"]
    with Session(engine) as s:
        a = s.get(SocialAccount, acc)
    assert a.autopost and a.autopost_since and a.slots == ["12:00", "21:00"]
    assert (a.max_per_day, a.min_gap_minutes, a.streamer_ids) == (2, 90, [1])
    bad = client.post(f"/publicar/contas/{acc}", data={"slots": "", "privacy": "public"})
    assert bad.status_code == 400 and "horário" in bad.text
    client.post("/publicar/pausar-tudo", data={"pausar": "1"})
    with Session(engine) as s:
        assert s.get(SocialAccount, acc).paused
    r = client.post("/publicar/fila/999/liberar")
    assert r.status_code == 400
    r = client.post(f"/publicar/contas/{acc}/desconectar", follow_redirects=False)
    with Session(engine) as s:
        a = s.get(SocialAccount, acc)
    assert not a.active and a.credentials == {} and not a.autopost


# --- several profiles (general + per streamer) ------------------------------------------------------


def test_project_quota_is_shared_between_channels():
    a = _acc(slots=["10:00", "12:00", "14:00", "16:00"], max_per_day=10, min_gap_minutes=0)
    now = _at(18, 9)
    pt = timezone(timedelta(hours=-8))
    # the project already has 2 uploads that Pacific day (from another channel): cap 2 -> next day
    other = [_at(18, 10, 30), _at(18, 11)]
    t = accounts.next_slot(now, a, [], BRT, group_taken=other, group_cap=2, group_tz=pt)
    assert t.astimezone(pt).date() > _at(18, 11).astimezone(pt).date()
    assert accounts.next_slot(now, a, [], BRT, group_taken=other, group_cap=3,
                              group_tz=pt) == _at(18, 10)  # fmt: skip


def test_general_profile_alternates_streamers_and_waits_its_delay(ysettings, engine, tmp_path):
    from datetime import date

    from clipador.models import Source, Streamer

    cid = _approved_with_final(ysettings, engine, tmp_path)
    with Session(engine) as s:
        base = s.get(Candidate, cid)
        src = s.get(Source, base.source_id)
        st2 = Streamer(
            display_name="Outro", permission_confirmed=True, permission_date=date.today()
        )
        s.add(st2)
        s.flush()
        src2 = Source(streamer_id=st2.id, platform="twitch", kind="vod", external_id="2", url="u2",
                      status="ready", file_path=src.file_path)  # fmt: skip
        s.add(src2)
        s.flush()
        ids = {}
        for name, source_id, scores in (("a", src.id, [99, 98, 97]), ("b", src2.id, [50])):
            for i, sc in enumerate(scores):
                c = Candidate(source_id=source_id, analysis_run_id=base.analysis_run_id, rank=i,
                              start=100 + i * 40, end=130 + i * 40, peak_t=110 + i * 40, score=sc,
                              status=CandidateStatus.APPROVED, reviewed_at=utcnow())  # fmt: skip
                s.add(c)
                s.flush()
                ids[(name, i)] = c.id
        base.status = CandidateStatus.SUGGESTED
        s.add(base)
        s.commit()
        acc = s.get(SocialAccount, _account(engine, delay_minutes=0))
        order = [c.id for c in accounts._candidates_for(s, acc)]
    # best of A, then B (alternating), then the rest of A
    assert order[:3] == [ids[("a", 0)], ids[("b", 0)], ids[("a", 1)]]
    with Session(engine) as s:
        acc.delay_minutes = 6 * 60
        assert accounts._candidates_for(s, acc) == []  # approved just now: waits 6 h
        later = accounts._candidates_for(s, acc, utcnow() + timedelta(hours=7))
        assert len(later) == 4


def test_same_clip_gets_another_title_on_a_second_channel(ysettings, engine, tmp_path):
    from clipador.models import CandidateEnrichment

    cid = _approved_with_final(ysettings, engine, tmp_path)
    a1 = _account(engine, ext="UC1")
    a2 = _account(engine, ext="UC2")
    with Session(engine) as s:
        src_id = s.get(Candidate, cid).source_id
        s.add(CandidateEnrichment(candidate_id=cid, source_id=src_id, status="ok", prompt_version="v1", model="m", input_hash="h",
                                  titulos=["Capotou o carro", "Que capotada!", "Olha isso"]))  # fmt: skip
        s.commit()
        c = s.get(Candidate, cid)
        p1 = accounts.new_publication(s, s.get(SocialAccount, a1), c, 1, privacy="public")
        s.add(p1)
        s.flush()
        p2 = accounts.new_publication(s, s.get(SocialAccount, a2), c, 1, privacy="public")
    assert p1.title == "Capotou o carro" and p2.title == "Que capotada!"


def test_tiktok_daily_cap_is_enforced_in_settings(ysettings, engine):
    from clipador.web.app import create_app

    acc = _account(engine, platform="tiktok", ext="tt9")
    client = TestClient(create_app(ysettings))
    client.post(f"/publicar/contas/{acc}", data={"autopost": "on", "slots": "12:00",
                                                 "max_per_day": "40", "privacy": "public",
                                                 "delay_hours": "1,5"})  # fmt: skip
    with Session(engine) as s:
        a = s.get(SocialAccount, acc)
    assert a.max_per_day == accounts.TIKTOK_MAX_PER_DAY and a.delay_minutes == 90


def test_several_google_projects(ysettings, tmp_path):
    d = youtube.clients_dir(ysettings)
    d.mkdir(parents=True)
    (d / "projeto2.json").write_text(json.dumps({"installed": {"client_id": "cid2",
                                                               "client_secret": "s2"}}))  # fmt: skip
    clients = youtube.load_clients(ysettings)
    assert [c.client_id for c in clients] == ["cid", "cid2"]
    assert youtube.load_client(ysettings, "cid2").name == "projeto2"
    with pytest.raises(youtube.YouTubeError, match="não está mais configurado"):
        youtube.load_client(ysettings, "sumiu")
