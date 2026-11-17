"""Publishing to YouTube (step 6) against a fake Google API (httpx.MockTransport): OAuth code
exchange and refresh, resumable upload with a 308 resume, scheduling, the rules (approved +
final render + connected + confirmation), metrics and the performance labels for calibrate."""

import json
import time
from datetime import timedelta

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
    Render,
    RenderKind,
    RenderStatus,
)
from clipador.pipeline.base import StageContext
from clipador.pipeline.stages.publish import PublishStage
from clipador.services import publishing, youtube
from clipador.utils import utcnow
from tests.render_helpers import seed


class FakeGoogle:
    """Just enough of accounts.google.com + the YouTube Data API."""

    def __init__(self, resume_once=True):
        self.calls: list[tuple[str, str]] = []
        self.uploaded = b""
        self.meta = None
        self.resume_once = resume_once
        self.token_requests = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        self.calls.append((req.method, url.split("?")[0]))
        if url.startswith(youtube.TOKEN_URL):
            form = dict(x.split("=", 1) for x in req.content.decode().split("&"))
            self.token_requests.append(form)
            body = {"access_token": f"at-{len(self.token_requests)}", "expires_in": 3600}
            if form["grant_type"] == "authorization_code":
                body["refresh_token"] = "rt-1"
            return httpx.Response(200, json=body)
        assert req.headers.get("authorization", "").startswith("Bearer at-")
        if "/channels" in url:
            return httpx.Response(
                200, json={"items": [{"id": "UC1", "snippet": {"title": "Meu Canal"}}]}
            )
        if req.method == "POST" and url.startswith(youtube.UPLOAD_URL):
            self.meta = json.loads(req.content)
            return httpx.Response(200, headers={"Location": "https://upload.example/session/1"})
        if req.method == "PUT" and url.startswith("https://upload.example/session/1"):
            rng = req.headers["content-range"]  # bytes a-b/total
            a, b = (int(x) for x in rng.split()[1].split("/")[0].split("-"))
            if self.resume_once:  # Google keeps only part of the first chunk
                self.resume_once = False
                keep = (b - a + 1) // 2
                self.uploaded += req.content[:keep]
                return httpx.Response(308, headers={"Range": f"bytes=0-{a + keep - 1}"})
            self.uploaded += req.content
            total = int(rng.split("/")[1])
            if b + 1 < total:
                return httpx.Response(308, headers={"Range": f"bytes=0-{b}"})
            return httpx.Response(200, json={"id": "vid123"})
        if "/videos" in url:
            return httpx.Response(200, json={"items": [
                {"id": "vid123", "statistics": {"viewCount": "1500", "likeCount": "90",
                                                "commentCount": "7"},
                 "status": {"privacyStatus": "public"}}]})  # fmt: skip
        return httpx.Response(404, json={"error": {"errors": [{"reason": "notFound"}]}})


@pytest.fixture
def google(monkeypatch):
    g = FakeGoogle()
    transport = httpx.MockTransport(g.handler)
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: real_client(transport=transport))
    return g


@pytest.fixture
def ysettings(settings, tmp_path):
    secrets = tmp_path / "client_secret.json"
    secrets.write_text(json.dumps({"installed": {"client_id": "cid", "client_secret": "cs"}}))
    settings.youtube_client_secrets = secrets
    return settings


def _connect(engine, platform="youtube", ext="UC1", name="Meu Canal"):
    """A connected account (token valid for a while). Returns its id."""
    from clipador.services import accounts

    with Session(engine) as s:
        acc = accounts.upsert_account(
            s,
            platform,
            ext,
            name,
            {"access_token": "at-0", "refresh_token": "rt-1", "expires_at": time.time() + 3000},
        )
        s.commit()
        return acc.id


def _approved_with_final(settings, engine, tmp_path, with_render=True):
    from unittest import mock

    from clipador.services.ffmpeg import MediaInfo

    video = settings.vods_dir / "1" / "video.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"x")
    info = MediaInfo(12.0, 1280, 720, 30.0, "h264", "aac", 1)
    with mock.patch("tests.render_helpers.probe", return_value=info):
        _sid, (cid,), _ = seed(settings, engine, video, candidates=((0.5, 5.5, 2.0),))
    with Session(engine) as s:
        s.add(PostDraft(candidate_id=cid, platform="youtube", title="Capotou o carro",
                        description="Que isso\n\nClip de @fulano", hashtags=["#gta6", "#shorts"]))  # fmt: skip
        if with_render:
            out = settings.clips_dir / "1" / "final.mp4"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"\x00" * 3000)
            s.add(Render(candidate_id=cid, source_id=1, kind=RenderKind.FINAL, params_hash="h",
                         status=RenderStatus.DONE, path=settings.rel_path(out)))  # fmt: skip
        s.commit()
    return cid


class _Rep:
    def report_progress(self, job_id, pct):
        self.pct = pct

    def append_log(self, job_id, line):
        pass

    def is_canceled(self, job_id):
        return False

    def is_stopping(self):
        return False


def _run_stage(settings, engine, pub_id):
    ctx = StageContext(1, "publish", None, {"publication_id": pub_id}, 1, settings,
                       engine, _Rep())  # fmt: skip
    PublishStage().run(ctx)


# --- pure / API ---------------------------------------------------------------------------------


def test_video_meta_schedule_and_hashtags():
    m = youtube.video_meta("T", "desc", ["#gta6", "shorts"], "public", "2026-12-01T18:00:00Z")
    assert m["status"] == {"privacyStatus": "private", "selfDeclaredMadeForKids": False,
                           "publishAt": "2026-12-01T18:00:00Z"}  # fmt: skip
    assert m["snippet"]["description"].endswith("#gta6 #shorts")
    assert m["snippet"]["tags"] == ["gta6", "shorts"] and m["snippet"]["categoryId"] == "20"


def test_oauth_exchange_and_refresh(ysettings, google):
    v, ch = youtube.new_pkce()
    url = youtube.auth_url(youtube.load_client(ysettings), "http://127.0.0.1:8000/cb", "st", ch)
    assert "code_challenge_method=S256" in url and "access_type=offline" in url
    tok = youtube.exchange_code(ysettings, "code1", "http://127.0.0.1:8000/cb", v)
    assert tok["channel"]["title"] == "Meu Canal" and tok["refresh_token"] == "rt-1"
    assert google.token_requests[0]["code_verifier"] == v
    tok["expires_at"] = 0  # expired -> refreshed with the refresh token, which is kept
    store = youtube.MemoryStore(tok)
    youtube.YouTube(ysettings, store=store).channel()
    assert google.token_requests[-1]["grant_type"] == "refresh_token"
    assert store.token["refresh_token"] == "rt-1" and store.token["expires_at"] > time.time()


def test_not_configured_message(settings):
    with pytest.raises(youtube.YouTubeError, match="CLIPADOR_YOUTUBE_CLIENT_SECRETS"):
        youtube.load_client(settings)


# --- rules + the upload job -----------------------------------------------------------------------


def test_publish_rules(ysettings, engine, tmp_path):
    cid = _approved_with_final(ysettings, engine, tmp_path, with_render=False)
    with Session(engine) as s, pytest.raises(Exception, match="conta conectada"):
        publishing.request_publication(s, ysettings, cid, 0, privacy="public")
    acc = _connect(engine)
    with Session(engine) as s:
        with pytest.raises(Exception, match="Renderize o corte final"):
            publishing.request_publication(s, ysettings, cid, acc, privacy="public")
        c = s.get(Candidate, cid)
        c.status = CandidateStatus.SUGGESTED
        s.add(c)
        with pytest.raises(Exception, match="aprovados"):
            publishing.request_publication(s, ysettings, cid, acc, privacy="public")
    with pytest.raises(Exception, match="futuro"):
        publishing.parse_publish_at((utcnow() - timedelta(hours=1)).isoformat())


def test_upload_end_to_end_with_resume(ysettings, engine, tmp_path, google, monkeypatch):
    monkeypatch.setattr(youtube, "CHUNK", 1000)  # several chunks for a 3000-byte file
    acc = _connect(engine)
    cid = _approved_with_final(ysettings, engine, tmp_path)
    when = publishing.parse_publish_at((utcnow() + timedelta(days=1)).isoformat())
    with Session(engine, expire_on_commit=False) as s:
        pub = publishing.request_publication(s, ysettings, cid, acc, privacy="public",
                                             publish_at=when)  # fmt: skip
        s.commit()
        with pytest.raises(Exception, match="já foi"):
            publishing.request_publication(s, ysettings, cid, acc, privacy="public")
        job = s.get(Job, pub.job_id)
    assert job.type == "publish" and pub.title == "Capotou o carro"
    _run_stage(ysettings, engine, pub.id)
    with Session(engine) as s:
        p = s.get(Publication, pub.id)
    assert p.status == PublicationStatus.PUBLISHED and p.url.endswith("/shorts/vid123")
    assert google.uploaded == b"\x00" * 3000  # resumed correctly after the 308
    assert (
        google.meta["status"]["privacyStatus"] == "private" and google.meta["status"]["publishAt"]
    )

    with Session(engine) as s:
        assert publishing.refresh_metrics(s, ysettings, engine) == 1
        s.commit()
        p = s.get(Publication, pub.id)
    assert (p.views, p.likes, p.comments, p.privacy) == (1500, 90, 7, "public")


def test_quota_error_is_clear_and_final(ysettings, engine, tmp_path, monkeypatch):
    def handler(req):
        if req.url.path.endswith("/token"):
            return httpx.Response(200, json={"access_token": "at-9", "expires_in": 3600})
        return httpx.Response(403, json={"error": {"errors": [{"reason": "quotaExceeded"}]}})

    real = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda *a, **k: real(transport=httpx.MockTransport(handler))
    )
    acc = _connect(engine)
    youtube_tok_expired = {"access_token": "x", "refresh_token": "rt-1", "expires_at": 0}
    with Session(engine) as s:
        from clipador.models import SocialAccount

        a = s.get(SocialAccount, acc)
        a.credentials = youtube_tok_expired
        s.add(a)
        s.commit()
    cid = _approved_with_final(ysettings, engine, tmp_path)
    with Session(engine, expire_on_commit=False) as s:
        pub = publishing.request_publication(s, ysettings, cid, acc, privacy="unlisted")
        s.commit()
    with pytest.raises(youtube.YouTubeError, match="Limite diário") as e:
        _run_stage(ysettings, engine, pub.id)
    assert not e.value.retryable


def test_performance_labels_feed_calibration(ysettings, engine, tmp_path):
    cid = _approved_with_final(ysettings, engine, tmp_path)
    with Session(engine) as s:
        base = s.get(Candidate, cid)
        old = utcnow() - timedelta(days=5)
        ids = []
        for i, views in enumerate([10, 20, 30, 400, 500, 900]):
            c = Candidate(source_id=base.source_id, analysis_run_id=base.analysis_run_id,
                          rank=10 + i, start=20 + i * 40, end=50 + i * 40, peak_t=30 + i * 40,
                          score=50, status=CandidateStatus.RENDERED,
                          breakdown={"audio": 50.0})  # fmt: skip
            s.add(c)
            s.flush()
            ids.append(c.id)
            s.add(Publication(candidate_id=c.id, render_id=1, status=PublicationStatus.PUBLISHED,
                              privacy="public", views=views, created_at=old))  # fmt: skip
        s.commit()
        labels = publishing.performance_labels(s)
        from clipador.services.calibration import collect_samples

        samples = collect_samples(s, ysettings)
    assert labels[ids[0]] == "ruim" and labels[ids[-1]] == "bom" and ids[2] not in labels
    assert sum(x.label == 0 for x in samples) >= 2  # flops became negatives


# --- web ------------------------------------------------------------------------------------------


def test_publish_pages(ysettings, engine, tmp_path, google):
    from clipador.web.app import create_app

    client = TestClient(create_app(ysettings))
    r = client.get("/publicar")
    assert r.status_code == 200 and "Conectar canal do YouTube" in r.text
    r = client.get("/publicar/youtube/conectar", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith(youtube.AUTH_URL)
    state = dict(x.split("=", 1) for x in r.headers["location"].split("?")[1].split("&"))["state"]
    r = client.get(f"/publicar/youtube/callback?state={state}&code=abc", follow_redirects=False)
    assert "conectado" in r.headers["location"].lower()
    with Session(engine) as s:
        from clipador.models import SocialAccount

        acc = s.exec(select(SocialAccount)).one()
    assert acc.name == "Meu Canal" and acc.credentials["refresh_token"] == "rt-1"
    assert "Meu Canal" in client.get("/publicar").text

    cid = _approved_with_final(ysettings, engine, tmp_path)
    page = client.get(f"/publicar?candidato={cid}")
    assert "Publicar o clip" in page.text and "Capotou o carro" in page.text
    r = client.post(
        f"/candidatos/{cid}/publicar", data={"privacy": "public", "account_id": str(acc.id)}
    )
    assert r.status_code == 400 and "Confirme" in r.text
    r = client.post(f"/candidatos/{cid}/publicar",
                    data={"privacy": "public", "confirm": "on", "account_id": str(acc.id)},
                    follow_redirects=False)  # fmt: skip
    assert r.status_code == 303
    with Session(engine) as s:
        assert s.exec(select(Publication)).one().status == PublicationStatus.QUEUED
    assert "callback" not in client.get("/publicar/youtube/callback?state=x&code=y",
                                        follow_redirects=False).headers["location"]  # fmt: skip
