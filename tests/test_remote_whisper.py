"""Sharing the work with another computer: the transcription server (the PC's GPU) and the
client used by the main Clipador, with the local fallback when the PC is off."""

import wave

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from clipador.models import Job
from clipador.services import transcriber as tx
from clipador.services import transcription_server as srv


class FakeWhisper:
    detected_language = "pt"
    calls = 0

    def transcribe(self, path, language, progress_cb):
        FakeWhisper.calls += 1
        return [tx.Word(0.1, 0.4, "olha", 0.9), tx.Word(0.5, 0.9, "isso", 0.8)]


def _wav(path):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
    return path


@pytest.fixture(autouse=True)
def _reset_remote_state():
    tx._REMOTE_DOWN_UNTIL[0] = 0.0
    FakeWhisper.calls = 0
    yield
    tx._REMOTE_DOWN_UNTIL[0] = 0.0


@pytest.fixture
def server(settings):
    settings.transcription_server_token = "segredo"
    return TestClient(srv.create_app(settings, FakeWhisper()))


def test_server_needs_the_token_and_a_wav(server, tmp_path):
    assert server.get("/health").status_code == 401
    ok = server.get("/health", headers={"Authorization": "Bearer segredo"})
    assert ok.status_code == 200 and ok.json()["ok"]
    bad = server.post("/transcribe", content=b"oi", headers={"Authorization": "Bearer segredo"})
    assert bad.status_code == 400
    r = server.post("/transcribe", content=_wav(tmp_path / "a.wav").read_bytes(),
                    headers={"Authorization": "Bearer segredo"})  # fmt: skip
    assert r.json() == {
        "language": "pt",
        "words": [[0.1, 0.4, "olha", 0.9], [0.5, 0.9, "isso", 0.8]],
    }


def test_token_is_generated_once(settings):
    a = srv.server_token(settings)
    assert a and srv.server_token(settings) == a


def test_client_uses_the_other_computer(settings, server, tmp_path, monkeypatch):
    settings.remote_whisper_url, settings.remote_whisper_token = "http://pc:8765", "segredo"

    def fake_post(url, content=None, params=None, timeout=None, headers=None):
        return server.post(url.replace("http://pc:8765", ""), content=content, params=params,
                           headers=headers)  # fmt: skip

    monkeypatch.setattr(httpx, "post", fake_post)
    t = tx.make_transcriber(settings, print)
    assert isinstance(t, tx.RemoteTranscriber)
    words = t.transcribe(_wav(tmp_path / "a.wav"), "pt", lambda f: None)
    assert [w.word for w in words] == ["olha", "isso"] and t.detected_language == "pt"


def test_falls_back_to_this_machine_when_the_pc_is_off(settings, tmp_path, monkeypatch):
    settings.remote_whisper_url, settings.remote_whisper_token = "http://127.0.0.1:9", "x"
    monkeypatch.setattr(tx, "_local_transcriber", lambda s, log: FakeWhisper())
    t = tx.make_transcriber(settings, print)
    words = t.transcribe(_wav(tmp_path / "a.wav"), "pt", lambda f: None)
    assert [w.word for w in words] == ["olha", "isso"] and FakeWhisper.calls == 1
    assert not tx.remote_up(settings)  # marked down for a while: no waiting on every chunk


def test_worker_releases_transcriptions_from_the_local_gpu(engine):
    from clipador.worker import queue

    with Session(engine) as s:
        j = queue.enqueue(s, "transcribe", None, {}, dedupe=False)
        r = queue.enqueue(s, "render", None, {"render_ids": [1]}, dedupe=False)
        s.commit()
        assert j.needs_gpu and r.needs_gpu
    assert queue.release_from_gpu(engine, ("transcribe", "live_analyze")) == 1
    with Session(engine) as s:
        assert not s.get(Job, j.id).needs_gpu and s.get(Job, r.id).needs_gpu
