"""Ingest stage end-to-end with a synthetic ffmpeg video and a fake platform adapter (offline)."""

import shutil
from datetime import date
from typing import ClassVar

import pytest
from sqlmodel import Session

from clipador.models import Job, JobStatus, Source, SourceStatus, Streamer
from clipador.platforms.base import VideoMetadata
from clipador.platforms.twitch import TwitchAdapter
from clipador.services.downloader import DownloadError, DownloadErrorKind, DownloadResult
from clipador.services.sources import SourceDraft, add_channel, create_source
from clipador.worker import queue
from clipador.worker.runner import Worker
from tests.conftest import requires_ffmpeg

pytestmark = requires_ffmpeg


class FakeTwitch(TwitchAdapter):
    calls: ClassVar[dict[str, int]] = {"meta": 0, "download": 0}
    mode = "ok"
    video = None

    def fetch_metadata(self, url):
        FakeTwitch.calls["meta"] += 1
        return VideoMetadata(
            external_id="2283746281",
            title="GTA 6 — primeiro dia",
            duration_seconds=3.0,
            thumbnail_url="https://static-cdn.jtvnw.net/x.jpg",
            channel_name="Fulano",
            published_at=None,
            channel_candidates=["outro_canal"],
        )

    def download(self, url, dest, progress_cb, log_cb=None):
        FakeTwitch.calls["download"] += 1
        if FakeTwitch.mode == "private":
            raise DownloadError(
                DownloadErrorKind.PRIVATE,
                "Conteúdo privado.",
                detail="ERROR: sub-only",
                retryable=False,
            )
        dest.mkdir(parents=True, exist_ok=True)
        for p in (0, 30, 60, 100):
            progress_cb(p, None)
        out = dest / "video.mp4"
        shutil.copy(FakeTwitch.video, out)
        return DownloadResult(out, None)


@pytest.fixture
def fake_adapter(monkeypatch, synthetic_video):
    FakeTwitch.calls = {"meta": 0, "download": 0}
    FakeTwitch.mode = "ok"
    FakeTwitch.video = synthetic_video
    monkeypatch.setattr(
        "clipador.pipeline.stages.ingest.get_adapter",
        lambda platform, ytdlp=None: FakeTwitch(ytdlp),
    )
    return FakeTwitch


def _setup(engine, start=None, end=None) -> tuple[int, int]:
    with Session(engine, expire_on_commit=False) as s:
        st = Streamer(
            display_name="Fulano", permission_confirmed=True, permission_date=date.today()
        )
        s.add(st)
        s.flush()
        add_channel(s, st, "twitch", "fulano")
        src, _ = create_source(
            s,
            SourceDraft(
                url="https://www.twitch.tv/videos/2283746281",
                streamer_id=st.id,
                start_offset=start,
                end_offset=end,
            ),
        )
        job = queue.enqueue(s, "ingest", src.id)
        s.commit()
        return src.id, job.id


def _requeue(engine, job_id):
    with Session(engine) as s:
        job = s.get(Job, job_id)
        job.status = JobStatus.QUEUED
        s.add(job)
        s.commit()


def _run_one(settings, engine):
    w = Worker(settings, engine)
    job = queue.claim(engine, w.worker_id)
    assert job is not None
    w.execute(job.id)
    return w


def test_ingest_downloads_probes_and_saves(settings, engine, fake_adapter):
    source_id, job_id = _setup(engine, start=1.0, end=2.5)
    _run_one(settings, engine)

    with Session(engine) as s:
        src = s.get(Source, source_id)
        job = s.get(Job, job_id)
    assert job.status == JobStatus.DONE and job.progress == 100
    assert src.status == SourceStatus.READY
    assert src.title == "GTA 6 — primeiro dia"
    assert src.duration_seconds == pytest.approx(3.0, abs=0.2)
    assert src.file_path == f"vods/{source_id}/video.mp4"
    assert (settings.data_dir / src.file_path).is_file()
    assert src.thumbnail_path and (settings.data_dir / src.thumbnail_path).is_file()
    assert src.media_info["width"] == 320
    assert (src.start_offset, src.end_offset) == (1.0, 2.5)  # kept, file not cut
    assert src.warning and "não bate" in src.warning  # channel mismatch -> warning only
    assert "Pronto." in job.log


def test_ingest_is_idempotent(settings, engine, fake_adapter):
    source_id, job_id = _setup(engine)
    _run_one(settings, engine)
    assert fake_adapter.calls == {"meta": 1, "download": 1}

    # Re-run the same stage (e.g. reprocess): is_done() skips everything.
    _requeue(engine, job_id)
    _run_one(settings, engine)
    assert fake_adapter.calls == {"meta": 1, "download": 1}

    # Source not marked ready but file intact on disk (crash after download): skip download only.
    with Session(engine) as s:
        src = s.get(Source, source_id)
        src.status = SourceStatus.DOWNLOADING
        s.add(src)
        s.commit()
    _requeue(engine, job_id)
    _run_one(settings, engine)
    assert fake_adapter.calls == {"meta": 1, "download": 1}
    with Session(engine) as s:
        assert s.get(Source, source_id).status == SourceStatus.READY


def test_ingest_redownloads_corrupt_file(settings, engine, fake_adapter):
    source_id, _job_id = _setup(engine)
    dest = settings.vods_dir / str(source_id)
    dest.mkdir(parents=True)
    (dest / "video.mp4").write_bytes(b"truncated garbage")
    _run_one(settings, engine)
    assert fake_adapter.calls["download"] == 1
    with Session(engine) as s:
        assert s.get(Source, source_id).status == SourceStatus.READY


def test_ingest_private_video_fails_without_retry(settings, engine, fake_adapter):
    fake_adapter.mode = "private"
    source_id, job_id = _setup(engine)
    _run_one(settings, engine)
    with Session(engine) as s:
        job = s.get(Job, job_id)
        src = s.get(Source, source_id)
    assert job.status == JobStatus.FAILED and job.attempts == 1
    assert src.status == SourceStatus.FAILED and src.error == "Conteúdo privado."
