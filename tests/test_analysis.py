"""Analysis chain end-to-end with a fake transcriber (no GPU/model) and synthetic audio."""

import time
import wave
from datetime import date
from typing import ClassVar

import numpy as np
import pytest
from sqlmodel import Session, select

from clipador.detection.params import DetectParams
from clipador.errors import ClipadorError
from clipador.models import (
    AnalysisRun,
    ArtifactStatus,
    Candidate,
    Job,
    JobStatus,
    Source,
    SourceStatus,
    StageArtifact,
    Streamer,
)
from clipador.services import analysis
from clipador.services import transcriber as tx
from clipador.worker.runner import Worker
from tests.conftest import requires_ffmpeg

pytestmark = requires_ffmpeg

SR = 16000
DURATION = 200
BURSTS = [40.0, 150.0]
# Absolute (file-time) words the "speaker" says: a hype phrase right before each burst.
SPOKEN = [
    (t - 3 + 0.4 * i, t - 3 + 0.4 * i + 0.3, w)
    for t in BURSTS
    for i, w in enumerate(["meu", "deus", "olha", "a", "policia"])
] + [(100.0 + 0.4 * i, 100.3 + 0.4 * i, w) for i, w in enumerate(["vamos", "pegar", "o", "carro"])]


class FakeTranscriber:
    calls: ClassVar[list[int]] = []
    fail_on: ClassVar[set[int]] = set()
    audio_offset = 0.0  # start_offset of the source (wav starts there)

    def __init__(self, settings, log_cb):
        self.settings = settings
        self.device, self.compute_type = "fake", "none"

    def transcribe(self, audio_path, language, progress_cb):
        index = int(audio_path.stem.split("_")[1])
        if index in FakeTranscriber.fail_on:
            FakeTranscriber.fail_on.discard(index)
            raise ClipadorError("GPU caiu (simulado)", retryable=True)
        FakeTranscriber.calls.append(index)
        off = FakeTranscriber.audio_offset
        chunk = tx.plan_chunks(DURATION - off, self.settings.transcribe_chunk_seconds,
                               self.settings.transcribe_chunk_overlap_seconds)[index]  # fmt: skip
        progress_cb(0.5)
        base = chunk.read_start + off
        return [
            tx.Word(s - base, e - base, w, 0.9)
            for s, e, w in SPOKEN
            if chunk.read_start <= s - off < chunk.read_end
        ]

    def close(self):
        pass


@pytest.fixture
def fake_tx(monkeypatch):
    FakeTranscriber.calls = []
    FakeTranscriber.fail_on = set()
    FakeTranscriber.audio_offset = 0.0
    monkeypatch.setattr(tx, "make_transcriber", FakeTranscriber)
    return FakeTranscriber


@pytest.fixture
def asettings(settings, tmp_path):
    settings.transcribe_chunk_seconds = 60
    settings.transcribe_chunk_overlap_seconds = 2
    settings.twitch_downloader_path = str(tmp_path / "missing_TwitchDownloaderCLI.exe")
    return settings


def _write_wav(path, seconds=DURATION, bursts=BURSTS):
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.02, seconds * SR)
    for t in bursts:
        x[int(t * SR) : int((t + 2) * SR)] *= 15
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def _source(engine, settings, start_offset=None) -> int:
    f = settings.vods_dir / "1" / "video.wav"
    _write_wav(f)
    with Session(engine, expire_on_commit=False) as s:
        st = Streamer(
            display_name="Fulano", permission_confirmed=True, permission_date=date.today()
        )
        s.add(st)
        s.flush()
        src = Source(
            streamer_id=st.id, platform="twitch", kind="vod", external_id="999", url="u",
            status=SourceStatus.READY, file_path=settings.rel_path(f), duration_seconds=DURATION,
            start_offset=start_offset,
        )  # fmt: skip
        s.add(src)
        s.commit()
        return src.id


def _drain(worker: Worker, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        worker.tick()
        time.sleep(0.02)
        if worker.active_job_ids():
            continue
        with Session(worker.engine) as s:
            pending = s.exec(
                select(Job).where(Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]))
            ).all()
            # skip retry backoff in tests
            for j in pending:
                j.run_after = j.created_at
                s.add(j)
            s.commit()
        if not pending:
            return
    raise AssertionError("timeout")


def _analyze(engine, source_id, detect_params=None):
    with Session(engine) as s:
        analysis.start_analysis(s, source_id, detect_params)
        s.commit()


def test_full_chain_produces_candidates_and_skips_chat(asettings, engine, fake_tx):
    sid = _source(engine, asettings)
    _analyze(engine, sid)
    _drain(Worker(asettings, engine))
    with Session(engine) as s:
        jobs = {j.type: j for j in s.exec(select(Job)).all()}
        arts = {a.stage: a for a in s.exec(select(StageArtifact)).all()}
        run = s.exec(select(AnalysisRun)).one()
        cands = s.exec(select(Candidate).order_by(Candidate.rank)).all()
    assert [jobs[k].status for k in analysis.ANALYSIS_CHAIN] == [JobStatus.DONE] * 4
    assert arts["fetch_chat"].status == ArtifactStatus.SKIPPED
    assert "TwitchDownloaderCLI" in arts["fetch_chat"].message
    assert arts["transcribe"].summary["words"] == len(SPOKEN)  # no duplicates at chunk seams
    assert not run.has_chat and "chat" not in run.params["effective_weights"]
    assert len(cands) >= 2
    for t in BURSTS:
        assert any(c.start <= t + 1 <= c.end for c in cands[:2])
    assert "policia" in cands[0].transcript_snippet
    assert (asettings.work_dir / str(sid) / "transcript.json.gz").is_file()
    assert (asettings.work_dir / str(sid) / "features.npz").is_file()


def test_reanalyze_is_cached_and_never_retranscribes(asettings, engine, fake_tx):
    sid = _source(engine, asettings)
    _analyze(engine, sid)
    w = Worker(asettings, engine)
    _drain(w)
    calls = list(fake_tx.calls)

    # same params -> detect is a no-op (no new run)
    with Session(engine) as s:
        analysis.reanalyze(s, sid, DetectParams.from_preset("equilibrado"))
        s.commit()
    _drain(w)
    with Session(engine) as s:
        assert len(s.exec(select(AnalysisRun)).all()) == 1

    # different weights -> new run, from cached features, no transcription
    params = DetectParams.from_preset("foco_audio")
    with Session(engine) as s:
        analysis.reanalyze(s, sid, params)
        s.commit()
    t0 = time.monotonic()
    _drain(w)
    assert time.monotonic() - t0 < 10
    with Session(engine) as s:
        runs = s.exec(select(AnalysisRun).order_by(AnalysisRun.id)).all()
        last_detect = s.exec(
            select(Job).where(Job.type == "detect").order_by(Job.id.desc())
        ).first()
    assert len(runs) == 2 and runs[1].params["candidates"]["preset"] == "foco_audio"
    assert fake_tx.calls == calls
    assert "Features em cache" in last_detect.log

    # full "Analisar" again: every stage is already done (chat retried since it was skipped)
    _analyze(engine, sid, params.model_dump())
    _drain(w)
    assert fake_tx.calls == calls


def test_transcription_resumes_from_last_chunk(asettings, engine, fake_tx):
    sid = _source(engine, asettings)
    fake_tx.fail_on = {2}
    _analyze(engine, sid)
    _drain(Worker(asettings, engine))
    # chunk 2 failed once (retryable) -> job retried and resumed: 0,1 never re-done
    assert fake_tx.calls == [0, 1, 2, 3]
    with Session(engine) as s:
        tr = s.exec(select(Job).where(Job.type == "transcribe")).one()
        assert tr.status == JobStatus.DONE and tr.attempts == 2
        assert "Retomando: 2/4 blocos" in tr.log
        assert s.exec(select(AnalysisRun)).first() is not None


def test_times_are_relative_to_original_video_with_offset(asettings, engine, fake_tx):
    fake_tx.audio_offset = 50.0
    sid = _source(engine, asettings, start_offset=50.0)  # analyze only 50 s .. end
    _analyze(engine, sid)
    _drain(Worker(asettings, engine))
    with Session(engine) as s:
        cands = s.exec(select(Candidate).order_by(Candidate.rank)).all()
    assert cands and all(c.start >= 50.0 for c in cands)
    top = cands[0]
    assert top.start <= BURSTS[1] + 1 <= top.end  # same instant as in the original video
    assert "policia" in top.transcript_snippet
    assert not any(c.start <= BURSTS[0] + 1 <= c.end for c in cands)  # before the range


def test_analyze_requires_downloaded_source(asettings, engine):
    sid = _source(engine, asettings)
    with Session(engine) as s:
        src = s.get(Source, sid)
        src.status = SourceStatus.DOWNLOADING
        s.add(src)
        s.commit()
        with pytest.raises(Exception, match="download"):
            analysis.start_analysis(s, sid)


def test_plan_and_merge_chunks_dedupes_seams():
    chunks = tx.plan_chunks(125, 60, 2)
    assert [(c.start, c.end) for c in chunks] == [(0, 60), (60, 125)]  # tiny tail merged
    assert chunks[1].read_start == 58 and chunks[0].read_end == 62
    a = [tx.Word(57.0, 57.5, "olha"), tx.Word(59.8, 60.3, "polícia"), tx.Word(60.5, 61, "corre")]
    b = [tx.Word(59.85, 60.3, "policia"), tx.Word(60.52, 61, "corre"), tx.Word(70, 71, "fim")]
    merged = tx.merge_chunk_words([(chunks[0], a), (chunks[1], b)])
    assert [w.word for w in merged] == ["olha", "polícia", "corre", "fim"]
