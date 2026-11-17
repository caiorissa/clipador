"""Live capture (step 5E) without internet or GPU: a synthetic video played as a "live" through
the real capture process (ffmpeg segments), the incremental analysis with a fake transcriber,
auto-clip, the final VOD, renders from segments, retention, supervision and the web pages."""

import itertools
import subprocess
import time
import wave
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from sqlmodel import Session, select

from clipador.detection.candidates import CandidateResult
from clipador.errors import ValidationError
from clipador.models import (
    AnalysisRun,
    Candidate,
    CandidateStatus,
    Job,
    JobStatus,
    LiveSegment,
    LiveSession,
    LiveStatus,
    Render,
    SegmentStatus,
    Source,
    Streamer,
    StreamerChannel,
)
from clipador.pipeline.stages import live as live_stage
from clipador.platforms import live as plive
from clipador.services import live, rendering
from clipador.services import transcriber as tx
from clipador.services.live_capture import (
    Capture,
    TwitchChatReader,
    read_segment_list,
    segment_args,
)
from clipador.utils import utcnow
from clipador.worker.runner import Worker
from tests.conftest import requires_ffmpeg

SR = 16000
LENGTH = 30
BURSTS = (6.0, 22.0)


# --- fixtures ---------------------------------------------------------------------------------


@pytest.fixture
def lsettings(settings):
    settings.live_heartbeat_seconds = 0.05
    settings.live_reconnect_wait_seconds = 0
    settings.live_min_free_gb = 0
    settings.live_retention_hours = 0
    settings.live_twitch_chat = False  # never touch the network in tests
    return settings


def _streamer(engine, permission=True, handle="fulano") -> int:
    with Session(engine) as s:
        st = Streamer(display_name="Fulano", permission_confirmed=permission,
                      permission_date=date.today())  # fmt: skip
        s.add(st)
        s.flush()
        s.add(StreamerChannel(streamer_id=st.id, platform="twitch", channel_handle=handle))
        s.commit()
        return st.id


@pytest.fixture(scope="module")
def live_video(tmp_path_factory) -> Path:
    """30 s video (keyframe every second) whose audio has two loud bursts."""
    d = tmp_path_factory.mktemp("livemedia")
    wav = d / "a.wav"
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.02, LENGTH * SR)
    for t in BURSTS:
        x[int(t * SR) : int((t + 2) * SR)] *= 15
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    out = d / "live.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=15", "-i", str(wav),
             "-t", str(LENGTH), "-c:v", "libx264", "-g", "15", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-shortest", str(out)],
            check=True,
        )  # fmt: skip
    except FileNotFoundError:
        pytest.skip("ffmpeg não instalado")
    return out


class FakeLiveTx:
    """Words relative to the segment WAV: an excited phrase 1 s before each burst."""

    def __init__(self, settings, log_cb):
        self.device, self.compute_type, self.detected_language = "fake", "none", "pt"
        self.settings = settings

    def transcribe(self, audio_path, language, progress_cb):
        index = int(audio_path.stem.split("_")[1])
        with Session(self.engine) as s:
            seg = s.exec(select(LiveSegment).where(LiveSegment.index == index)).first()
        a, b = seg.start_offset_s, seg.end_offset_s
        progress_cb(1.0)
        return [
            tx.Word(t - 1 + 0.3 * i - a, t - 0.8 + 0.3 * i - a, w, 0.9)
            for t in BURSTS
            for i, w in enumerate(["meu", "deus", "olha", "isso"])
            if a <= t - 1 + 0.3 * i < b
        ]

    def close(self):
        pass


@pytest.fixture
def fake_tx(monkeypatch, engine):
    FakeLiveTx.engine = engine
    monkeypatch.setattr(tx, "make_transcriber", FakeLiveTx)


def _drain(worker: Worker, timeout=120.0):
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
            for j in pending:
                j.run_after = j.created_at
                s.add(j)
            s.commit()
        if not pending:
            return
    raise AssertionError("jobs did not finish")


def _simulate(engine, settings, video, **opts) -> int:
    st = _streamer(engine)
    with Session(engine, expire_on_commit=False) as s:
        ls = live.create_session(
            s, settings, streamer_id=st, sim_file=video, sim_speed=opts.pop("speed", 30.0),
            options=live.LiveOptions(opts.pop("segment", 10), opts.pop("auto_clip", True),
                                     opts.pop("min_score", 0.0)),
        )  # fmt: skip
        s.commit()
    assert Capture(engine, settings, ls.id).run() == 0
    return ls.id


# --- pure parts -------------------------------------------------------------------------------


def test_parse_live_urls():
    t = plive.parse_live_url("https://www.twitch.tv/Gaules")
    assert t.platform == "twitch" and t.channel_handle == "gaules" and not t.supports_from_start
    assert plive.parse_live_url("https://kick.com/fulano").platform == "kick"
    yt = plive.parse_live_url("https://www.youtube.com/@canal/live")
    assert yt.platform == "youtube" and yt.supports_from_start
    with pytest.raises(plive.InvalidUrlError):
        plive.parse_live_url("https://www.twitch.tv/videos/123")
    assert plive.channel_live_url("twitch", "fulano") == "https://www.twitch.tv/fulano"


def test_live_status_output():
    assert plive.parse_status_output(0, "True|is_live|abc|Título", "").live
    off = plive.parse_status_output(1, "", "ERROR: fulano is offline")
    assert not off.live and "não está ao vivo" in off.reason


def test_stream_input_args_only_reconnect_on_http():
    from clipador.services.live_capture import stream_input_args

    assert "-reconnect" in stream_input_args("https://x.m3u8")
    assert stream_input_args("/tmp/file.ts") == ["-i", "/tmp/file.ts"]


def test_segment_list_and_args(tmp_path):
    (tmp_path / "segments.csv").write_text(
        "seg_00003.ts,0.000000,10.010000\nseg_00004.ts,10.010000,12.5\nlixo\n", "utf-8"
    )
    rows = read_segment_list(tmp_path / "segments.csv")
    assert [(r.index, round(r.duration, 2)) for r in rows] == [(3, 10.01), (4, 2.49)]
    args = segment_args(["-i", "x"], tmp_path, 300, 7)
    assert args[args.index("-segment_start_number") + 1] == "7" and "copy" in args


def test_twitch_irc_parse():
    msg = ":fulano!fulano@fulano.tmi.twitch.tv PRIVMSG #canal :KEKW que isso"
    assert TwitchChatReader.parse(msg) == ("fulano", "KEKW que isso")
    assert TwitchChatReader.parse(":tmi.twitch.tv 001 justinfan1 :Welcome") is None


# --- sessions -----------------------------------------------------------------------------------


def test_create_session_rules(engine, lsettings, tmp_path):
    no_perm = _streamer(engine, permission=False, handle="semperm")
    with Session(engine) as s, pytest.raises(ValidationError, match="autoriza"):
        live.create_session(s, lsettings, streamer_id=no_perm,
                            url="https://www.twitch.tv/semperm")  # fmt: skip
    ok = _streamer(engine, handle="fulano2")
    with Session(engine) as s:
        # streamer found by the channel of the link
        ls = live.create_session(s, lsettings, streamer_id=None,
                                 url="https://www.twitch.tv/fulano2")  # fmt: skip
        s.commit()
        src = s.get(Source, ls.source_id)
        assert src.is_live and src.kind == "live" and src.streamer_id == ok
        with pytest.raises(ValidationError, match="Já existe uma live"):
            live.create_session(s, lsettings, streamer_id=ok, url="https://www.twitch.tv/fulano2")
        with pytest.raises(ValidationError, match="não encontrado"):
            live.create_session(s, lsettings, streamer_id=ok, sim_file=tmp_path / "nada.mp4")


def test_merge_keeps_best_and_respects_decisions(engine, lsettings):
    st = _streamer(engine)
    with Session(engine, expire_on_commit=False) as s:
        ls = live.create_session(s, lsettings, streamer_id=st, url="https://www.twitch.tv/fulano")
        s.commit()

        def r(a, b, score):
            return CandidateResult(a, b, (a + b) / 2, score, {}, "x")

        live_stage.merge_into_session(s, ls, [r(10, 40, 50)], 0, 60)
        # overlapping, better: replaces limits; a weaker one that continues extends the end
        touched = live_stage.merge_into_session(s, ls, [r(15, 45, 70)], 0, 60)
        assert len(touched) == 1 and (touched[0].start, touched[0].score) == (15, 70)
        live_stage.merge_into_session(s, ls, [r(30, 70, 40)], 0, 60)
        c = s.exec(select(Candidate).where(Candidate.live_session_id == ls.id)).one()
        assert c.end == 70 and c.score == 70
        # a decided candidate is never changed; peaks before new_from are ignored
        c.status = CandidateStatus.APPROVED
        s.add(c)
        assert live_stage.merge_into_session(s, ls, [r(20, 50, 99)], 0, 60) == []
        assert live_stage.merge_into_session(s, ls, [r(100, 130, 99)], 200, 60) == []


def test_supervise_restarts_then_gives_up(engine, lsettings):
    st = _streamer(engine)
    with Session(engine, expire_on_commit=False) as s:
        ls = live.create_session(s, lsettings, streamer_id=st, url="https://www.twitch.tv/fulano")
        ls.heartbeat_at = utcnow() - timedelta(minutes=5)
        s.add(ls)
        s.commit()
    spawned = []
    notes = live.supervise(engine, lsettings, spawn=lambda st_, i: spawned.append(i) or 4242)
    assert spawned == [ls.id] and "reiniciado" in notes[0]
    with Session(engine) as s:
        row = s.get(LiveSession, ls.id)
        assert row.restarts == 1 and row.pid == 4242
        row.restarts = lsettings.live_max_restarts
        row.heartbeat_at = utcnow() - timedelta(minutes=5)
        s.add(row)
        s.commit()
    live.supervise(engine, lsettings, spawn=lambda *a: 1)
    with Session(engine) as s:
        assert s.get(LiveSession, ls.id).status == LiveStatus.FAILED
        assert s.exec(select(Job).where(Job.type == "live_finalize")).first() is not None


def test_auto_watch_starts_one_session_per_streamer(engine, lsettings):
    st = _streamer(engine, handle="vigiado")
    with Session(engine) as s:
        row = s.get(Streamer, st)
        row.auto_watch = True
        s.add(row)
        s.commit()
    started = []
    notes = live.auto_watch(engine, lsettings, check=lambda url: "vigiado" in url,
                            spawn=lambda st_, i: started.append(i) or 1)  # fmt: skip
    assert len(started) == 1 and "entrou ao vivo" in notes[0]
    assert live.auto_watch(engine, lsettings, check=lambda url: True, spawn=lambda *a: 1) == []


# --- the real thing: capture -> analysis -> auto-clip -> VOD ------------------------------------


@requires_ffmpeg
def test_simulated_live_end_to_end(engine, lsettings, live_video, fake_tx):
    live_id = _simulate(engine, lsettings, live_video, segment=10, min_score=0.0)
    with Session(engine) as s:
        segs = live.segments(s, live_id)
        ls = s.get(LiveSession, live_id)
    assert len(segs) >= 3 and all(g.status == SegmentStatus.CLOSED for g in segs)
    # contiguous global timeline
    for a, b in itertools.pairwise(segs):
        assert b.start_offset_s == pytest.approx(a.end_offset_s)
    assert segs[-1].end_offset_s == pytest.approx(LENGTH, abs=1.5)
    assert ls.status == LiveStatus.STOPPED

    _drain(Worker(lsettings, engine))
    with Session(engine) as s:
        cands = s.exec(select(Candidate).where(Candidate.live_session_id == live_id)).all()
        src = s.get(Source, ls.source_id)
        renders = s.exec(select(Render)).all()
        run = s.get(AnalysisRun, ls.run_id)
        segs = live.segments(s, live_id)
    assert cands, "the bursts should become candidates"
    peaks = sorted(c.peak_t for c in cands)
    assert any(abs(p - BURSTS[1] - 1) < 6 for p in peaks)  # global time, not segment time
    assert all(c.auto_clipped and c.status == CandidateStatus.SUGGESTED for c in cands)
    assert renders and all(r.kind == "preview" for r in renders)  # never approved/finalized
    assert run.candidate_count == len(cands)
    # the live became a normal VOD; analyzed segments were cleaned up
    assert not src.is_live and src.kind == "vod" and src.file_path
    assert lsettings.abs_path(src.file_path).is_file()
    assert all(g.status == SegmentStatus.EXPIRED and g.path is None for g in segs)
    words = live.load_words(lsettings, live_id)
    assert words and words[0][2] == "meu"


@requires_ffmpeg
def test_render_plan_reads_segments_while_live(engine, lsettings, live_video, fake_tx):
    live_id = _simulate(engine, lsettings, live_video, segment=10, auto_clip=False)
    with Session(engine) as s:  # analysis done, but no VOD yet
        for j in s.exec(select(Job).where(Job.type == "live_finalize")).all():
            s.delete(j)
        s.commit()
    _drain(Worker(lsettings, engine))
    with Session(engine, expire_on_commit=False) as s:
        segs = live.segments(s, live_id)
        assert all(g.status == SegmentStatus.ANALYZED for g in segs)
        src = s.get(Source, s.get(LiveSession, live_id).source_id)
        c = Candidate(source_id=src.id, analysis_run_id=s.get(LiveSession, live_id).run_id,
                      rank=1, start=segs[1].end_offset_s - 3, end=segs[2].start_offset_s + 4,
                      peak_t=segs[2].start_offset_s, score=50, live_session_id=live_id)  # fmt: skip
        s.add(c)
        s.commit()
        plan = rendering.plan_render(s, engine, lsettings, c, "preview")
    assert plan.window is not None and plan.seek == pytest.approx(c.start - segs[1].start_offset_s)
    video = live.ensure_window_video(plan.window, lsettings.ffmpeg_path)  # 2 segments joined
    from clipador.services import ffmpeg

    span = segs[2].end_offset_s - segs[1].start_offset_s
    assert ffmpeg.probe(video).duration == pytest.approx(span, abs=1.0)
    assert [w.text for w in plan.words][:2] == ["meu", "deus"]  # captions from the live words


@requires_ffmpeg
def test_disk_guard_stops_capture(engine, lsettings, live_video, monkeypatch):
    lsettings.live_min_free_gb = 10**9
    live_id = _simulate(engine, lsettings, live_video, speed=1.0)
    with Session(engine) as s:
        ls = s.get(LiveSession, live_id)
    assert ls.status == LiveStatus.STOPPED and "GB" in (ls.warning or "")


@requires_ffmpeg
def test_reconnects_when_stream_drops(engine, lsettings, live_video):
    """A non-simulated session whose 'stream' (a local file) ends twice: the first drop is a
    reconnect (still live), the second time the channel is offline -> the live is over. The
    global clock keeps counting across the reconnect."""
    lsettings.live_max_reconnect_failures = 1
    st = _streamer(engine)
    with Session(engine, expire_on_commit=False) as s:
        ls = live.create_session(s, lsettings, streamer_id=st, url="https://www.twitch.tv/fulano",
                                 options=live.LiveOptions(10))  # fmt: skip
        s.commit()
    answers = iter([True, False])
    cap = Capture(engine, lsettings, ls.id, resolve_url=lambda url: str(live_video),
                  is_live=lambda url: next(answers))  # fmt: skip
    assert cap.run() == 0
    with Session(engine) as s:
        row = s.get(LiveSession, ls.id)
        segs = live.segments(s, ls.id)
    assert row.reconnects == 1 and row.status == LiveStatus.STOPPED
    assert segs[-1].end_offset_s == pytest.approx(2 * LENGTH, abs=3)
    assert [g.index for g in segs] == list(range(len(segs)))


# --- web ------------------------------------------------------------------------------------------


@pytest.fixture
def client(lsettings, engine, monkeypatch):
    from fastapi.testclient import TestClient

    from clipador.web.app import create_app

    monkeypatch.setattr(live, "spawn_capture", lambda settings, live_id: 999)
    return TestClient(create_app(lsettings))


def test_live_pages(client, engine):
    _streamer(engine, handle="webfulano")
    r = client.get("/ao-vivo")
    assert r.status_code == 200 and "Começar a gravar" in r.text and "Ao vivo" in r.text
    r = client.post("/ao-vivo", data={"url": "https://www.twitch.tv/webfulano",
                                      "segment_seconds": "60", "min_score": "65",
                                      "auto_clip": "on"}, follow_redirects=False)  # fmt: skip
    assert r.status_code == 303
    with Session(engine) as s:
        ls = s.exec(select(LiveSession)).one()
        assert ls.pid == 999 and ls.min_score == 65 and ls.segment_seconds == 60
    assert "AO VIVO" in client.get("/ao-vivo/sessoes").text
    r = client.post(f"/ao-vivo/{ls.id}/parar")
    assert "parando" in r.text
    with Session(engine) as s:
        assert s.get(LiveSession, ls.id).stop_requested
    bad = client.post("/ao-vivo", data={"url": "https://www.twitch.tv/videos/1"})
    assert bad.status_code == 400
    assert client.get("/ao-vivo/alertas.json").json()["pending"] == 0


def test_review_live_filter_puts_auto_clips_first(client, engine, lsettings):
    st = _streamer(engine)
    with Session(engine, expire_on_commit=False) as s:
        ls = live.create_session(s, lsettings, streamer_id=st, url="https://www.twitch.tv/fulano")
        for i, auto in enumerate((False, True)):
            s.add(Candidate(source_id=ls.source_id, analysis_run_id=ls.run_id, rank=i + 1,
                            start=10 + 50 * i, end=40 + 50 * i, peak_t=20 + 50 * i, score=50,
                            live_session_id=ls.id, auto_clipped=auto))  # fmt: skip
        s.commit()
    from clipador.services import review

    with Session(engine) as s:
        f = review.ReviewFilters(status="ao_vivo")
        first = review.next_pending(s, f)
        assert first.auto_clipped
    assert client.get("/ao-vivo/alertas.json").json()["pending"] == 1
    r = client.get("/revisao?status=ao_vivo")
    assert r.status_code == 200 and "AO VIVO" in r.text
