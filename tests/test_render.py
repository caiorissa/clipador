"""Rendering: pure builders + real ffmpeg renders of a synthetic video (no network)."""

import json
import subprocess
import time
from datetime import date
from itertools import pairwise

import pytest
from sqlmodel import Session, select

from clipador.detection.params import DetectParams
from clipador.models import (
    AnalysisRun,
    Candidate,
    CandidateStatus,
    Job,
    JobStatus,
    RenderedClip,
    Source,
    SourceStatus,
    StageArtifact,
    Streamer,
)
from clipador.render import filters
from clipador.render.styles import STYLES, get_style
from clipador.render.subtitles import (
    align_edited_text,
    ass_time,
    build_ass,
    chunk_words,
    clean_word,
    words_in_window,
)
from clipador.services import analysis, ffmpeg
from clipador.services.artifacts import save_artifact
from clipador.services.transcriber import Word, save_transcript
from clipador.worker.runner import Worker
from tests.conftest import requires_ffmpeg

WORDS = [(1.0, 1.3, "olha"), (1.35, 1.6, "a"), (1.65, 2.2, "polícia!"), (3.5, 3.8, "corre"),
         (3.85, 4.2, "{corre}"), (4.25, 4.9, "\\Ncorre"), (6.0, 6.4, "meu"), (6.45, 7.0, "deus")]  # fmt: skip


# --- pure --------------------------------------------------------------------------------------


def test_ass_helpers():
    assert ass_time(0) == "0:00:00.00" and ass_time(3723.456) == "1:02:03.46"
    assert clean_word("{\\b1}oi") == "(/b1)oi"


def test_words_in_window_relative_and_clipped():
    w = words_in_window(WORDS, 3.6, 6.2)
    assert w[0] == (0.0, 0.2, "corre") and w[-1][2] == "meu" and w[-1][1] == pytest.approx(2.6)


def test_chunking_breaks_on_sentence_gap_and_size():
    chunks = chunk_words(words_in_window(WORDS, 0, 10), max_words=3)
    texts = [[w[2] for w in c] for c in chunks]
    assert texts[0] == ["olha", "a", "polícia!"]  # sentence end closes the chunk
    assert texts[1] == ["corre", "(corre)"]  # braces neutralized; 18-char limit splits here
    assert texts[2] == ["/Ncorre"]  # backslash neutralized (no ASS line break injection)
    assert texts[3] == ["meu", "deus"]  # gap > 0.6 s starts a new chunk


def test_align_edited_text():
    orig = words_in_window(WORDS, 0, 10)[:3]
    same = align_edited_text(orig, "Olha a POLÍCIA")
    assert [w[2] for w in same] == ["Olha", "a", "POLÍCIA"] and same[2][:2] == orig[2][:2]
    diff = align_edited_text(orig, "olha só a viatura chegando")
    assert len(diff) == 5 and diff[0][0] == orig[0][0] and diff[-1][1] <= orig[-1][1]
    assert all(a[1] <= b[0] for a, b in pairwise(diff))
    assert align_edited_text(orig, "   ") == []


def test_build_ass_highlight_title_and_uppercase():
    ass = build_ass(words_in_window(WORDS, 0, 10), 10.0, get_style("classico"),
                    filters.LAYOUT_BLUR, "Fugiu da polícia")  # fmt: skip
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert "Dialogue: 1,0:00:00.00,0:00:10.00,Title,,0,0,0,,Fugiu da polícia" in ass
    lines = [ln for ln in ass.splitlines() if ln.startswith("Dialogue: 0")]
    assert len(lines) == 8  # one event per word
    assert "\\c&H0000E0FF" in lines[0] and "OLHA" in lines[0]  # yellow highlight, uppercase
    for style in STYLES.values():
        build_ass([(0, 1, "oi")], 2.0, style, filters.LAYOUT_FACECAM, None)


def test_zoom_expression():
    assert filters.Zoom(None).expr() == "1"
    z = filters.Zoom(5.0, amount=0.12).expr()
    assert "between(t,4.750,5.000)" in z and "between(t,5.000,6.000)" in z


def test_filters_per_layout():
    g, lay = filters.vertical_filter("blur", 1920, 1080, 30, filters.Zoom(10))
    assert (
        "boxblur" in g
        and "overlay" in g
        and "subtitles=subs.ass[v]" in g
        and lay is filters.LAYOUT_BLUR
    )
    assert "loudnorm=I=-14" in g and "eval=frame" in g
    g, lay = filters.vertical_filter("facecam", 1920, 1080, 30, filters.Zoom(None),
                                     facecam_rect=(1500, 700, 400, 360))  # fmt: skip
    assert "vstack" in g and "crop=400:360:1500:700" in g and lay is filters.LAYOUT_FACECAM
    g, _ = filters.vertical_filter("facecam", 1920, 1080, 30, filters.Zoom(None), facecam_rect=None)
    assert "boxblur" in g  # no rect -> blur fallback
    g, lay = filters.vertical_filter("blur", 1080, 1920, 30, filters.Zoom(None))
    assert "boxblur" not in g and lay is filters.LAYOUT_VERTICAL  # already vertical
    assert filters.scale_rect((1500, 700, 400, 360), 1280, 720) == (1000, 467, 266, 240)


# --- real renders -------------------------------------------------------------------------------


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=width,height,codec_type:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
    data = json.loads(out)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])
    return v["width"], v["height"], float(data["format"]["duration"]), has_audio


@pytest.fixture
def rsettings(settings):
    settings.render_encoder = "x264"
    return settings


@pytest.fixture
def setup(rsettings, engine):
    """Source with a 14 s 1280x720 video, a transcript and 2 approved candidates."""
    vid = rsettings.vods_dir / "1" / "video.mp4"
    vid.parent.mkdir(parents=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30",
         "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000", "-t", "14",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(vid)],
        check=True,
    )  # fmt: skip
    tr = rsettings.work_dir / "1" / "transcript.json.gz"
    save_transcript(tr, [Word(s, e, w) for s, e, w in WORDS], {"language": "pt"})
    with Session(engine, expire_on_commit=False) as s:
        st = Streamer(display_name="Fulano", permission_confirmed=True, permission_date=date.today(),
                      credit_text="Clip de @fulano na Twitch", facecam_x=1500, facecam_y=700,
                      facecam_w=400, facecam_h=360, default_layout="facecam")  # fmt: skip
        s.add(st)
        s.flush()
        src = Source(streamer_id=st.id, platform="twitch", kind="vod", external_id="1", url="https://www.twitch.tv/videos/1",
                     status=SourceStatus.READY, file_path=rsettings.rel_path(vid), duration_seconds=14,
                     media_info={"width": 1280, "height": 720})  # fmt: skip
        s.add(src)
        s.flush()
        run = AnalysisRun(source_id=src.id, params_hash="x", params=DetectParams().model_dump())
        s.add(run)
        s.flush()
        c1 = Candidate(source_id=src.id, analysis_run_id=run.id, rank=1, start=0.5, end=5.5,
                       peak_t=2.0, score=80, status=CandidateStatus.APPROVED)  # fmt: skip
        c2 = Candidate(source_id=src.id, analysis_run_id=run.id, rank=2, start=6.0, end=12.0,
                       peak_t=6.5, score=60, status=CandidateStatus.APPROVED,
                       overlay_title="Meu deus")  # fmt: skip
        s.add_all([c1, c2])
        s.commit()
    save_artifact(engine, src.id, "transcribe", "h", files={"transcript": rsettings.rel_path(tr)})
    return src.id, c1.id, c2.id


def drain(worker, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        worker.tick()
        time.sleep(0.05)
        if worker.active_job_ids():
            continue
        with Session(worker.engine) as s:
            if not s.exec(select(Job).where(Job.status.in_(["queued", "running"]))).first():
                return
    raise AssertionError("timeout")


def render(engine, worker, sid, ids):
    with Session(engine) as s:
        analysis.enqueue_render(s, sid, ids)
        s.commit()
    drain(worker)


@requires_ffmpeg
def test_render_blur_and_facecam_idempotent(rsettings, engine, setup):
    sid, c1, c2 = setup
    w = Worker(rsettings, engine)
    render(engine, w, sid, [c1, c2])
    with Session(engine) as s:
        jobs = s.exec(select(Job)).all()
        assert all(j.status == JobStatus.DONE for j in jobs), [j.error for j in jobs]
        clips = {r.candidate_id: r for r in s.exec(select(RenderedClip)).all()}
    r1 = clips[c1]
    out = rsettings.abs_path(r1.file_path)
    width, height, dur, audio = probe(out)
    assert (width, height, audio) == (1080, 1920, True) and dur == pytest.approx(5.0, abs=0.2)
    assert r1.layout == "facecam" and r1.encoder == "libx264"  # streamer default + rect
    pkg = rsettings.abs_path(r1.package_path).read_text("utf-8")
    assert "Clip de @fulano na Twitch" in pkg and "YouTube Shorts" in pkg and "música" in pkg
    assert rsettings.abs_path(r1.thumbnail_path).is_file()

    # same settings -> nothing re-rendered
    mtime = out.stat().st_mtime
    render(engine, w, sid, [c1])
    assert out.stat().st_mtime == mtime
    with Session(engine) as s:
        log = s.exec(select(Job).order_by(Job.id.desc())).first().log
    assert "já renderizado" in log

    # change layout + corrected subtitles -> new file, old one removed
    with Session(engine) as s:
        c = s.get(Candidate, c1)
        c.render_layout, c.subtitle_text, c.render_style = "blur", "Olha a POLÍCIA", "caixa"
        s.add(c)
        s.commit()
    render(engine, w, sid, [c1])
    with Session(engine) as s:
        r1b = s.exec(select(RenderedClip).where(RenderedClip.candidate_id == c1)).one()
    assert r1b.layout == "blur" and r1b.style == "caixa" and r1b.file_path != r1.file_path
    assert not out.exists() and rsettings.abs_path(r1b.file_path).is_file()


@requires_ffmpeg
def test_compile_long(rsettings, engine, setup):
    sid, _c1, _c2 = setup
    w = Worker(rsettings, engine)
    with Session(engine) as s:
        analysis.enqueue_compile_long(s, sid, None)
        s.commit()
    drain(w)
    with Session(engine) as s:
        art = s.exec(select(StageArtifact).where(StageArtifact.stage == "compile_long")).one()
    width, height, dur, audio = probe(rsettings.abs_path(art.files["video"]))
    assert (width, height, audio) == (1920, 1080, True)
    assert dur == pytest.approx(11.0, abs=0.4)  # 5 s + 6 s
    chapters = rsettings.abs_path(art.files["chapters"]).read_text("utf-8")
    assert "00:00 " in chapters and "00:05 Meu deus" in chapters


def test_render_requires_candidates(rsettings, engine):
    with Session(engine) as s, pytest.raises(Exception, match="pelo menos um"):
        analysis.enqueue_render(s, 1, [])


@requires_ffmpeg
@pytest.mark.skipif(not ffmpeg.try_encoder(ffmpeg.NVENC)[0], reason="NVENC indisponível")
def test_render_with_nvenc_on_this_gpu(rsettings, engine, setup):
    sid, c1, _ = setup
    rsettings.render_encoder = "nvenc"
    render(engine, Worker(rsettings, engine), sid, [c1])
    with Session(engine) as s:
        r = s.exec(select(RenderedClip)).one()
    assert r.encoder == "h264_nvenc"
    assert probe(rsettings.abs_path(r.file_path))[:2] == (1080, 1920)
