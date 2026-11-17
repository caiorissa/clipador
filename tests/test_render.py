"""Rendering: pure builders (ASS, layouts, safe zone) + real ffmpeg renders (no GPU needed)."""

import shutil
import subprocess
import time
from itertools import pairwise

import numpy as np
import pytest
from sqlmodel import Session, select

from clipador.errors import ClipadorError, JobCanceled, ToolError
from clipador.models import (
    Candidate,
    CandidateStatus,
    Job,
    Render,
    RenderStatus,
    StageArtifact,
    Streamer,
)
from clipador.render import audio as raudio
from clipador.render import filters
from clipador.render.styles import load_styles, parse_styles
from clipador.render.subtitles import (
    Overlays,
    apply_overrides,
    ass_time,
    build_ass,
    chunk_words,
    clean_text,
    credit_overlay,
    highlight_intervals,
    safe_zone,
    split_lines,
    words_in_window,
)
from clipador.render.validate import validate_render
from clipador.services import analysis, ffmpeg, rendering
from clipador.worker.runner import Worker
from tests.conftest import requires_ffmpeg
from tests.render_helpers import (
    FLASH_T,
    WORDS,
    brightest_frame_time,
    drain,
    ffprobe_json,
    frame_at,
    loudest_audio_time,
    make_video,
    seed,
)

ZONE = safe_zone(1080, 1920, 0.20, 0.10, 0.04)


def _secs(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


# --- pure: text, grouping, ASS -------------------------------------------------------------------


def test_clean_text_escapes_and_drops_emoji():
    assert clean_text("{\\b1}oi\nlinha 😂 ★") == "(/b1)oi linha"
    assert clean_text("ação São Paulo") == "ação São Paulo"


def test_words_in_window_keeps_global_index_and_relative_times():
    w = words_in_window(WORDS, 3.6, 6.2)
    assert [x.idx for x in w] == [3, 4, 5, 6]
    assert (w[0].start, w[0].end, w[0].text) == (0.0, 0.2, "corre")
    assert w[2].text == "/Ncorre" and w[3].end == pytest.approx(2.6)


def test_overrides_keep_times_and_can_hide():
    w = words_in_window(WORDS, 0, 12)
    o = apply_overrides(w, {"2": "POLÍCIA", "4": "", "99": "x"})
    assert [x.text for x in o][:3] == ["olha", "a", "POLÍCIA"]
    assert (o[2].start, o[2].end) == (w[2].start, w[2].end)
    assert all(x.idx != 4 for x in o)


def test_chunking_breaks_and_two_lines():
    chunks = chunk_words(words_in_window(WORDS, 0, 12), max_words=3, max_line_chars=16)
    texts = [[w.text for w in c] for c in chunks]
    assert texts[0] == ["olha", "a", "polícia!"]  # sentence end closes the chunk
    assert ["meu", "deus"] in texts  # pause > 0.6 s starts a new chunk
    assert split_lines(["CAPOTOU", "O", "CARRO"], 10) == 1
    assert split_lines(["OI"], 10) is None


def test_highlight_intervals_respect_min_event():
    w = words_in_window([(0.0, 0.1, "a"), (0.1, 0.2, "b"), (0.2, 1.0, "carro")], 0, 2)
    iv = highlight_intervals(w, 1.2, 0.25)
    assert all(e - s >= 0.25 for s, e, _ in iv)
    assert 0 in iv[0][2] and 1 in iv[0][2]  # tiny words merged


def test_build_ass_times_styles_overlays():
    styles = load_styles()
    assert {"classico", "destaque_amarelo", "pop"} <= set(styles)
    words = words_in_window(WORDS, 0.5, 5.5)
    ass = build_ass(words, 5.0, styles["destaque_amarelo"], ZONE,
                    Overlays("FUGIU {DA} POLÍCIA", 2.0, "via @fulano"))  # fmt: skip
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass and ",Anton," in ass
    assert f",{ZONE.caption_margin_v},1" in ass  # captions above the bottom UI band
    assert ("Dialogue: 2,0:00:00.00,0:00:02.00,Opening,,0,0,0,,{\\fad(150,250)}FUGIU (DA) POLÍCIA"
            in ass)  # fmt: skip
    assert "Dialogue: 1,0:00:00.00,0:00:05.00,Credit,,0,0,0,,via @fulano" in ass
    first = next(ln for ln in ass.splitlines() if ln.startswith("Dialogue: 0"))
    assert first.startswith("Dialogue: 0,0:00:00.50,") and "\\c&H0000E0FF" in first  # relative
    for ln in ass.splitlines():
        if ln.startswith("Dialogue: 0"):
            s, e = ln.split(",")[1:3]
            assert _secs(e) - _secs(s) >= 0.25 - 1e-6
    classico = build_ass(words, 5.0, styles["classico"], ZONE)
    assert "\\c&H" not in classico.split("[Events]")[1]  # no per-word colour
    assert "\\t(0,90,\\fscx" in build_ass(words, 5.0, styles["pop"], ZONE)


def test_styles_toml_parsing():
    s = parse_styles('[x]\nnome = "X"\nmodo = "nada"\npalavras_por_grupo = 9\n')["x"]
    assert s.mode == "destaque" and s.max_words == 4 and s.font == "Anton"
    with pytest.raises(ValueError):
        parse_styles("")


def test_ass_time_and_credit():
    assert ass_time(3723.456) == "1:02:03.46"
    assert credit_overlay("Clip de @fulano.tv na Twitch", "Fulano") == "via @fulano.tv"
    assert credit_overlay("", "Fulana") == "via Fulana"


def test_word_order():
    starts = [w.start for w in words_in_window(WORDS, 0, 12)]
    assert all(a <= b for a, b in pairwise(starts))


# --- pure: layouts -----------------------------------------------------------------------------


def test_layout_resolution_and_fallback():
    assert filters.resolve_layout("facecam_embaixo", None)[0] == "blur"
    assert "não tem retângulo" in filters.resolve_layout("facecam_embaixo", None)[1]
    assert filters.resolve_layout("facecam", (0.7, 0.6, 0.2, 0.3)) == ("facecam_embaixo", None)
    assert filters.normalize_layout("xyz") == "blur"


def test_relative_rect_scales_with_resolution():
    r = (0.75, 0.6, 0.2, 0.3)
    assert filters.rect_to_pixels(r, 1920, 1080) == (1440, 648, 384, 324)
    assert filters.rect_to_pixels(r, 1280, 720) == (960, 432, 256, 216)
    assert filters.rect_to_pixels(r, 2560, 1440) == (1920, 864, 512, 432)
    x, y, w, h = filters.rect_to_pixels((0.95, 0.95, 0.5, 0.5), 1920, 1080)
    assert x + w <= 1920 and y + h <= 1080  # clamped


def test_graphs_per_layout():
    af = raudio.loudnorm_filter(-14, -1.5, None, 5, 0.08)
    g = filters.vertical_graph("blur", 1920, 1080, 5, filters.Zoom(2), af)
    assert "boxblur" in g and "subtitles=subs.ass:fontsdir=fonts" in g and "eval=frame" in g
    g = filters.vertical_graph("facecam_embaixo", 1920, 1080, 5, filters.Zoom(None), af,
                               rect=(0.75, 0.6, 0.2, 0.3))  # fmt: skip
    assert "vstack" in g and "crop=384:324:1440:648" in g
    left = filters.LayoutOptions(zoom=1.0, foco_x=0.0)
    right = filters.LayoutOptions(zoom=1.0, foco_x=1.0)
    g = filters.vertical_graph("jogo_cheio", 1920, 1080, 5, filters.Zoom(None), af, opts=left)
    assert "crop=608:1080:0:0" in g
    g = filters.vertical_graph("jogo_cheio", 1920, 1080, 5, filters.Zoom(None), af, opts=right)
    assert "crop=608:1080:1312:0" in g
    g = filters.vertical_graph("blur", 1920, 1080, 5, filters.Zoom(None), af, out_size=(540, 960))
    assert "[s2]scale=540:304[fg]" in g and "scale=540:960:flags=bilinear[bg]" in g  # same layout
    assert filters.output_rate_args(59.94, 60) == ["-fps_mode", "cfr", "-r", "59.94"]
    assert filters.output_rate_args(144, 60)[-1] == "60"


def test_loudness_filters_and_parse():
    f = raudio.loudnorm_filter(-14, -1.5, 17.2, 10, 0.08)
    assert "aformat=channel_layouts=stereo,volume=17.20dB" in f and "alimiter=limit=0.8414" in f
    assert "afade=t=in:d=0.08" in f and "afade=t=out:st=9.920" in f
    assert "loudnorm=I=-14" in raudio.loudnorm_filter(-14, -1.5, None, 10, 0.08)  # preview
    assert raudio.parse_integrated("  Integrated loudness:\n    I:         -14.2 LUFS\n") == -14.2
    assert raudio.parse_integrated("I: -inf LUFS") is None


def test_streamer_facecam_box_property():
    st = Streamer(display_name="x", facecam_box=[0.7, 0.6, 0.2, 0.3])
    assert st.facecam_rect == (0.7, 0.6, 0.2, 0.3)
    assert Streamer(display_name="y", facecam_box=[0.1, 0.1, 0, 0.2]).facecam_rect is None


def test_encoder_fallback_nvenc_to_x264(monkeypatch, settings):
    calls = []

    def fake(args, **kw):
        calls.append(args)
        if len(calls) == 1:
            raise ToolError("Falha", detail="[h264_nvenc] OpenEncodeSessionEx failed")

    monkeypatch.setattr(ffmpeg, "run_with_progress", fake)
    plan = type("P", (), {"duration": 3.0, "out_duration": 3.0})()
    used = rendering.run_encode(lambda enc: ["-c:v", enc], settings, ffmpeg.NVENC, plan, None,
                                lambda f: None, lambda m: None)  # fmt: skip
    assert used == "libx264" and calls == [["-c:v", "h264_nvenc"], ["-c:v", "libx264"]]

    calls.clear()
    used = rendering.run_encode(lambda enc: ["-c:v", enc], settings, ffmpeg.VIDEOTOOLBOX, plan,
                                None, lambda f: None, lambda m: None)  # fmt: skip
    assert used == "libx264" and calls[0] == ["-c:v", "h264_videotoolbox"]

    def always_fail(*a, **k):
        raise ToolError("x")

    monkeypatch.setattr(ffmpeg, "run_with_progress", always_fail)
    with pytest.raises(ToolError):  # libx264 failures are not retried
        rendering.run_encode(lambda enc: [], settings, ffmpeg.X264, plan, None,
                             lambda f: None, lambda m: None)  # fmt: skip


# --- real renders --------------------------------------------------------------------------------


@pytest.fixture
def rsettings(settings):
    settings.render_encoder = "x264"
    return settings


@pytest.fixture(scope="session")
def media(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg não instalado")
    d = tmp_path_factory.mktemp("render media")
    return {
        "16x9": make_video(d / "src169.mp4"),
        "21x9": make_video(d / "src219.mp4", size="2560x1080", seconds=7),
        "vfr": make_video(d / "srcvfr.mp4", seconds=7, vfr=True),
    }


def _copy(media, key, settings):
    dst = settings.vods_dir / "1" / "video.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(media[key], dst)
    return dst


def _render(settings, engine, cid, kind="final"):
    with Session(engine) as s:
        analysis.enqueue_render(s, settings, 1, [cid], kind)
        s.commit()
    drain(Worker(settings, engine))
    with Session(engine) as s:
        return s.exec(
            select(Render)
            .where(Render.candidate_id == cid, Render.kind == kind)
            .order_by(Render.id.desc())
        ).first()


def _set(engine, model, oid, **fields):
    with Session(engine) as s:
        obj = s.get(model, oid)
        for k, v in fields.items():
            setattr(obj, k, v)
        s.add(obj)
        s.commit()


def _mp4s(settings):
    d = settings.clips_dir / "1"
    return list(d.glob("*.mp4")) if d.exists() else []


@requires_ffmpeg
@pytest.mark.parametrize("layout", ["blur", "facecam_embaixo", "jogo_cheio"])
def test_each_layout_renders_1080x1920_with_burned_captions(rsettings, engine, media, layout):
    video = _copy(media, "16x9", rsettings)
    _sid, (c1, _c2), _ = seed(rsettings, engine, video, rect=(0.7, 0.55, 0.25, 0.4), layout=layout)
    r = _render(rsettings, engine, c1)
    assert r.status == RenderStatus.DONE, r.error
    out = rsettings.abs_path(r.path)
    p = ffprobe_json(out, "stream=codec_type,codec_name,width,height:format=duration")
    v = next(s for s in p["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"], v["codec_name"]) == (1080, 1920, "h264")
    assert any(s["codec_type"] == "audio" and s["codec_name"] == "aac" for s in p["streams"])
    assert float(p["format"]["duration"]) == pytest.approx(5.0, abs=0.15)
    assert r.params["layout"] == layout and r.encoder == "libx264"

    # the same clip without captions: frames differ in the caption band (proof of burn-in)
    with Session(engine) as s:
        plan = rendering.plan_render(s, engine, rsettings, s.get(Candidate, c1))
    bare = rsettings.tmp_dir / "bare.mp4"
    graph = filters.vertical_graph(plan.layout, plan.src_w, plan.src_h, plan.duration,
                                   filters.Zoom(plan.zoom_peak), "anull", rect=plan.rect,
                                   opts=plan.opts, captions=False)  # fmt: skip
    ffmpeg.run_with_progress(
        ["-ss", f"{plan.start}", "-t", f"{plan.duration}", "-i", str(video), "-filter_complex",
         graph, "-map", "[v]", "-map", "[a]", *filters.encoder_args("libx264"), str(bare)],
        duration=plan.duration, progress_cb=lambda f: None,
    )  # fmt: skip
    a = frame_at(out, 1.0, 1080, 1920).astype(int)  # "polícia!" on screen at t=1.0
    b = frame_at(bare, 1.0, 1080, 1920).astype(int)
    bottom = 1920 - ZONE.caption_margin_v
    band = np.abs(a[bottom - 300 : bottom + 20] - b[bottom - 300 : bottom + 20]).mean()
    middle = np.abs(a[700:900] - b[700:900]).mean()
    assert band > 5 and band > 3 * middle


@requires_ffmpeg
def test_facecam_missing_falls_back_to_blur_with_warning(rsettings, engine, media):
    video = _copy(media, "16x9", rsettings)
    _sid, (c1, _), _ = seed(rsettings, engine, video, rect=None, layout="facecam_embaixo")
    r = _render(rsettings, engine, c1, "preview")
    assert r.status == RenderStatus.DONE and r.params["layout"] == "blur"
    assert "não tem retângulo de facecam" in r.warning


@requires_ffmpeg
@pytest.mark.parametrize("key", ["21x9", "vfr"])
def test_ultrawide_and_vfr_sources(rsettings, engine, media, key):
    video = _copy(media, key, rsettings)
    _sid, (c1, _), _ = seed(rsettings, engine, video, candidates=((0.5, 5.5, 2.0), (5.6, 6.8, 6)))
    r = _render(rsettings, engine, c1, "preview")
    assert r.status == RenderStatus.DONE, r.error
    v = ffprobe_json(rsettings.abs_path(r.path), "stream=width,height,r_frame_rate,avg_frame_rate")
    v = v["streams"][0]
    assert (v["width"], v["height"]) == (540, 960)
    assert v["r_frame_rate"] == v["avg_frame_rate"]  # constant frame rate output
    assert r.duration == pytest.approx(5.0, abs=0.15)


@requires_ffmpeg
def test_path_with_spaces_and_accents(tmp_path, media):
    from clipador.config import Settings
    from clipador.db import make_engine, run_migrations

    s = Settings(data_dir=tmp_path / "Meus Vídeos ção" / "dados clipador", _env_file=None,
                 render_encoder="x264")  # fmt: skip
    run_migrations(s)
    eng = make_engine(s)
    video = _copy(media, "16x9", s)
    _sid, (c1, _), _ = seed(s, eng, video)
    r = _render(s, eng, c1, "preview")
    assert r.status == RenderStatus.DONE, r.error
    full = str(s.abs_path(r.path))
    assert " " in full and "ção" in full
    eng.dispose()


@requires_ffmpeg
def test_final_loudness_fades_and_av_sync(rsettings, engine, media):
    video = _copy(media, "16x9", rsettings)
    _sid, (c1, _), _ = seed(rsettings, engine, video, candidates=((1.0, 9.0, 2.0), (9.5, 11.5, 10)))
    r = _render(rsettings, engine, c1)
    assert r.status == RenderStatus.DONE, r.error
    out = rsettings.abs_path(r.path)
    assert r.loudness_lufs == pytest.approx(-14.0, abs=1.5)
    assert raudio.integrated_loudness(out) == pytest.approx(-14.0, abs=1.5)
    # A/V sync: flash (video) and beep (audio) both at FLASH_T - start in the output
    expected = FLASH_T - 1.0
    assert brightest_frame_time(out) == pytest.approx(expected, abs=0.07)
    assert loudest_audio_time(out) == pytest.approx(expected + 0.05, abs=0.07)
    assert frame_at(out, 0.0, 1080, 1920).mean() < frame_at(out, 0.6, 1080, 1920).mean()  # fade in


@requires_ffmpeg
def test_cache_and_caption_change_invalidates(rsettings, engine, media):
    video = _copy(media, "16x9", rsettings)
    _sid, (c1, _), _ = seed(rsettings, engine, video)
    r1 = _render(rsettings, engine, c1, "preview")
    with Session(engine) as s:
        job, todo, reused = analysis.enqueue_render(s, rsettings, 1, [c1], "preview")
        s.commit()
        reused_ids = [r.id for r in reused]
    assert job is None and not todo and reused_ids == [r1.id]  # same params: nothing to do
    _set(engine, Candidate, c1, caption_overrides={"2": "VIATURA"})
    r2 = _render(rsettings, engine, c1, "preview")
    assert r2.id != r1.id and r2.params_hash != r1.params_hash and r2.status == RenderStatus.DONE
    assert not rsettings.abs_path(r1.path).exists()  # older render replaced
    with Session(engine) as s:
        assert s.get(Render, r1.id) is None


@requires_ffmpeg
def test_cancel_and_failure_leave_no_files(rsettings, engine, media, monkeypatch):
    video = _copy(media, "16x9", rsettings)
    _sid, (c1, c2), _ = seed(rsettings, engine, video)

    def canceled(*a, **k):
        raise JobCanceled()

    with Session(engine) as s:
        todo, _ = rendering.request_renders(s, engine, rsettings, [c1], "final")
        s.commit()
        plan = rendering.plan_render(s, engine, rsettings, s.get(Candidate, c1), "final")
        with pytest.raises(JobCanceled):
            rendering.execute_render(s, rsettings, todo[0], plan, canceled, lambda m: None)
    assert _mp4s(rsettings) == [] and not list(rsettings.tmp_dir.glob("render_*"))

    monkeypatch.setattr(rendering, "validate_render", lambda *a, **k: (None, ["resolução errada"]))
    with Session(engine) as s:
        todo, _ = rendering.request_renders(s, engine, rsettings, [c2], "preview")
        s.commit()
        plan = rendering.plan_render(s, engine, rsettings, s.get(Candidate, c2), "preview")
        with pytest.raises(ClipadorError, match="resolução errada"):
            rendering.execute_render(s, rsettings, todo[0], plan, lambda f: None, lambda m: None)
    assert _mp4s(rsettings) == [] and not list(rsettings.tmp_dir.glob("render_*"))


@requires_ffmpeg
def test_canceled_job_marks_render_canceled(rsettings, engine, media, monkeypatch):
    video = _copy(media, "16x9", rsettings)
    _sid, (c1, _), _ = seed(rsettings, engine, video)
    original = ffmpeg.run_with_progress

    def cancel_midway(args, *, progress_cb, **kw):
        with Session(engine) as s:  # the user clicks "Cancelar" while ffmpeg runs
            job = s.exec(select(Job).where(Job.type == "render")).one()
            job.status = "canceled"
            s.add(job)
            s.commit()
        time.sleep(1.2)  # let the cancel-check cache expire
        return original(args, progress_cb=progress_cb, **kw)

    monkeypatch.setattr(ffmpeg, "run_with_progress", cancel_midway)
    r = _render(rsettings, engine, c1)
    assert r.status == RenderStatus.CANCELED
    assert _mp4s(rsettings) == [] and not list(rsettings.tmp_dir.glob("render_*"))


@requires_ffmpeg
def test_validation_rejects_bad_files(tmp_path, media):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    assert validate_render(bad, 1080, 1920, 5.0)[1]
    no_fs = tmp_path / "nofaststart.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(media["16x9"]), "-t", "2",
                    "-c", "copy", str(no_fs)], check=True)  # fmt: skip
    problems = validate_render(no_fs, 1280, 720, 2.0)[1]
    assert any("faststart" in p for p in problems)
    problems = validate_render(no_fs, 1080, 1920, 5.0)[1]
    assert any("resolução" in p for p in problems) and any("duração" in p for p in problems)


@requires_ffmpeg
def test_final_sets_rendered_status_package_and_compile_long(rsettings, engine, media):
    video = _copy(media, "16x9", rsettings)
    sid, (c1, _c2), _ = seed(rsettings, engine, video)
    r = _render(rsettings, engine, c1)
    with Session(engine) as s:
        assert s.get(Candidate, c1).status == CandidateStatus.RENDERED
        assert "Clip de @fulano" in rsettings.abs_path(r.package_path).read_text("utf-8")
        analysis.enqueue_compile_long(s, sid, None)
        s.commit()
    drain(Worker(rsettings, engine))
    with Session(engine) as s:
        art = s.exec(select(StageArtifact).where(StageArtifact.stage == "compile_long")).one()
    p = ffprobe_json(rsettings.abs_path(art.files["video"]), "stream=width,height:format=duration")
    assert (p["streams"][0]["width"], p["streams"][0]["height"]) == (1920, 1080)
    assert float(p["format"]["duration"]) == pytest.approx(10.0, abs=0.5)
    assert "00:05" in rsettings.abs_path(art.files["chapters"]).read_text("utf-8")


@requires_ffmpeg
@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not ffmpeg.try_encoder(ffmpeg.NVENC)[0],
    reason="NVENC indisponível nesta máquina",
)
def test_render_with_nvenc_on_this_gpu(rsettings, engine, media):
    rsettings.render_encoder = "nvenc"
    video = _copy(media, "16x9", rsettings)
    _sid, (c1, _), _ = seed(rsettings, engine, video)
    r = _render(rsettings, engine, c1)
    assert r.status == RenderStatus.DONE and r.encoder == "h264_nvenc"


@requires_ffmpeg
def test_jump_cut_removes_dead_air_and_keeps_captions_in_sync(rsettings, engine, media):
    """The test source is silent between 5 s and 6 s, with no words there (4.9-6.0): that gap
    is cut. The 7-8 s gap has the background tone (as loud as the 'voice') and stays."""
    video = _copy(media, "16x9", rsettings)
    _sid, (c1,), _ = seed(rsettings, engine, video, candidates=((0.5, 11.0, 2.0),))
    r = _render(rsettings, engine, c1, "preview")
    assert r.status == RenderStatus.DONE, r.error
    out = rsettings.abs_path(r.path)
    dur = float(ffprobe_json(out, "format=duration")["format"]["duration"])
    assert dur == pytest.approx(10.5 - 0.7, abs=0.2)
    assert r.params["jumpcut"][0] == 1
    # the flash at 2.0 s (before the cut) is still at 1.5 s of the clip: A/V/caption timing kept
    assert brightest_frame_time(out) == pytest.approx(FLASH_T - 0.5, abs=0.1)

    _set(engine, Candidate, c1, render_options={"cut_silences": False})
    r2 = _render(rsettings, engine, c1, "preview")
    dur2 = float(ffprobe_json(rsettings.abs_path(r2.path), "format=duration")["format"]["duration"])
    assert dur2 == pytest.approx(10.5, abs=0.2) and "jumpcut" not in r2.params
