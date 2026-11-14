"""Facecam auto-detection (step 5C): synthetic frames + a fake face detector (no GPU, no
internet). Only the optional real-face test needs a real video (CLIPADOR_FACECAM_SAMPLE)."""

import os
import subprocess
import zlib
from pathlib import Path

import numpy as np
import pytest
import sqlalchemy as sa
from alembic import command
from sqlmodel import Session, select

from clipador.cli import main
from clipador.db import alembic_config, make_engine
from clipador.models import Source, SourceStatus, Streamer
from clipador.render import facecam_detect as fd
from clipador.services import facecam
from tests.conftest import requires_ffmpeg
from tests.test_ingest import _run_one, _setup, fake_adapter  # noqa: F401  (fixtures)
from tests.test_web import _create_streamer, client  # noqa: F401  (fixtures)

W, H = 640, 360
CAM = (440, 170, 180, 170)  # x, y, w, h (pixels) of the synthetic camera
FACE_W, FACE_H = 50, 60


def synthetic(n=16, cam=CAM, border=True, seed=0, jitter=2):
    """Moving noise ("game") + a quiet rectangle ("camera") with a drawn face that jitters."""
    rng = np.random.default_rng(seed)
    frames, faces = [], []
    yy, xx = np.ogrid[:H, :W]
    for _ in range(n):
        f = rng.integers(0, 256, (H, W)).astype(np.uint8)
        if cam is not None:
            x, y, w, h = cam
            f[y : y + h, x : x + w] = (90 + rng.integers(0, 3, (h, w))).astype(np.uint8)
            if border:
                f[y : y + 3, x : x + w] = 255
                f[y + h - 3 : y + h, x : x + w] = 255
                f[y : y + h, x : x + 3] = 255
                f[y : y + h, x + w - 3 : x + w] = 255
            jx, jy = rng.integers(-jitter, jitter + 1, 2)
            cx, cy = x + w // 2 + jx, y + h // 2 - 10 + jy
            face = ((xx - cx) / (FACE_W / 2)) ** 2 + ((yy - cy) / (FACE_H / 2)) ** 2 <= 1
            f[face] = 200
            f[cy - 8 : cy - 4, cx - 12 : cx - 6] = 30  # eyes
            f[cy - 8 : cy - 4, cx + 6 : cx + 12] = 30
            faces.append([(float(cx - FACE_W / 2), float(cy - FACE_H / 2), FACE_W, FACE_H)])
        else:
            faces.append([])
        frames.append(np.repeat(f[..., None], 3, axis=2))
    return frames, faces


class ListDetector:
    """Returns pre-computed boxes, one list per call (frames are analyzed in order)."""

    name = "fake"

    def __init__(self, per_frame):
        self.per_frame = list(per_frame)
        self.i = 0

    def detect(self, frame):
        out = self.per_frame[self.i % len(self.per_frame)]
        self.i += 1
        return out


def _expected():
    x, y, w, h = CAM
    return (x / W, y / H, w / W, h / H)


# --- detector core -----------------------------------------------------------------------------


def test_sample_times_skip_start_and_end():
    ts = fd.sample_times(3600, 16)
    assert len(ts) == 16 and ts[0] == 30 and ts[-1] == 3570  # 8% capped at 30 s
    ts = fd.sample_times(100, 5, start=20, end=60)
    assert ts[0] > 20 and ts[-1] < 60 and ts == sorted(ts)
    assert fd.sample_times(0, 16) == [0.0] and fd.sample_times(10, 1) == [5.0]
    assert fd.sample_size(1920, 1080) == (640, 360) and fd.sample_size(320, 180) == (320, 180)


def test_cluster_keeps_static_face_and_drops_moving_ones():
    rng = np.random.default_rng(1)
    dets = []
    for i in range(12):
        static = (500 + i % 2, 220 - i % 3, 50.0, 60.0)
        moving = (float(rng.integers(0, 400)), float(rng.integers(0, 250)), 40.0, 40.0)
        dets.append([moving, static] if i % 3 else [static])
    clusters = fd.cluster_faces(dets)
    best = fd.best_cluster(clusters)
    assert best.hits == 12 and best.stability() > 0.9
    assert all(c.hits <= 2 for c in clusters if c is not best)  # game characters scattered
    assert fd.iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3)


@pytest.mark.parametrize("border", [True, False])
def test_expand_face_to_camera_borders(border):
    frames, faces = synthetic(border=border)
    face = fd.FaceCluster([f[0] for f in faces], set(range(len(faces)))).median()
    (x, y, w, h), found = fd.expand_to_camera(frames, face)
    assert found == 4
    got = (x / W, y / H, w / W, h / H)
    assert got == pytest.approx(_expected(), abs=0.015)


def test_expand_prefers_motion_border_over_stronger_static_edge():
    """Regression: faint motion outside the camera (STD ~12) + a strong static color edge a bit
    further out. The quiet/moving border must win over the static edge."""
    frames, faces = synthetic(border=False)
    rng = np.random.default_rng(7)
    x, y, w, h = CAM
    cam = (slice(y, y + h), slice(x, x + w))
    out = []
    for f in frames:
        g = np.full((H, W), 60, np.int16)
        g[:, : x - 60] = 230  # static bar edge 60 px left of the camera
        g += rng.normal(0, 12, (H, W)).astype(np.int16)
        g[cam] = f[..., 0][cam]
        g = np.clip(g, 0, 255).astype(np.uint8)
        out.append(np.repeat(g[..., None], 3, axis=2))
    face = fd.FaceCluster([f[0] for f in faces], set(range(len(faces)))).median()
    (bx, _, bw, _), _ = fd.expand_to_camera(out, face)
    assert bx == pytest.approx(x, abs=6) and bw == pytest.approx(w, abs=8)


def test_analyze_frames_returns_normalized_box_with_confidence():
    frames, faces = synthetic()
    res = fd.analyze_frames(frames, ListDetector(faces))
    b = res.box
    assert b is not None and b.method == "bordas"
    assert (b.x, b.y, b.w, b.h) == pytest.approx(_expected(), abs=0.015)
    assert b.frames == 16 and b.total_frames == 16 and b.confidence >= 0.85
    assert b.as_list() == [round(v, 4) for v in (b.x, b.y, b.w, b.h)]
    assert b.x + b.w <= 1 and b.y + b.h <= 1


def test_none_without_camera_or_with_moving_faces_only():
    frames, faces = synthetic(cam=None)
    res = fd.analyze_frames(frames, ListDetector(faces))
    assert res.box is None and res.confidence == 0 and res.message == "nenhum rosto"
    rng = np.random.default_rng(3)
    moving = [[(float(rng.integers(0, 500)), float(rng.integers(0, 250)), 50.0, 60.0)]
              for _ in frames]  # fmt: skip
    res = fd.analyze_frames(frames, ListDetector(moving))
    assert res.box is None and res.frames <= 2
    assert fd.analyze_frames([], ListDetector([[]])).box is None


def test_fallback_margin_prefers_nearest_corner():
    frames = [np.full((H, W, 3), 120, np.uint8) for _ in range(8)]  # no borders, no motion
    face = (40.0, 30.0, 40.0, 48.0)
    res = fd.analyze_frames(frames, ListDetector([[face]]))
    b = res.box
    assert b is not None and b.x == 0 and b.y == 0  # touches the top-left corner
    assert b.w * W >= face[0] + face[2] and b.h * H >= face[1] + face[3]  # contains the face
    assert b.w < 0.4 and b.h < 0.6


def test_preview_png_is_valid(tmp_path):
    frames, faces = synthetic(n=4)
    res = fd.analyze_frames(frames, ListDetector(faces))
    img = fd.draw_detection(frames[0], res)
    assert img.shape == (H, W, 3)
    assert (img[CAM[1] - 5 : CAM[1] + 5, CAM[0] + 40] == (0, 230, 60)).all(axis=1).any()
    p = fd.write_png(tmp_path / "x.png", img)
    data = p.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    start = data.index(b"IDAT") + 4
    length = int.from_bytes(data[start - 8 : start - 4], "big")
    assert len(zlib.decompress(data[start : start + length])) == H * (W * 3 + 1)


def test_haar_detector_loads_and_finds_nothing_in_noise():
    pytest.importorskip("cv2")
    det = fd.HaarFaceDetector()
    frames, _ = synthetic(n=1, cam=None)
    assert det.detect(frames[0]) == []


@pytest.mark.skipif(
    not os.environ.get("CLIPADOR_FACECAM_SAMPLE"),
    reason="precisa de um vídeo real com facecam: defina CLIPADOR_FACECAM_SAMPLE=<arquivo.mp4>",
)
def test_real_video_with_facecam(tmp_path):  # pragma: no cover - manual check
    pytest.importorskip("cv2")
    from clipador.services import ffmpeg

    video = Path(os.environ["CLIPADOR_FACECAM_SAMPLE"])
    info = ffmpeg.probe(video)
    det = fd.make_face_detector(tmp_path, download=True)
    box = fd.detect_facecam(video, det, duration=info.duration, src_w=info.width,
                            src_h=info.height)  # fmt: skip
    assert box is not None and box.confidence >= fd.MIN_CONFIDENCE


# --- service: never overwrite a hand-marked box -------------------------------------------------


def _result(found=True, conf=0.9):
    box = fd.FacecamBox(0.7, 0.6, 0.25, 0.35, confidence=conf, frames=14, total_frames=16)
    return fd.DetectionResult(box if found else None, conf if found else 0.2, 14, 16,
                              detector="fake")  # fmt: skip


def test_apply_detection_respects_manual_box():
    manual = Streamer(display_name="m", facecam_box=[0.1, 0.1, 0.2, 0.2], facecam_auto=False)
    assert facecam.apply_detection(manual, _result()) is False
    assert manual.facecam_box == [0.1, 0.1, 0.2, 0.2]
    assert facecam.apply_detection(manual, _result(), overwrite_manual=True) is True
    assert manual.facecam_box == [0.7, 0.6, 0.25, 0.35] and manual.facecam_auto

    empty = Streamer(display_name="e")
    assert facecam.apply_detection(empty, _result(found=False)) is True
    assert empty.facecam_box is None and empty.facecam_auto and empty.facecam_confidence == 0.2
    auto = Streamer(display_name="a", facecam_box=[0.5, 0.5, 0.2, 0.2], facecam_auto=True)
    assert facecam.apply_detection(auto, _result(found=False)) is False  # keeps the old box
    assert facecam.apply_detection(auto, _result()) is True
    assert auto.facecam_box == [0.7, 0.6, 0.25, 0.35] and auto.facecam_confidence == 0.9


def _fake_detect(monkeypatch, found=True, calls=None):
    def fake(settings, source, detector=None, log_cb=None):
        if calls is not None:
            calls.append(source.id)
        return _result(found), [np.zeros((36, 64, 3), np.uint8)]

    monkeypatch.setattr("clipador.services.facecam.detect_for_source", fake)


@requires_ffmpeg
def test_ingest_fills_empty_box_but_not_manual(settings, engine, fake_adapter, monkeypatch):
    settings.facecam_auto_detect = True
    calls = []
    _fake_detect(monkeypatch, calls=calls)
    source_id, _ = _setup(engine)
    _run_one(settings, engine)
    with Session(engine) as s:
        src = s.get(Source, source_id)
        st = s.get(Streamer, src.streamer_id)
        assert st.facecam_box == [0.7, 0.6, 0.25, 0.35] and st.facecam_auto
        assert st.facecam_confidence == 0.9
        st.facecam_box, st.facecam_auto = [0.1, 0.1, 0.3, 0.3], False  # user marks by hand
        s.add(st)
        s.commit()
    assert facecam.preview_path(settings, source_id).is_file()
    with Session(engine, expire_on_commit=False) as s:
        src = s.get(Source, source_id)
        out = facecam.run_detection(s, settings, src, auto=True)
        assert out.status == facecam.Status.KEPT_EXISTING
        out = facecam.run_detection(s, settings, src)  # explicit, but no overwrite
        assert out.status == facecam.Status.FOUND_KEPT_MANUAL and not out.saved
        assert s.get(Streamer, src.streamer_id).facecam_box == [0.1, 0.1, 0.3, 0.3]
    assert calls == [source_id, source_id]


@requires_ffmpeg
def test_real_sampling_on_synthetic_video(tmp_path):
    video = tmp_path / "cam.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=640x360:rate=30",
         "-vf", "drawbox=x=440:y=170:w=180:h=170:color=gray:t=fill",
         "-t", "4", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )  # fmt: skip
    times = fd.sample_times(4.0, 6)
    frames = fd.sample_frames(video, times, 640, 360)
    assert len(frames) == 6 and frames[0].shape == (360, 640, 3)
    assert abs(int(frames[0][250, 530, 1]) - 128) < 20  # inside the gray box
    face = (505.0, 215.0, 50.0, 60.0)
    res = fd.analyze_frames(frames, ListDetector([[face]]))
    assert res.box is not None and res.box.frames == 6
    b = res.box
    assert b.x * W <= face[0] and (b.x + b.w) * W >= face[0] + face[2]


# --- migration, routes, command ----------------------------------------------------------------


def _columns(eng, table):
    return {c["name"] for c in sa.inspect(eng).get_columns(table)}


def test_0008_facecam_auto_upgrade_and_downgrade(settings):
    settings.ensure_dirs()
    cfg = alembic_config(settings)
    command.upgrade(cfg, "0007")
    eng = make_engine(settings)
    with eng.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO streamer (id, display_name, permission_confirmed, permission_note, "
            "credit_text, default_layout, facecam_box, show_credit, show_opening_text, active, "
            "created_at, updated_at) VALUES (1, 'F', 1, '', '', 'blur', '[0.7, 0.6, 0.2, 0.3]', "
            "1, 1, 1, '2026-11-13 00:00:00', '2026-11-13 00:00:00')"
        ))  # fmt: skip
    eng.dispose()
    command.upgrade(cfg, "0008")
    eng = make_engine(settings)
    assert {"facecam_auto", "facecam_confidence"} <= _columns(eng, "streamer")
    with eng.connect() as conn:
        row = conn.execute(sa.text("SELECT facecam_auto, facecam_confidence FROM streamer")).one()
    assert tuple(row) == (0, None)  # existing boxes count as hand-marked
    eng.dispose()
    command.downgrade(cfg, "0007")
    eng = make_engine(settings)
    assert "facecam_auto" not in _columns(eng, "streamer")
    eng.dispose()
    command.upgrade(cfg, "head")


def _ready_source(engine, settings, sid) -> int:
    video = settings.vods_dir / "1" / "video.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"fake")
    with Session(engine) as s:
        src = Source(streamer_id=sid, platform="twitch", kind="vod", external_id="77",
                     url="https://www.twitch.tv/videos/77", status=SourceStatus.READY,
                     file_path="vods/1/video.mp4", duration_seconds=60.0,
                     media_info={"width": 1920, "height": 1080, "duration": 60.0})  # fmt: skip
        s.add(src)
        s.commit()
        return src.id


def test_streamer_page_detect_again_and_manual_edit(client, monkeypatch):
    sid = _create_streamer(client)
    r = client.post(f"/streamers/{sid}/facecam/detectar")
    assert "Baixe uma fonte" in r.text
    _ready_source(client.engine, client.app.state.settings, sid)
    _fake_detect(monkeypatch)
    r = client.post(f"/streamers/{sid}/facecam/detectar")
    assert "Facecam detectada automaticamente" in r.text
    assert "Detectada automaticamente (90% de confiança), ajuste se precisar" in r.text
    with Session(client.engine) as s:
        st = s.get(Streamer, sid)
        assert st.facecam_rect == (0.7, 0.6, 0.25, 0.35) and st.facecam_auto

    form = {"display_name": "Fulano", "default_layout": "facecam_embaixo"}
    # saving the form without touching the box keeps it automatic
    client.post(f"/streamers/{sid}", data={**form, "fc_x": "0.7", "fc_y": "0.6", "fc_w": "0.25",
                                            "fc_h": "0.35"})  # fmt: skip
    with Session(client.engine) as s:
        assert s.get(Streamer, sid).facecam_auto
    # fine-tuning by hand turns it into a manual box
    client.post(f"/streamers/{sid}", data={**form, "fc_x": "0.72", "fc_y": "0.6", "fc_w": "0.25",
                                            "fc_h": "0.35"})  # fmt: skip
    with Session(client.engine) as s:
        st = s.get(Streamer, sid)
        assert not st.facecam_auto and st.facecam_confidence is None
    r = client.post(f"/streamers/{sid}/facecam/detectar")
    assert "marcada à mão foi mantida" in r.text
    with Session(client.engine) as s:
        assert s.get(Streamer, sid).facecam_rect == (0.72, 0.6, 0.25, 0.35)
    client.post(f"/streamers/{sid}/facecam/detectar", data={"sobrescrever": "1"})
    with Session(client.engine) as s:
        assert s.get(Streamer, sid).facecam_rect == (0.7, 0.6, 0.25, 0.35)


def test_streamer_page_warns_when_not_detected(client, monkeypatch):
    sid = _create_streamer(client)
    _ready_source(client.engine, client.app.state.settings, sid)
    _fake_detect(monkeypatch, found=False)
    r = client.post(f"/streamers/{sid}/facecam/detectar")
    assert "Facecam não detectada, marque à mão se quiser" in r.text
    with Session(client.engine) as s:
        st = s.get(Streamer, sid)
        assert st.facecam_box is None and st.facecam_confidence == 0.2


def test_cli_facecam_by_source_and_streamer(client, monkeypatch, capsys):
    settings = client.app.state.settings
    monkeypatch.setattr("clipador.cli.get_settings", lambda: settings)
    sid = _create_streamer(client)
    assert main(["facecam", "Ninguém"]) == 1
    assert "não encontrado" in capsys.readouterr().err
    assert main(["facecam", "fulano"]) == 1  # no downloaded source yet
    assert "nenhuma fonte baixada" in capsys.readouterr().err
    src_id = _ready_source(client.engine, settings, sid)
    _fake_detect(monkeypatch)
    assert main(["facecam", str(src_id), "--nao-salvar"]) == 0
    out = capsys.readouterr().out
    assert "x=0.7000 y=0.6000 largura=0.2500 altura=0.3500" in out and "90%" in out
    assert "Salvo no streamer" not in out and "facecam_preview.png" in out
    with Session(client.engine) as s:
        assert s.get(Streamer, sid).facecam_box is None
    assert main(["facecam", "Fulano"]) == 0
    assert "Salvo no streamer" in capsys.readouterr().out
    with Session(client.engine) as s:
        st = s.exec(select(Streamer).where(Streamer.id == sid)).one()
        assert st.facecam_auto and st.facecam_rect == (0.7, 0.6, 0.25, 0.35)
    assert main(["facecam", "999"]) == 1
