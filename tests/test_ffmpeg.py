import subprocess
from unittest import mock

import pytest

from clipador.errors import ToolError
from clipador.services import ffmpeg
from tests.conftest import requires_ffmpeg


def _completed(returncode: int, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


def test_pick_encoder_uses_nvenc_when_test_encode_succeeds():
    with mock.patch("subprocess.run", return_value=_completed(0)) as run:
        assert ffmpeg.pick_encoder() == "h264_nvenc"
    cmd = run.call_args.args[0]
    assert "h264_nvenc" in cmd and "-t" in cmd and "0.2" in cmd


def test_pick_encoder_falls_back_when_nvenc_fails_to_open():
    err = "[h264_nvenc @ 0x1] Cannot load nvcuda.dll\nError while opening encoder"
    with mock.patch("subprocess.run", return_value=_completed(1, err)):
        assert ffmpeg.pick_encoder() == "libx264"


def test_pick_encoder_falls_back_when_ffmpeg_missing():
    with mock.patch("subprocess.run", side_effect=FileNotFoundError("ffmpeg")):
        assert ffmpeg.pick_encoder() == "libx264"


def test_tool_version_missing():
    with mock.patch("subprocess.run", side_effect=FileNotFoundError("x")):
        assert ffmpeg.tool_version("nao-existe") is None


def test_probe_missing_file(tmp_path):
    with pytest.raises(ToolError):
        ffmpeg.probe(tmp_path / "nope.mp4")


@requires_ffmpeg
def test_probe_and_thumbnail_on_synthetic_video(synthetic_video, tmp_path):
    info = ffmpeg.probe(synthetic_video)
    assert info.duration == pytest.approx(3.0, abs=0.2)
    assert (info.width, info.height) == (320, 180)
    assert info.fps == pytest.approx(30, abs=0.1)
    assert info.video_codec == "h264"
    assert info.audio_codec == "aac"
    thumb = ffmpeg.make_thumbnail(synthetic_video, tmp_path / "t.jpg", 1.0)
    assert thumb.stat().st_size > 0


@requires_ffmpeg
def test_probe_corrupt_file(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ToolError):
        ffmpeg.probe(bad)
