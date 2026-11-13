from unittest import mock

import httpx

from clipador.services import doctor
from clipador.services.doctor import CheckStatus


def test_python_check_ok():
    assert doctor.check_python().status == CheckStatus.OK


def test_missing_ffmpeg_fails_with_hint():
    with mock.patch("clipador.services.ffmpeg.tool_version", return_value=None):
        r = doctor.check_binary("ffmpeg", "ffmpeg")
    assert r.status == CheckStatus.FAIL
    assert "PATH" in r.hint


def test_nvenc_unavailable_is_warning(settings):
    with mock.patch("clipador.services.ffmpeg.try_encoder", return_value=(False, "no device")):
        r = doctor.check_nvenc(settings)
    assert r.status == CheckStatus.WARN
    assert "libx264" in r.detail


def test_ollama_offline_is_only_a_warning(settings):
    with mock.patch("httpx.get", side_effect=httpx.ConnectError("refused")):
        (r,) = doctor.check_ollama(settings)
    assert r.status == CheckStatus.WARN and "opcional" in r.detail


def test_ytdlp_version_shown_with_update_hint(settings):
    with mock.patch("clipador.services.downloader.YtDlp.version", return_value="2026.10.26"):
        r = doctor.check_ytdlp(settings)
    assert r.status == CheckStatus.OK
    assert "2026.10.26" in r.detail
    assert "yt-dlp -U" in r.hint


def test_missing_cookies_file_fails(settings, tmp_path):
    settings.cookies_file = tmp_path / "nope.txt"
    assert doctor.check_cookies(settings).status == CheckStatus.FAIL


def test_disk_check(settings):
    assert doctor.check_disk(settings).status in (
        CheckStatus.OK,
        CheckStatus.WARN,
        CheckStatus.FAIL,
    )
