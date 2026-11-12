from unittest import mock

from clipador.services import doctor
from clipador.services.doctor import CheckStatus
from clipador.services.downloader import is_noise


def test_js_runtime_missing_is_warning():
    with mock.patch("shutil.which", return_value=None):
        r = doctor.check_js_runtime()
    assert r.status == CheckStatus.WARN
    assert "deno" in r.hint.lower()


def test_ytdlp_noise_filter():
    assert is_noise("[info] Downloading video thumbnail 42 ...")
    assert is_noise("Deleting original file x.webp (pass -k to keep)")
    assert not is_noise("ERROR: [youtube] abc: Private video")
    assert not is_noise("[youtube] abc: Downloading webpage")
