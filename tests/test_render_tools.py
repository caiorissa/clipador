import subprocess
from unittest import mock

from clipador.cli import main
from clipador.services import doctor
from clipador.services.doctor import CheckStatus
from clipador.services.renderbench import BenchRow, format_rows
from tests.conftest import requires_ffmpeg


def _filters_output(names):
    return "\n".join(f" ... {n}  V->V  desc" for n in names)


def test_render_filters_check_ok_and_missing_libass(settings):
    full = subprocess.CompletedProcess([], 0, _filters_output(doctor.REQUIRED_FILTERS), "")
    with mock.patch("subprocess.run", return_value=full):
        assert doctor.check_render_filters(settings).status == CheckStatus.OK
    partial = subprocess.CompletedProcess([], 0, _filters_output(["overlay", "boxblur"]), "")
    with mock.patch("subprocess.run", return_value=partial):
        r = doctor.check_render_filters(settings)
    assert r.status == CheckStatus.FAIL and "libass" in r.detail
    assert "gyan.dev" in r.hint or "homebrew-ffmpeg" in r.hint


def test_fonts_and_clips_space_checks(settings, tmp_path):
    r = doctor.check_fonts(settings)
    assert r.status == CheckStatus.OK and "Anton" in r.detail and "destaque_amarelo" in r.detail
    settings.fonts_dir = tmp_path / "vazio"
    settings.fonts_dir.mkdir()
    assert doctor.check_fonts(settings).status == CheckStatus.FAIL
    assert doctor.check_clips_space(settings).status in (CheckStatus.OK, CheckStatus.WARN)


def test_format_bench_rows():
    rows = [BenchRow("h264_nvenc", "final", 8.0, 40.0, 50_000_000, True),
            BenchRow("libx264", "-", 0, 40.0, 0, False, "NVENC indisponível")]  # fmt: skip
    out = format_rows(rows, 40.0)
    assert "5.0x" in out and "50.0 MB" in out and "falhou: NVENC indisponível" in out


@requires_ffmpeg
def test_render_bench_synthetic_x264(settings, monkeypatch, capsys):
    monkeypatch.setattr("clipador.cli.get_settings", lambda: settings)
    assert main(["render-bench", "--seconds", "3", "--encoders", "x264"]) == 0
    out = capsys.readouterr().out
    assert "libx264     preview" in out and "libx264     final" in out and "x " in out


def test_render_bench_unknown_candidate(settings, monkeypatch, capsys):
    monkeypatch.setattr("clipador.cli.get_settings", lambda: settings)
    assert main(["render-bench", "999", "--encoders", "x264"]) == 1
    assert "não existe" in capsys.readouterr().err
