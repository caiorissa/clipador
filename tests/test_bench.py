from unittest import mock

from clipador.cli import main
from clipador.services.transcriber import Word
from tests.conftest import requires_ffmpeg


class FakeFW:
    def __init__(self, settings, log_cb):
        self.settings = settings
        self.device, self.compute_type, self.detected_language = "cuda", "float16", "pt"

    def _load(self):
        pass

    def transcribe(self, path, language, progress_cb):
        return [Word(0.1, 0.4, "oi")]

    def close(self):
        pass


@requires_ffmpeg
def test_bench_transcribe_reports_speed(settings, synthetic_video, monkeypatch, capsys):
    monkeypatch.setattr("clipador.cli.get_settings", lambda: settings)
    monkeypatch.setattr("clipador.services.transcriber.make_transcriber", FakeFW)
    with mock.patch("shutil.which", return_value=None):  # no nvidia-smi polling
        assert main(["bench-transcribe", str(synthetic_video), "--seconds", "2"]) == 0
    out = capsys.readouterr().out
    assert "x tempo real" in out and "min por hora de VOD" in out and "1 palavras" in out


def test_bench_transcribe_missing_file(settings, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("clipador.cli.get_settings", lambda: settings)
    assert main(["bench-transcribe", str(tmp_path / "nada.mp4")]) == 1
    assert "não encontrado" in capsys.readouterr().err
