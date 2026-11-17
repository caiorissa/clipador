"""Mac (MLX / VideoToolbox) vs PC (CUDA / NVENC) selection, without GPU or models."""

import wave
from unittest import mock

import numpy as np
import pytest

from clipador.config import Settings
from clipador.errors import ClipadorError
from clipador.render import filters
from clipador.services import doctor, rendering
from clipador.services import transcriber as tx


def _settings(tmp_path, **kw) -> Settings:
    return Settings(data_dir=tmp_path / "data", _env_file=None, **kw)


def test_auto_backend_picks_mlx_only_on_apple_silicon(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    monkeypatch.setattr(tx, "mlx_available", lambda: True)
    assert tx.resolve_backend(s) == "mlx"
    assert tx.resolve_model(s) == "large-v3-turbo"
    assert isinstance(tx.make_transcriber(s, print), tx.MlxWhisperTranscriber)
    monkeypatch.setattr(tx, "mlx_available", lambda: False)
    assert tx.resolve_backend(s) == "faster-whisper"
    assert tx.resolve_model(s) == "medium"  # unchanged default on the PC (old caches stay valid)
    assert isinstance(tx.make_transcriber(s, print), tx.FasterWhisperTranscriber)


def test_forced_backend_and_explicit_model(tmp_path, monkeypatch):
    monkeypatch.setattr(tx, "mlx_available", lambda: False)
    with pytest.raises(ClipadorError, match="Apple Silicon"):
        tx.resolve_backend(_settings(tmp_path, whisper_backend="mlx"))
    s = _settings(tmp_path, whisper_backend="faster-whisper", whisper_model="small")
    assert tx.resolve_model(s) == "small"


def test_mlx_repo_names():
    assert tx.mlx_repo("large-v3-turbo") == "mlx-community/whisper-large-v3-turbo"
    assert tx.mlx_repo("mlx-community/whisper-small-mlx-q4") == "mlx-community/whisper-small-mlx-q4"
    with pytest.raises(ClipadorError, match="não existe para MLX"):
        tx.mlx_repo("gigante")


def test_mlx_transcriber_maps_words_and_progress(tmp_path, monkeypatch):
    """The MLX path with a fake mlx_whisper module: words, language and progress forwarding."""
    wav = tmp_path / "a.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(np.zeros(16000, dtype=np.int16).tobytes())

    class FakeModule:
        tqdm = None

        class ModelHolder:
            model = object()
            model_path = "x"

        @staticmethod
        def transcribe(audio, **kw):
            assert kw["word_timestamps"] and kw["language"] == "pt"
            with FakeModule.tqdm.tqdm(total=100) as bar:
                bar.update(50)
                bar.update(50)
            return {"language": "pt", "segments": [{"words": [
                {"word": " oi", "start": 0.1, "end": 0.3, "probability": 0.9},
                {"word": " ", "start": 0.3, "end": 0.3},
            ]}]}  # fmt: skip

    monkeypatch.setattr(tx, "_mlx_module", lambda: FakeModule)
    t = tx.MlxWhisperTranscriber(_settings(tmp_path), print)
    t.model_path = "local"  # skip the download
    seen = []
    words = t.transcribe(wav, "pt", seen.append)
    assert [(w.word, w.start) for w in words] == [("oi", 0.1)]
    assert t.detected_language == "pt" and seen[:2] == [0.5, 1.0]
    assert FakeModule.tqdm is None  # restored


def test_videotoolbox_encoder_args():
    final = filters.encoder_args("h264_videotoolbox")
    assert final[:2] == ["-c:v", "h264_videotoolbox"] and "-q:v" in final
    assert "+faststart" in final and "yuv420p" in final
    assert "-prio_speed" in filters.encoder_args("h264_videotoolbox", preview=True)


def test_resolve_encoder_forced(tmp_path):
    assert rendering.resolve_encoder(_settings(tmp_path, render_encoder="videotoolbox")) == (
        "h264_videotoolbox"
    )


def test_render_parallel_default_follows_platform(tmp_path, monkeypatch):
    import clipador.config as cfg

    monkeypatch.setattr(cfg.sys, "platform", "darwin")
    assert _settings(tmp_path).render_parallel_with_gpu_jobs is True
    monkeypatch.setattr(cfg.sys, "platform", "win32")
    assert _settings(tmp_path).render_parallel_with_gpu_jobs is False


def test_doctor_on_a_mac(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    monkeypatch.setattr(doctor, "IS_MAC", True)
    monkeypatch.setattr(doctor.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        doctor,
        "_sysctl",
        lambda k: {"machdep.cpu.brand_string": "Apple M4", "hw.memsize": str(16 * 1024**3)}[k],
    )
    r = doctor.check_gpu()
    assert r.status == doctor.CheckStatus.OK and "Apple M4" in r.detail and "16 GB" in r.detail
    with (
        mock.patch("clipador.services.ffmpeg.sys.platform", "darwin"),
        mock.patch("clipador.services.ffmpeg.try_encoder", return_value=(True, "")),
    ):
        r = doctor.check_hw_encoder(s)
    assert r.name == "Encoder VideoToolbox" and r.status == doctor.CheckStatus.OK
    monkeypatch.setattr(tx, "mlx_available", lambda: True)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.0")
    r = doctor.check_whisper_accel(s)
    assert r.status == doctor.CheckStatus.OK and "large-v3-turbo" in r.detail
