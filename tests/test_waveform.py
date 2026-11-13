import numpy as np
import pytest

from clipador.services import waveform as wf
from clipador.services.artifacts import save_artifact
from tests.review_helpers import seed_review, write_wav


@pytest.fixture(autouse=True)
def _source(settings, engine):
    seed_review(settings, engine, video=False, transcript=False)  # source #1 exists


def _audio(settings, engine, seconds, bursts, offset=0.0):
    path = write_wav(settings.work_dir / "1" / "audio.wav", seconds, bursts)
    save_artifact(engine, 1, "extract_audio", "aud1", files={"audio": settings.rel_path(path)},
                  summary={"offset": offset, "duration": seconds})  # fmt: skip
    return path


def _burst_times(d, threshold=150):
    p = np.array(d["peaks"])
    idx = np.flatnonzero(p > threshold)
    # first bar of each burst
    starts = idx[np.insert(np.diff(idx) > 1, 0, True)]
    return [round(d["t0"] + i * d["step"], 2) for i in starts]


def test_wav_layout_reads_header(settings, engine):
    path = _audio(settings, engine, 2, [])
    off, n, sr = wf.wav_layout(path)
    assert (off, n, sr) == (44, 32000, 16000)


def test_peaks_at_known_times_across_tiles(settings, engine):
    _audio(settings, engine, 200, [10.0, 59.5, 61.0, 150.0])
    d = wf.waveform(engine, settings, 1, 70, 100)  # window 10..160
    assert d["t0"] == pytest.approx(10.0) and d["step"] == pytest.approx(0.05)
    assert len(d["peaks"]) == 150 * 20
    assert _burst_times(d) == [10.0, 59.5, 61.0, 150.0]
    assert max(d["peaks"]) > 200 and np.median(d["peaks"]) < 10


def test_cache_is_reused(settings, engine, monkeypatch):
    _audio(settings, engine, 130, [30.0])
    calls = []
    real = wf.compute_tile
    monkeypatch.setattr(wf, "compute_tile", lambda *a, **k: calls.append(a[1]) or real(*a, **k))
    first = wf.waveform(engine, settings, 1, 40, 50)  # tiles 0 and 1
    assert sorted(calls) == [0, 1]
    second = wf.waveform(engine, settings, 1, 41, 52)
    assert sorted(calls) == [0, 1]  # nothing recomputed
    assert first["peaks"][:200] == second["peaks"][:200]
    assert len(list((settings.work_dir / "1" / "waveform").glob("*.npy"))) == 2


def test_respects_source_start_offset(settings, engine):
    # the WAV starts at 1000 s of the original video; a burst 12 s into the WAV is at 1012 s
    _audio(settings, engine, 60, [12.0], offset=1000.0)
    d = wf.waveform(engine, settings, 1, 1005, 1020, pad=10)
    assert d["t0"] == pytest.approx(1000.0)  # clamped to the start of the audio
    assert _burst_times(d) == [1012.0]
    assert d["offset"] == 1000.0


def test_missing_audio_is_reported(settings, engine):
    d = wf.waveform(engine, settings, 1, 0, 10)
    assert d["peaks"] == [] and "analise" in d["error"]
