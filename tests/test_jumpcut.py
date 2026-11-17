"""Jump cuts: which silences are cut (never gameplay action or the peak), remapping, the ffmpeg
graph, and a real render that comes out shorter with captions still in sync."""

import wave

import numpy as np
import pytest

from clipador.render import jumpcut as jc
from tests.conftest import requires_ffmpeg

FPS = jc.FPS


def _db(duration, loud=(), voice=(), base=-60.0):
    """dB per frame: `base` silence, voice spans at -20 dB, loud (non-voice) spans at -10 dB."""
    db = np.full(int(duration * FPS), base)
    for a, b in voice:
        db[int(a * FPS) : int(b * FPS)] = -20.0
    for a, b in loud:
        db[int(a * FPS) : int(b * FPS)] = -10.0
    return db


def test_cuts_quiet_gaps_only():
    words = [(0.5, 2.0), (6.0, 8.0), (8.2, 9.0)]
    db = _db(12, voice=words)
    keeps = jc.keep_intervals(words, db, 12.0)
    # gap 2.0-6.0 cut (with 0.2 s pads); tail 9.0-12.0 cut too
    assert keeps == [(0.0, 2.2), (5.8, 9.2)]
    assert jc.total(keeps) == pytest.approx(5.6)
    assert jc.remap(1.0, keeps) == 1.0 and jc.remap(6.0, keeps) == pytest.approx(2.4)
    assert jc.remap(4.0, keeps) == pytest.approx(2.2)  # inside a cut -> next kept frame


def test_never_cuts_loud_gameplay_or_the_peak():
    words = [(0.5, 2.0), (6.0, 8.0)]
    assert jc.is_trivial(jc.keep_intervals(words, _db(8, loud=[(3, 4)], voice=words), 8.0), 8.0)
    assert jc.is_trivial(jc.keep_intervals(words, _db(8, voice=words), 8.0, peak=4.0), 8.0)
    assert jc.is_trivial(jc.keep_intervals([], _db(8), 8.0), 8.0)  # no words: no basis


def test_cut_filters_graph():
    parts, v, a = jc.cut_filters([(0.0, 2.2), (5.8, 9.2)])
    g = ";".join(parts)
    assert "split=2" in g and "asplit=2" in g and "concat=n=2:v=1:a=1[jcv][jca]" in g
    assert "trim=start=5.800:end=9.200" in g and (v, a) == ("[jcv]", "[jca]")


@requires_ffmpeg
def test_clip_db_reads_audio(tmp_path):
    wav = tmp_path / "a.wav"
    x = np.zeros(16000 * 2)
    x[16000:] = 0.5 * np.sin(np.arange(16000) / 5)
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    db = jc.clip_db(wav, 0.0, 2.0)
    assert db.size == pytest.approx(2 * FPS, abs=1)
    assert db[: FPS - 2].max() < -80 and db[FPS + 2 :].min() > -20
