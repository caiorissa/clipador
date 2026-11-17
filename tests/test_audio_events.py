"""YAMNet audio events without the real model: grouping, streaming, the 'reaction' signal,
the music share of a candidate, and the detect stage falling back when the model is missing."""

import numpy as np
import pytest

from clipador.detection import audio_events as ae
from clipador.detection.candidates import detect_candidates
from clipador.detection.features import compute_features, reaction_signal
from clipador.detection.params import CandidateParams, FeatureParams

SR = ae.SAMPLE_RATE


class FakeYamnet:
    """Frame scores from the loudness of each 0.48 s hop: loud = laughter, very quiet = music."""

    def scores(self, samples):
        n = max(1, int(samples.size / SR / ae.HOP_S))
        out = np.zeros((n, 521), dtype=np.float32)
        for i in range(n):
            seg = samples[int(i * ae.HOP_S * SR) : int((i * ae.HOP_S + 0.96) * SR)]
            rms = float(np.sqrt(np.mean(seg**2))) if seg.size else 0.0
            out[i, 13] = 0.9 if rms > 0.2 else 0.01  # Laughter
            out[i, 132] = 0.8 if 0 < rms < 0.01 else 0.0  # Music
        return out


def _blocks(x, block_s=60):
    for t in range(0, int(np.ceil(x.size / SR)), block_s):
        yield float(t), x[t * SR : (t + block_s) * SR]


def _audio(seconds=120, laugh_at=(50,), music=(0, 20)):
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.05, seconds * SR).astype(np.float32)
    x[music[0] * SR : music[1] * SR] *= 0.1
    for t in laugh_at:
        x[t * SR : (t + 3) * SR] *= 8
    return x


def test_per_second_groups_and_streaming():
    x = _audio()
    ev = ae.compute_events(_blocks(x, 30), 120, 0.0, FakeYamnet())
    assert set(ev.series) == {"laugh", "scream", "cheer", "music"} and ev.n == 120
    assert ev.series["laugh"][51] > 0.5 and ev.series["laugh"][90] < 0.1
    assert ev.series["music"][10] > 0.5 and ev.series["music"][60] == 0


def test_reaction_signal_and_music_share_on_candidates():
    x = _audio()
    ev = ae.compute_events(_blocks(x), 120, 0.0, FakeYamnet()).series
    p = FeatureParams()
    sig, extra = reaction_signal(ev, 120, p)
    assert sig[51] > 0.5 > sig[100]
    assert extra["music"][:19].mean() > 0.9  # flags, not probabilities
    feats = compute_features(blocks=_blocks(x), sample_rate=SR, t0=0.0, words=[], chat=None,
                             lex=_lex(), p=p, events=ev)  # fmt: skip
    assert "reaction" in feats.signals
    cp = CandidateParams(min_duration_s=10, max_duration_s=20, num_candidates=5)
    cp.weights = {"reaction": 1.0}
    results, _curve, eff = detect_candidates(feats, [], cp)
    assert eff["reaction"] == 1.0
    best = results[0]
    assert best.start <= 51 <= best.end and best.breakdown["reaction"] > 50
    assert "music" in best.breakdown


def _lex():
    import tempfile
    from pathlib import Path

    from clipador.detection.hype import ensure_hype_file, load_lexicon

    return load_lexicon(ensure_hype_file(Path(tempfile.mkdtemp()) / "hype.toml"))


def test_model_download_is_verified(tmp_path, monkeypatch):
    with pytest.raises(ae.AudioEventsUnavailable, match="não baixado"):
        ae.ensure_model(tmp_path, download=False)

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def iter_bytes(self):
            yield b"not the model"

    import httpx

    monkeypatch.setattr(httpx, "stream", lambda *a, **k: Resp())
    with pytest.raises(ae.AudioEventsUnavailable, match="corrompido"):
        ae.ensure_model(tmp_path, download=True)
    assert not (tmp_path / ae.MODEL_FILE).exists()


# --- the detect stage ----------------------------------------------------------------------------

from sqlmodel import Session, select  # noqa: E402

from clipador.models import Candidate, Job  # noqa: E402
from clipador.worker.runner import Worker  # noqa: E402
from tests.conftest import requires_ffmpeg  # noqa: E402
from tests.test_analysis import _analyze, _drain, _source, asettings, fake_tx  # noqa: E402, F401


@requires_ffmpeg
def test_detect_stage_uses_events_and_survives_without_model(asettings, engine, fake_tx,
                                                             monkeypatch):  # fmt: skip
    asettings.audio_events = True
    monkeypatch.setattr(ae, "make_model", lambda settings, log_cb=None: FakeYamnet())
    sid = _source(engine, asettings)
    _analyze(engine, sid)
    _drain(Worker(asettings, engine))
    with Session(engine) as s:
        cands = s.exec(select(Candidate)).all()
        log = s.exec(select(Job).where(Job.type == "detect")).one().log
    assert cands and all("reaction" in c.breakdown and "music" in c.breakdown for c in cands)
    assert "Eventos de áudio" in log

    def unavailable(settings, log_cb=None):
        raise ae.AudioEventsUnavailable("sem internet")

    monkeypatch.setattr(ae, "make_model", unavailable)
    asettings.hype_words_path.write_text("[palavras]\nnovo = 1.0\n", "utf-8")  # new features
    _analyze(engine, sid)
    _drain(Worker(asettings, engine))
    with Session(engine) as s:
        logs = [j.log for j in s.exec(select(Job).where(Job.type == "detect")).all()]
    assert any("Sem eventos de áudio (sem internet)" in lg for lg in logs)
