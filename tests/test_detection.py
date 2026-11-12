"""Pure detection algorithms on synthetic data (no GPU, no network, no DB)."""

import numpy as np
import pytest

from clipador.detection import hype
from clipador.detection.candidates import (
    CandidateResult,
    detect_candidates,
    effective_weights,
    enforce_duration,
    interest_curve,
    non_max_suppression,
    snap_edges,
)
from clipador.detection.features import (
    audio_signal,
    chat_signal,
    compute_features,
    per_second_db,
    rms_db_frames,
    robust_local_z,
)
from clipador.detection.params import CandidateParams, FeatureParams
from clipador.platforms.base import ChatMessage

SR = 16000
FP = FeatureParams()
LEX = hype.HypeLexicon.from_toml(hype.DEFAULT_FILE.read_text("utf-8"))


def synth_audio(seconds: int, bursts: list[float], base_amp=0.02, ramp=False, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, seconds * SR).astype(np.float32)
    amp = np.full(x.size, base_amp, dtype=np.float32)
    if ramp:  # slow 4x volume increase over the whole file (game/music getting louder)
        amp *= np.linspace(1, 4, x.size, dtype=np.float32)
    for t in bursts:
        a, b = int(t * SR), int((t + 2) * SR)
        amp[a:b] = base_amp * 15
    return x * amp


def blocks(x, block_s=60):
    step = block_s * SR
    for i in range(0, x.size, step):
        yield i / SR, x[i : i + step]


def peaks_of(signal, height=0.5, distance=20):
    from scipy.signal import find_peaks

    idx, _ = find_peaks(signal, height=height, distance=distance)
    return list(idx)


def test_rms_frames_streaming_matches_direct():
    x = synth_audio(130, [60])
    fine = rms_db_frames(blocks(x, 7), SR, 10)  # odd block size to test the carry-over
    assert fine.size == 1300
    sec = per_second_db(fine, 10)
    assert sec.size == 130
    assert sec[60] - sec[30] > 20  # burst ~ +23 dB


def test_audio_peaks_found_at_known_times():
    bursts = [120.0, 400.0, 700.0]
    x = synth_audio(900, bursts)
    sec = per_second_db(rms_db_frames(blocks(x), SR, 10), 10)
    sig, _ = audio_signal(sec, FP)
    found = peaks_of(sig)
    assert len(found) == 3
    for t, f in zip(bursts, found, strict=True):
        assert abs(f - (t + 1)) <= 2  # burst lasts 2 s


def test_local_baseline_ignores_slow_volume_drift():
    x = synth_audio(1200, [], ramp=True)
    sec = per_second_db(rms_db_frames(blocks(x), SR, 10), 10)
    sig, _ = audio_signal(sec, FP)
    assert sig.max() < 0.45  # no peaks from the slow ramp
    assert peaks_of(sig, height=0.5) == []
    # a global (not local) z-score would see the end of the ramp as "loud"
    global_z = (sec - np.median(sec)) / (1.4826 * np.median(np.abs(sec - np.median(sec))))
    assert global_z[-60:].mean() > 1.0


def test_robust_z_short_series_does_not_crash():
    assert robust_local_z(np.array([1.0, 2.0]), 301, 1.0).shape == (2,)


def test_chat_peak_shifted_by_delay():
    rng = np.random.default_rng(1)
    msgs = [ChatMessage(float(t), f"u{i}", "oi") for i, t in enumerate(rng.uniform(0, 600, 600))]
    msgs += [ChatMessage(305 + i * 0.05, f"x{i}", "KEKW", ()) for i in range(60)]  # spam at 305 s
    sig, _ = chat_signal(sorted(msgs, key=lambda m: m.t), 0.0, 600, LEX, FP)
    peak = int(np.argmax(sig))
    assert abs(peak - (305 + 1.5 - FP.chat_delay_s)) <= 3  # chat reacts ~5 s late
    assert sig[peak] > 0.6
    assert sig[100] < 0.3


def test_weights_renormalize_without_chat():
    eff = effective_weights(
        {"audio": 1, "chat": 2, "speech": 0.5, "keywords": 0.5}, {"audio", "speech", "keywords"}
    )
    assert set(eff) == {"audio", "speech", "keywords"}
    assert sum(eff.values()) == pytest.approx(1.0)
    assert eff["audio"] == pytest.approx(0.5)


def test_hype_matching_accents_case_and_expressions():
    words = [(1.0, "Meu"), (1.3, "DEUS!"), (2.0, "olha"), (2.4, "a"), (2.6, "polícia"),
             (3.0, "kkkkkk"), (4.0, "cara"), (5.0, "retirou")]  # fmt: skip
    hits = hype.match_words(words, LEX)
    phrases = [h[2] for h in hits]
    assert "meu deus" in phrases and "policia" in phrases and "risada" in phrases
    assert "deus" not in phrases  # inside the longer expression
    assert not any(p in ("cara", "tiro") for p in phrases)  # generic / substring not matched
    assert hype.chat_message_weight("KEKW", (), LEX) == 1.0
    assert hype.chat_message_weight("kkkkkkk", (), LEX) == 1.0
    assert hype.chat_message_weight("boa noite", (), LEX) == 0.0


WORDS = [
    # sentence 1 (pause after 12.0)
    (5.0, 5.4, "então"), (5.5, 6.0, "vamos"), (6.1, 6.6, "pegar"), (6.7, 7.2, "o"), (7.3, 12.0, "carro"),
    # sentence 2 starts after a 1 s pause
    (13.0, 13.5, "olha"), (13.6, 14.0, "a"), (14.1, 15.0, "polícia"), (15.1, 16.0, "corre"),
    (16.1, 30.0, "corre"), (30.05, 31.0, "sai"), (31.6, 32.0, "daqui"),
]  # fmt: skip


def _inside_any(t):
    return any(w[0] < t < w[1] for w in WORDS)


def test_snap_prefers_pauses_and_never_cuts_words():
    p = CandidateParams(min_duration_s=10, max_duration_s=40, snap_window_s=3)
    start, end = snap_edges(14.5, 31.2, WORDS, p, (0, 100))
    assert not _inside_any(start) and not _inside_any(end)
    assert start == pytest.approx(13.0 - 0.2)  # sentence start after the pause at 12-13 s
    assert 31.0 <= end <= 31.6  # "sai" ends at 31.0 with a 0.6 s pause after


@pytest.mark.parametrize("start,end", [(7.0, 20.0), (15.5, 28.0), (16.5, 31.8), (0.0, 13.2)])
def test_snap_edges_property_no_mid_word(start, end):
    p = CandidateParams(min_duration_s=5, max_duration_s=60)
    s, e = snap_edges(start, end, WORDS, p, (0, 100))
    assert not _inside_any(s) and not _inside_any(e)
    assert s < e


def test_enforce_duration_extends_and_trims():
    p = CandidateParams(min_duration_s=20, max_duration_s=45)
    curve = np.zeros(300)
    curve[150] = 1
    s, e = enforce_duration(curve, 145, 155, 150, p, 300)
    assert e - s == pytest.approx(20) and s < 145  # more context before
    curve[100:200] = 0.2
    curve[140:160] = 0.9
    s, e = enforce_duration(curve, 90, 210, 150, p, 300)
    assert e - s == pytest.approx(45) and s <= 140 and e >= 160


def test_nms_removes_overlaps():
    a = CandidateResult(10, 40, 20, 90)
    b = CandidateResult(30, 60, 45, 80)  # overlaps a
    c = CandidateResult(41, 70, 50, 70)  # too close to a (gap < 3)
    d = CandidateResult(100, 130, 110, 60)
    kept = non_max_suppression([b, c, a, d], min_gap=3)
    assert [k.score for k in kept] == [90, 60]


def test_end_to_end_candidates_on_synthetic_stream():
    bursts = [200.0, 600.0, 1000.0]
    x = synth_audio(1200, bursts)
    words = [(t + 0.2 * i, t + 0.2 * i + 0.15, w) for t in (190.0, 590.0) for i, w in
             enumerate(["meu", "deus", "olha", "isso", "caralho"])]  # fmt: skip
    feats = compute_features(
        blocks=blocks(x), sample_rate=SR, t0=0.0, words=words, chat=None, lex=LEX, p=FP
    )
    assert not feats.has_chat
    p = CandidateParams(num_candidates=5)
    cands, curve, eff = detect_candidates(feats, words, p)
    assert "chat" not in eff
    assert len(cands) >= 3
    top3 = sorted(cands[:3], key=lambda c: c.peak_t)
    for c, t in zip(top3, bursts, strict=True):
        assert c.start <= t + 1 <= c.end
        assert p.min_duration_s - 0.5 <= c.duration <= p.max_duration_s + 0.5
        assert c.breakdown["chat"] is None and 0 <= c.score <= 100
    # bursts with hype words score higher than the one without
    by_t = {round(c.peak_t / 100): c for c in cands}
    assert by_t[2].score > by_t[10].score
    curve2, _ = interest_curve(feats, {"audio": 1, "speech": 0, "keywords": 0, "chat": 5})
    assert curve2.shape == curve.shape
