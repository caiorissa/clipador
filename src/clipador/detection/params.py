"""Detection parameters (all tunables live here, hashed for caching) and presets."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

SIGNALS = ("audio", "chat", "speech", "keywords", "reaction")
SIGNAL_LABELS = {
    "audio": "Áudio",
    "chat": "Chat",
    "speech": "Fala",
    "keywords": "Palavras",
    "reaction": "Reação",  # laughs, screams, cheering (YAMNet audio events)
}

PRESETS: dict[str, dict[str, float]] = {
    "equilibrado": {"audio": 1.0, "chat": 1.0, "speech": 0.6, "keywords": 0.8, "reaction": 1.0},
    "foco_chat": {"audio": 0.6, "chat": 2.0, "speech": 0.4, "keywords": 0.6, "reaction": 0.6},
    "foco_audio": {"audio": 2.0, "chat": 0.6, "speech": 0.6, "keywords": 0.6, "reaction": 1.5},
    "foco_fala": {"audio": 0.6, "chat": 0.6, "speech": 1.5, "keywords": 1.5, "reaction": 0.8},
}
PRESET_LABELS = {
    "equilibrado": "Equilibrado",
    "foco_chat": "Foco no chat",
    "foco_audio": "Foco no áudio",
    "foco_fala": "Foco na fala",
}


class FeatureParams(BaseModel):
    """Per-second feature extraction. Changing these recomputes features.npz."""

    # Local baseline (rolling median) window: game/music volume drifts over a long stream.
    baseline_window_s: int = 300
    # Saturation scale for robust z-scores -> 0..1 (z of this size maps to ~0.63).
    z_scale: float = 3.0
    # Seconds of relative quiet before a burst that count as "silence then explosion".
    quiet_lookback_s: int = 4
    quiet_z: float = -0.8
    # Chat reacts after the moment happens; chat signals are shifted back by this much.
    chat_delay_s: float = 5.0
    # Window (s) to measure message repetition ("KEKW" spam).
    repeat_window_s: int = 5
    # Gaussian smoothing applied to each signal (s).
    smoothing_sigma_s: float = 1.5
    # Fine RMS resolution used to find pauses when snapping edges (frames per second).
    fine_fps: int = 10
    # Audio events (YAMNet): an event probability of this size counts as a full reaction.
    reaction_full_prob: float = 0.35
    # A second "has music" at this probability (copyright warning on the candidate).
    music_prob: float = 0.4


class CandidateParams(BaseModel):
    """Peak picking and window shaping. Changing these only re-runs the cheap detect step."""

    preset: str = "equilibrado"
    weights: dict[str, float] = Field(default_factory=lambda: dict(PRESETS["equilibrado"]))
    num_candidates: int = 20
    peak_min_height: float = 0.2  # on the 0..1 interest curve
    peak_min_distance_s: int = 20
    context_before_s: float = 4.0  # setup before the peak (guide: ~3 s before the peak)
    tail_after_s: float = 6.0  # after the curve decays
    decay_ratio: float = 0.45  # curve below peak*ratio = moment over
    min_duration_s: float = 30.0  # Shorts/Reels/TikTok sweet spot: 30-60 s
    max_duration_s: float = 60.0
    pause_min_s: float = 0.4  # gap between words that counts as a pause
    snap_window_s: float = 3.0  # how far an edge may move to find a pause
    min_gap_s: float = 3.0  # candidates closer than this are considered overlapping
    score_peak_weight: float = 0.7  # final score = w*peak + (1-w)*mean over the window


class DetectParams(BaseModel):
    features: FeatureParams = Field(default_factory=FeatureParams)
    candidates: CandidateParams = Field(default_factory=CandidateParams)
    # Detection x AI weights for the run ({"w_detect", "w_llm", ...}); None = app settings.
    # Set by calibrated presets (5B); RankingParams.from_run reads run.params["ranking"].
    ranking: dict[str, float] | None = None

    @classmethod
    def from_preset(cls, preset: str) -> DetectParams:
        p = cls()
        if preset in PRESETS:
            p.candidates.preset = preset
            p.candidates.weights = dict(PRESETS[preset])
        return p

    @classmethod
    def from_payload(cls, data: dict[str, Any] | None, default_preset: str) -> DetectParams:
        if not data:
            return cls.from_preset(default_preset)
        return cls.model_validate(data)
