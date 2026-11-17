"""Audio events per second with YAMNet (Google, AudioSet 521 classes, Apache-2.0), on the CPU via
onnxruntime (already installed with faster-whisper, on the PC and on the Mac).

What volume alone can't tell: a laugh, a scream, cheering - and background MUSIC, which is the
real copyright risk of a clip. ~5 s of CPU per hour of audio; the model (16 MB) is downloaded
once to data/models and checked by SHA-256.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MODEL_FILE = "yamnet.onnx"
MODEL_URL = (  # pinned revision of an ONNX conversion of the official TF model
    "https://huggingface.co/zeropointnine/yamnet-onnx/resolve/"
    "ac2ca3bd45d12ec1f19f1144205ea529b4e9dedf/yamnet.onnx"
)
MODEL_SHA256 = "1510041dce24a2e9e84ec546807ac408ae496da6d1ed41bc3ccba649623f8e19"
SAMPLE_RATE = 16000
HOP_S = 0.48  # YAMNet frame hop (0.96 s windows)

# AudioSet class indexes (YAMNet class map).
GROUPS: dict[str, tuple[int, ...]] = {
    "laugh": (13, 14, 15, 16, 17, 18),  # Laughter, Baby laughter, Giggle, Snicker, Belly, Chuckle
    "scream": (6, 7, 8, 9, 11),  # Shout, Bellow, Whoop, Yell, Screaming
    "cheer": (58, 61, 62, 64),  # Clapping, Cheering, Applause, Crowd
    "music": (132,),  # Music (any genre also fires this top class)
}
EVENT_NAMES = tuple(GROUPS)
VERSION = 1


class AudioEventsUnavailable(Exception):
    """No model / no onnxruntime: detection simply runs without the 'reaction' signal."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(models_dir: Path, download: bool = True,
                 log_cb: Callable[[str], None] | None = None) -> Path:  # fmt: skip
    model = models_dir / MODEL_FILE
    if model.is_file():
        return model
    if not download:
        raise AudioEventsUnavailable("modelo de eventos de áudio não baixado")
    import httpx

    (log_cb or log.info)("Baixando o modelo de eventos de áudio YAMNet (~16 MB, uma vez só)…")
    model.parent.mkdir(parents=True, exist_ok=True)
    tmp = model.with_suffix(".part")
    try:
        with httpx.stream("GET", MODEL_URL, follow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise AudioEventsUnavailable(f"não foi possível baixar o YAMNet ({e})") from e
    if _sha256(tmp) != MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise AudioEventsUnavailable("o download do YAMNet veio corrompido (hash diferente)")
    tmp.replace(model)
    return model


class Yamnet:
    def __init__(self, model: Path):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise AudioEventsUnavailable("onnxruntime não instalado") from e
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def scores(self, samples: np.ndarray) -> np.ndarray:
        """(frames, 521) class scores for mono float32 16 kHz samples (frame i ~ i*0.48 s)."""
        if samples.size < SAMPLE_RATE:  # the model needs ~1 s
            samples = np.pad(samples, (0, SAMPLE_RATE - samples.size))
        return self.session.run(None, {self.input: samples.astype(np.float32, copy=False)})[0]


@dataclass
class AudioEvents:
    """Per-second probabilities (0..1) of each event group; index i = second [t0+i, t0+i+1)."""

    t0: float
    series: dict[str, np.ndarray]

    @property
    def n(self) -> int:
        return len(next(iter(self.series.values()))) if self.series else 0


def per_second(frame_scores: np.ndarray, n_seconds: int) -> dict[str, np.ndarray]:
    """Group max per frame, then max over the frames whose window center falls in each second."""
    out = {}
    centers = np.arange(frame_scores.shape[0]) * HOP_S + 0.48
    sec = np.clip(np.floor(centers).astype(int), 0, max(0, n_seconds - 1))
    for name, idx in GROUPS.items():
        g = frame_scores[:, list(idx)].max(axis=1) if frame_scores.size else np.zeros(0)
        arr = np.zeros(n_seconds)
        if g.size:
            np.maximum.at(arr, sec, g)
        out[name] = arr
    return out


def compute_events(
    blocks: Iterable[tuple[float, np.ndarray]], n_seconds: int, t0: float, model: Yamnet
) -> AudioEvents:
    """Stream (t, samples) blocks (whole seconds, as WavReader.blocks gives) through YAMNet."""
    series = {k: np.zeros(n_seconds) for k in GROUPS}
    for bt, samples in blocks:
        if samples.size == 0:
            continue
        start = round(bt)
        n = int(np.ceil(samples.size / SAMPLE_RATE))
        part = per_second(model.scores(samples), n)
        for k, v in part.items():
            end = min(n_seconds, start + v.size)
            if end > start:
                series[k][start:end] = np.maximum(series[k][start:end], v[: end - start])
    return AudioEvents(t0, series)


def make_model(settings, log_cb: Callable[[str], None] | None = None) -> Yamnet:
    """The YAMNet runner (tests monkeypatch this). Raises AudioEventsUnavailable."""
    return Yamnet(ensure_model(settings.models_dir, settings.audio_events_download, log_cb))
