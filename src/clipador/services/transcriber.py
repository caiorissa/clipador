"""Speech-to-text behind a small Protocol, plus chunk planning/merging (pure functions).

Tests use a fake Transcriber; nothing here needs a GPU or model unless FasterWhisperTranscriber
is actually used.

Backends: faster-whisper (CUDA on the Windows/Linux PC, CPU fallback) and mlx-whisper (GPU of
Apple Silicon Macs). `make_transcriber` picks the right one (CLIPADOR_WHISPER_BACKEND=auto).
"""

from __future__ import annotations

import contextlib
import gc
import gzip
import importlib.util
import json
import logging
import platform
import sys
import threading
import time
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from clipador.config import Settings
from clipador.errors import ClipadorError
from clipador.services.audio import WavReader

log = logging.getLogger(__name__)


@dataclass
class Word:
    start: float
    end: float
    word: str
    prob: float = 1.0


class Transcriber(Protocol):
    def transcribe(
        self, audio_path: Path, language: str | None, progress_cb: Callable[[float], None]
    ) -> list[Word]:
        """Words with times relative to the start of `audio_path`. progress_cb gets 0..1."""
        ...


# --- chunking (pure) -------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float  # nominal span owned by this chunk
    end: float
    read_start: float  # span actually transcribed (with overlap)
    read_end: float


def plan_chunks(total_s: float, chunk_s: float, overlap_s: float) -> list[Chunk]:
    chunks = []
    i, t = 0, 0.0
    while t < total_s:
        end = min(total_s, t + chunk_s)
        if total_s - end < chunk_s * 0.1:  # avoid a tiny last chunk
            end = total_s
        chunks.append(Chunk(i, t, end, max(0.0, t - overlap_s), min(total_s, end + overlap_s)))
        i += 1
        t = end
    return chunks


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if c.isalnum())


def merge_chunk_words(parts: list[tuple[Chunk, list[Word]]]) -> list[Word]:
    """Each chunk keeps only words that START inside its nominal span; then near-duplicates at
    the seams (same word transcribed by both neighbors with slightly different times) are
    dropped. Word times must already be absolute."""
    kept: list[Word] = []
    for chunk, words in sorted(parts, key=lambda p: p[0].index):
        for w in words:
            if not (chunk.start <= w.start < chunk.end):
                continue
            if kept:
                prev = kept[-1]
                if w.start < prev.end - 0.05 and (
                    _norm(w.word) == _norm(prev.word) or w.start <= prev.start
                ):
                    continue
            kept.append(w)
    return kept


def save_transcript(path: Path, words: list[Word], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump({**meta, "words": [asdict(w) for w in words]}, f, ensure_ascii=False)
    tmp.replace(path)


def load_transcript(path: Path) -> tuple[list[Word], dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    words = [Word(**w) for w in data.pop("words")]
    return words, data


# --- faster-whisper ----------------------------------------------------------------------------


def _classify_gpu_error(e: Exception) -> ClipadorError:
    from clipador.services.cuda import CUDA_LIBS_HINT

    msg = str(e)
    low = msg.lower()
    if "out of memory" in low:
        return ClipadorError(
            "Faltou memória de vídeo (VRAM) para o modelo de transcrição. Feche jogos/OBS, ou use "
            "CLIPADOR_WHISPER_COMPUTE_TYPE=int8_float16, ou um modelo menor "
            "(CLIPADOR_WHISPER_MODEL=small).",
            detail=msg,
        )
    if any(k in low for k in ("cudnn", "cublas", "cuda", ".dll", "libcu")):
        return ClipadorError(
            "A GPU não pôde ser usada para transcrever (biblioteca CUDA/cuDNN ausente ou "
            "incompatível). " + CUDA_LIBS_HINT,
            detail=msg,
        )
    if any(k in low for k in ("connection", "timed out", "huggingface", "resolve", "http")):
        return ClipadorError(
            "Não foi possível baixar o modelo de transcrição (rede). Tente de novo.",
            detail=msg,
            retryable=True,
        )
    return ClipadorError(f"Falha na transcrição: {msg[:300]}", detail=msg, retryable=True)


class FasterWhisperTranscriber:
    def __init__(self, settings: Settings, log_cb: Callable[[str], None] = log.info):
        self.settings = settings
        self.log_cb = log_cb
        self.model = None
        self.device = "cpu"
        self.compute_type = "int8"
        self.detected_language: str | None = None

    def _load(self) -> None:
        if self.model is not None:
            return
        from clipador.services.cuda import cuda_status

        want = self.settings.whisper_device
        status = cuda_status() if want in ("auto", "cuda") else None
        if status is not None and status.usable:
            self.device, self.compute_type = "cuda", self.settings.whisper_compute_type
        else:
            if want == "cuda":
                from clipador.services.cuda import CUDA_LIBS_HINT

                raise ClipadorError(
                    "CLIPADOR_WHISPER_DEVICE=cuda, mas a GPU não está disponível. "
                    + CUDA_LIBS_HINT,
                    detail=str(status),
                )
            self.device, self.compute_type = "cpu", "int8"
            if want == "auto":
                self.log_cb(
                    "AVISO: GPU/CUDA indisponível; transcrevendo na CPU (int8). Isso é MUITO mais "
                    f"lento (horas para uma live longa). Detalhe: {status}"
                )
        self.log_cb(
            f"Carregando modelo '{resolve_model(self.settings, 'faster-whisper')}' ({self.device}, "
            f"{self.compute_type})… na primeira vez ele é baixado."
        )
        try:
            from faster_whisper import WhisperModel

            self.settings.models_dir.mkdir(parents=True, exist_ok=True)
            self.model = WhisperModel(
                resolve_model(self.settings, "faster-whisper"),
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.settings.models_dir),
            )
        except Exception as e:
            raise _classify_gpu_error(e) from e

    def transcribe(
        self, audio_path: Path, language: str | None, progress_cb: Callable[[float], None]
    ) -> list[Word]:
        self._load()
        reader = WavReader(audio_path)
        samples = reader.read(0, reader.duration)  # one chunk (~10 min) fits easily in RAM
        total = max(reader.duration, 1e-6)
        words: list[Word] = []
        try:
            segments, info = self.model.transcribe(
                samples,
                language=language,
                beam_size=self.settings.whisper_beam_size,
                vad_filter=self.settings.whisper_vad_filter,
                word_timestamps=True,
                condition_on_previous_text=False,
            )
            self.detected_language = info.language
            for seg in segments:
                for w in seg.words or []:
                    text = w.word.strip()
                    if text:
                        words.append(Word(float(w.start), float(w.end), text, float(w.probability)))
                progress_cb(min(1.0, float(seg.end) / total))
        except ClipadorError:
            raise
        except Exception as e:
            if type(e).__name__ in ("JobCanceled", "WorkerStopping"):
                raise
            raise _classify_gpu_error(e) from e
        return words

    def close(self) -> None:
        self.model = None
        gc.collect()


# --- MLX (Apple Silicon GPU) ----------------------------------------------------------------

# Short names (same as faster-whisper's) -> MLX weights on the Hugging Face hub.
MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def mlx_available() -> bool:
    return is_apple_silicon() and importlib.util.find_spec("mlx_whisper") is not None


def resolve_backend(settings: Settings) -> str:
    """'mlx' (Apple Silicon GPU) or 'faster-whisper' (CUDA on the PC, CPU elsewhere)."""
    want = settings.whisper_backend
    if want == "mlx":
        if not mlx_available():
            raise ClipadorError(
                "CLIPADOR_WHISPER_BACKEND=mlx só funciona num Mac com Apple Silicon (M1/M2/M3/M4) "
                "com o pacote mlx-whisper instalado (`uv sync`)."
            )
        return "mlx"
    if want == "faster-whisper":
        return "faster-whisper"
    return "mlx" if mlx_available() else "faster-whisper"


def resolve_model(settings: Settings, backend: str | None = None) -> str:
    """'auto' = the best model that is fast on this machine: large-v3-turbo on Apple Silicon
    (MLX), medium with faster-whisper (measured on an 8 GB RTX 5060)."""
    if settings.whisper_model != "auto":
        return settings.whisper_model
    backend = backend or resolve_backend(settings)
    return "large-v3-turbo" if backend == "mlx" else "medium"


def mlx_repo(model: str) -> str:
    if "/" in model:  # already a hub repo or a local folder
        return model
    if model not in MLX_REPOS:
        raise ClipadorError(
            f"Modelo Whisper '{model}' não existe para MLX. Use um destes: "
            + ", ".join(sorted(MLX_REPOS))
            + " (ou um repositório mlx-community/...)."
        )
    return MLX_REPOS[model]


def _mlx_module():
    """The mlx_whisper.transcribe MODULE (the package re-exports a function with that name)."""
    return importlib.import_module("mlx_whisper.transcribe")


class _ProgressBar:
    """Stand-in for tqdm.tqdm inside mlx_whisper: forwards progress (and lets the callback raise
    to cancel the job)."""

    def __init__(self, total=None, *_, callback: Callable[[float], None], **__):
        self.total = max(1, int(total or 1))
        self.n = 0
        self.callback = callback

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, n=1):
        self.n += n
        self.callback(min(1.0, self.n / self.total))


class MlxWhisperTranscriber:
    """Whisper on the Apple Silicon GPU (Metal) through mlx-whisper. ~10-30x faster than
    faster-whisper on the Mac CPU. MLX has no VAD filter, so long silences are guarded with
    hallucination_silence_threshold instead."""

    device = "mps"
    compute_type = "float16"

    def __init__(self, settings: Settings, log_cb: Callable[[str], None] = log.info):
        self.settings = settings
        self.log_cb = log_cb
        self.model_name = resolve_model(settings, "mlx")
        self.repo = mlx_repo(self.model_name)
        self.model_path: str | None = None
        self.detected_language: str | None = None

    def _load(self) -> None:
        if self.model_path is not None:
            return
        if Path(self.repo).is_dir():
            self.model_path = self.repo
            return
        self.log_cb(
            f"Carregando modelo '{self.model_name}' (MLX, GPU do Apple Silicon)… na primeira vez "
            "ele é baixado."
        )
        try:
            from huggingface_hub import snapshot_download

            cache = self.settings.models_dir / "mlx"
            cache.mkdir(parents=True, exist_ok=True)
            self.model_path = snapshot_download(self.repo, cache_dir=str(cache))
        except Exception as e:
            raise _classify_gpu_error(e) from e

    def transcribe(
        self, audio_path: Path, language: str | None, progress_cb: Callable[[float], None]
    ) -> list[Word]:
        self._load()
        mt = _mlx_module()
        reader = WavReader(audio_path)
        samples = reader.read(0, reader.duration).astype("float32", copy=False)
        original_tqdm = mt.tqdm

        class _Shim:
            @staticmethod
            def tqdm(*a, **k):
                return _ProgressBar(*a, callback=progress_cb, **k)

        mt.tqdm = _Shim
        try:
            result = mt.transcribe(
                samples,
                path_or_hf_repo=self.model_path,
                language=language,
                word_timestamps=True,
                condition_on_previous_text=False,
                verbose=False,
                hallucination_silence_threshold=(2.0 if self.settings.whisper_vad_filter else None),
            )
        except ClipadorError:
            raise
        except Exception as e:
            if type(e).__name__ in ("JobCanceled", "WorkerStopping"):
                raise
            raise _classify_gpu_error(e) from e
        finally:
            mt.tqdm = original_tqdm
        self.detected_language = result.get("language")
        words: list[Word] = []
        for seg in result.get("segments", []):
            for w in seg.get("words") or []:
                text = str(w.get("word", "")).strip()
                if text:
                    words.append(
                        Word(float(w["start"]), float(w["end"]), text,
                             float(w.get("probability", 1.0)))
                    )  # fmt: skip
        progress_cb(1.0)
        return words

    def close(self) -> None:
        """Free the model and MLX's buffer cache (unified memory goes back to Ollama/the OS)."""
        with contextlib.suppress(Exception):
            import mlx.core as mx

            mt = _mlx_module()
            mt.ModelHolder.model = None
            mt.ModelHolder.model_path = None
            gc.collect()
            mx.clear_cache()


# --- another computer's GPU (services/transcription_server.py) --------------------------------

_LOCAL_LOCK = threading.Lock()  # remote jobs run in parallel: a local fallback goes one at a time
_REMOTE_DOWN_UNTIL = [0.0]  # after a failure, skip the remote for a while (no waiting each time)


def remote_up(settings: Settings, timeout: float = 3.0) -> bool:
    """Is the other computer's transcription server answering (and the token right)?"""
    if not settings.remote_whisper_url or time.monotonic() < _REMOTE_DOWN_UNTIL[0]:
        return False
    import httpx

    try:
        auth = {"Authorization": f"Bearer {settings.remote_whisper_token}"}
        url = settings.remote_whisper_url.rstrip("/") + "/health"
        r = httpx.get(url, timeout=timeout, headers=auth)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


class RemoteTranscriber:
    """Sends each chunk to the transcription server on another computer (e.g. the Windows PC
    with an RTX). If it doesn't answer, transcribes here instead (and tries the remote again a
    few minutes later)."""

    device = "remoto"
    compute_type = "remoto"

    def __init__(self, settings: Settings, log_cb: Callable[[str], None] = log.info):
        self.settings, self.log_cb = settings, log_cb
        self.detected_language: str | None = None
        self._local: Transcriber | None = None

    def transcribe(
        self, audio_path: Path, language: str | None, progress_cb: Callable[[float], None]
    ) -> list[Word]:
        import httpx

        url = self.settings.remote_whisper_url.rstrip("/") + "/transcribe"
        if time.monotonic() >= _REMOTE_DOWN_UNTIL[0]:
            t0 = time.monotonic()
            try:
                r = httpx.post(url, content=audio_path.read_bytes(),
                               params={"language": language or ""}, timeout=900,
                               headers={"Authorization":
                                        f"Bearer {self.settings.remote_whisper_token}",
                                        "Content-Type": "audio/wav"})  # fmt: skip
                if r.status_code == 401:
                    raise ClipadorError(
                        "O servidor de transcrição recusou o token. Confira "
                        "CLIPADOR_REMOTE_WHISPER_TOKEN com o que o PC mostrou."
                    )
                r.raise_for_status()
                data = r.json()
                self.detected_language = data.get("language")
                dt = time.monotonic() - t0
                self.log_cb(f"Transcrito no outro computador em {dt:.0f}s.")
                progress_cb(1.0)
                return [Word(float(a), float(b), str(w), float(p)) for a, b, w, p in data["words"]]
            except (httpx.HTTPError, ValueError, KeyError) as e:
                _REMOTE_DOWN_UNTIL[0] = time.monotonic() + 300
                self.log_cb(f"Outro computador indisponível ({type(e).__name__}); "
                            "transcrevendo aqui.")  # fmt: skip
        with _LOCAL_LOCK:
            if self._local is None:
                self._local = _local_transcriber(self.settings, self.log_cb)
            words = self._local.transcribe(audio_path, language, progress_cb)
            self.detected_language = getattr(self._local, "detected_language", None)
            return words

    def close(self) -> None:
        if self._local is not None and hasattr(self._local, "close"):
            self._local.close()


def _local_transcriber(settings: Settings, log_cb: Callable[[str], None]) -> Transcriber:
    if resolve_backend(settings) == "mlx":
        return MlxWhisperTranscriber(settings, log_cb)
    return FasterWhisperTranscriber(settings, log_cb)


def make_transcriber(settings: Settings, log_cb: Callable[[str], None]) -> Transcriber:
    """Factory used by the transcribe stage (tests monkeypatch this)."""
    if settings.remote_whisper_url:
        return RemoteTranscriber(settings, log_cb)
    return _local_transcriber(settings, log_cb)


def realtime_factor(audio_seconds: float, started_monotonic: float) -> float:
    return audio_seconds / max(1e-6, time.monotonic() - started_monotonic)
