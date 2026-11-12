"""Speech-to-text behind a small Protocol, plus chunk planning/merging (pure functions).

Tests use a fake Transcriber; nothing here needs a GPU or model unless FasterWhisperTranscriber
is actually used.
"""

from __future__ import annotations

import gc
import gzip
import json
import logging
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
            f"Carregando modelo '{self.settings.whisper_model}' ({self.device}, "
            f"{self.compute_type})… na primeira vez ele é baixado."
        )
        try:
            from faster_whisper import WhisperModel

            self.settings.models_dir.mkdir(parents=True, exist_ok=True)
            self.model = WhisperModel(
                self.settings.whisper_model,
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


def make_transcriber(settings: Settings, log_cb: Callable[[str], None]) -> Transcriber:
    """Factory used by the transcribe stage (tests monkeypatch this)."""
    return FasterWhisperTranscriber(settings, log_cb)


def realtime_factor(audio_seconds: float, started_monotonic: float) -> float:
    return audio_seconds / max(1e-6, time.monotonic() - started_monotonic)
