"""Facecam auto-detection glue (step 5C): pick the source, run the detector, save the preview,
fill the streamer's box. A box marked by hand (`facecam_auto=False`) is never overwritten
unless the user explicitly asks for it."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlmodel import Session, col, select

from clipador.config import Settings
from clipador.errors import ClipadorError
from clipador.models import Source, SourceStatus, Streamer
from clipador.render import facecam_detect as fd
from clipador.services import ffmpeg
from clipador.services.sources import touch

log = logging.getLogger(__name__)

PREVIEW_NAME = "facecam_preview.png"
NOT_FOUND_HINT = "Facecam não detectada, marque à mão se quiser."


class Status:
    DETECTED = "detected"  # found and saved on the streamer
    FOUND_KEPT_MANUAL = "found_kept_manual"  # found, but the hand-marked box was kept
    NOT_FOUND = "not_found"
    KEPT_EXISTING = "kept_existing"  # automatic run skipped: the streamer already has a box
    DISABLED = "disabled"


@dataclass
class FacecamOutcome:
    status: str
    message: str
    result: fd.DetectionResult | None = None
    preview: Path | None = None
    saved: bool = False


def latest_ready_source(session: Session, streamer_id: int, settings: Settings) -> Source | None:
    """Most recent downloaded source of the streamer whose file is still on disk."""
    sources = session.exec(
        select(Source)
        .where(Source.streamer_id == streamer_id, Source.status == SourceStatus.READY)
        .order_by(col(Source.id).desc())
    ).all()
    for src in sources:
        if src.file_path and settings.abs_path(src.file_path).is_file():
            return src
    return None


def find_streamer(session: Session, name: str) -> Streamer | None:
    """By display name or channel handle, case-insensitive."""
    key = name.strip().lstrip("@").casefold()
    streamers = session.exec(select(Streamer)).all()
    for st in streamers:
        if st.display_name.casefold() == key:
            return st
    for st in streamers:
        if any(ch.channel_handle.casefold() == key for ch in st.channels):
            return st
    return None


def detect_for_source(
    settings: Settings,
    source: Source,
    detector: fd.FaceDetector | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> tuple[fd.DetectionResult, list[np.ndarray]]:
    """Sample frames of the source (honoring its analysis range) and run the detector."""
    if not source.file_path or not settings.abs_path(source.file_path).is_file():
        raise ClipadorError("O vídeo desta fonte não está baixado.")
    video = settings.abs_path(source.file_path)
    info = source.media_info or {}
    width, height = info.get("width"), info.get("height")
    duration = source.duration_seconds or info.get("duration")
    if not (width and height and duration):
        probed = ffmpeg.probe(video, settings.ffprobe_path)
        width, height, duration = probed.width, probed.height, probed.duration
    times = fd.sample_times(
        float(duration), settings.facecam_samples, source.start_offset, source.end_offset
    )
    frames = fd.sample_frames(
        video, times, int(width or 1920), int(height or 1080), ffmpeg=settings.ffmpeg_path
    )
    if not frames:
        raise ClipadorError("Não consegui ler quadros do vídeo para procurar a facecam.")
    det = detector or fd.make_face_detector(
        settings.models_dir, download=settings.facecam_model_download, log_cb=log_cb
    )
    return fd.analyze_frames(frames, det, settings.facecam_min_confidence), frames


def preview_path(settings: Settings, source_id: int) -> Path:
    return settings.source_work_dir(source_id) / PREVIEW_NAME


def save_preview(
    settings: Settings, source_id: int, frames: list[np.ndarray], result: fd.DetectionResult
) -> Path:
    """PNG of a middle frame with the detected box drawn (data/work/<source>/...)."""
    frame = frames[len(frames) // 2]
    return fd.write_png(preview_path(settings, source_id), fd.draw_detection(frame, result))


def is_manual(streamer: Streamer) -> bool:
    return streamer.facecam_rect is not None and not streamer.facecam_auto


def apply_detection(
    streamer: Streamer, result: fd.DetectionResult, *, overwrite_manual: bool = False
) -> bool:
    """Write the detection on the streamer. Returns True if something changed.

    - hand-marked box: untouched unless `overwrite_manual`;
    - nothing found: an existing box is kept; an empty one records the (low) confidence so the
      page can say "facecam não detectada"."""
    if is_manual(streamer) and not overwrite_manual:
        return False
    if result.box is None:
        if streamer.facecam_rect is not None:
            return False
        streamer.facecam_box = None
        streamer.facecam_auto = True
        streamer.facecam_confidence = round(result.confidence, 3)
    else:
        streamer.facecam_box = result.box.as_list()
        streamer.facecam_auto = True
        streamer.facecam_confidence = round(result.box.confidence, 3)
    touch(streamer)
    return True


def _describe(result: fd.DetectionResult) -> str:
    if result.box is None:
        return f"{NOT_FOUND_HINT} (confiança {result.confidence:.0%})"
    return (
        f"Facecam detectada automaticamente ({result.box.confidence:.0%} de confiança, "
        f"{result.frames}/{result.total_frames} quadros)."
    )


def run_detection(
    session: Session,
    settings: Settings,
    source: Source,
    *,
    auto: bool = False,
    overwrite_manual: bool = False,
    save: bool = True,
    preview: bool = True,
    detector: fd.FaceDetector | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> FacecamOutcome:
    """Detect the facecam on `source` and (if allowed) save it on its streamer.

    `auto=True` is the pipeline call: it only runs when enabled and the streamer has no box."""
    streamer = session.get(Streamer, source.streamer_id)
    if streamer is None:
        raise ClipadorError("Streamer da fonte não encontrado.")
    if auto:
        if not settings.facecam_auto_detect:
            return FacecamOutcome(Status.DISABLED, "Detecção automática da facecam desligada.")
        if streamer.facecam_rect is not None:
            return FacecamOutcome(Status.KEPT_EXISTING, "O streamer já tem facecam definida.")
    result, frames = detect_for_source(settings, source, detector, log_cb)
    png = save_preview(settings, source.id, frames, result) if preview else None
    saved = False
    if save:
        saved = apply_detection(streamer, result, overwrite_manual=overwrite_manual)
        if saved:
            session.add(streamer)
            session.commit()
    msg = _describe(result)
    if result.box is None:
        status = Status.NOT_FOUND
        if streamer.facecam_rect is not None:
            msg += " A facecam que já estava salva foi mantida."
    elif is_manual(streamer) and not saved:
        status = Status.FOUND_KEPT_MANUAL
        msg += " A caixa marcada à mão foi mantida (peça para substituir se quiser)."
    else:
        status = Status.DETECTED
    return FacecamOutcome(status, msg, result=result, preview=png, saved=saved)


def autodetect_in_stage(ctx) -> None:
    """Called at the end of ingest / audio extraction. Never fails the stage."""
    settings = ctx.settings
    if not settings.facecam_auto_detect or ctx.source_id is None:
        return
    try:
        with ctx.session() as s:
            source = s.get(Source, ctx.source_id)
            if source is None or source.status != SourceStatus.READY:
                return
            out = run_detection(s, settings, source, auto=True, log_cb=ctx.log)
    except Exception as e:  # detection is a convenience: log and move on
        log.warning("facecam auto-detection failed: %s", e, exc_info=True)
        msg = getattr(e, "message", None) or str(e)
        ctx.log(f"AVISO: a detecção automática da facecam falhou ({msg}); marque à mão se quiser.")
        return
    if out.status not in (Status.KEPT_EXISTING, Status.DISABLED):
        ctx.log(out.message)


def outcome_text(out: FacecamOutcome, source: Source, settings: Settings) -> str:
    """Terminal report for `clipador facecam`."""
    r = out.result
    lines = [f"Fonte #{source.id} · streamer #{source.streamer_id}"]
    if r is not None:
        lines.append(
            f"detector: {r.detector or '?'} · rosto em {r.frames}/{r.total_frames} quadros"
        )
        if r.box is not None:
            b = r.box
            lines.append(
                f"facecam: x={b.x:.4f} y={b.y:.4f} largura={b.w:.4f} altura={b.h:.4f} "
                f"· confiança {b.confidence:.0%} ({b.method})"
            )
        else:
            lines.append(f"facecam: não detectada (confiança {r.confidence:.0%})")
    lines.append(out.message)
    if out.saved:
        lines.append("Salvo no streamer.")
    if out.preview is not None:
        lines.append(f"Prévia: {out.preview}")
    return "\n".join(lines)
