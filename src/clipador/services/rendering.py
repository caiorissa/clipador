"""Render planning + execution (DB/files side). Filter graphs and ASS come from clipador.render."""

from __future__ import annotations

import contextlib
import shutil
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session, select

from clipador.config import Settings
from clipador.errors import ClipadorError
from clipador.llm.drafts import DraftData, build_drafts
from clipador.llm.schema import truncate_words
from clipador.models import (
    Candidate,
    CandidateEnrichment,
    EnrichStatus,
    PostDraft,
    RenderedClip,
    Source,
    Streamer,
)
from clipador.render import filters
from clipador.render.package import package_text
from clipador.render.styles import STYLES, get_style
from clipador.render.subtitles import (
    SubWord,
    align_edited_text,
    build_ass,
    words_in_window,
)
from clipador.services import ffmpeg
from clipador.services.artifacts import get_artifact, params_hash
from clipador.services.transcriber import load_transcript
from clipador.utils import format_hms, utcnow

VERSION = 1
TITLE_MAX = 42
LAYOUT_LABELS = {"blur": "Fundo borrado", "facecam": "Facecam embaixo"}


@lru_cache(maxsize=4)
def _auto_encoder(ffmpeg_path: str) -> str:
    return ffmpeg.pick_encoder(ffmpeg_path)


def resolve_encoder(settings: Settings) -> str:
    if settings.render_encoder == "x264":
        return ffmpeg.X264
    if settings.render_encoder == "nvenc":
        return ffmpeg.NVENC
    return _auto_encoder(settings.ffmpeg_path)


def transcript_words(engine: Engine, settings: Settings, source_id: int) -> list[SubWord]:
    art = get_artifact(engine, source_id, "transcribe")
    if art is None or "transcript" not in art.files:
        return []
    path = settings.abs_path(art.files["transcript"])
    if not path.is_file():
        return []
    return [(w.start, w.end, w.word) for w in load_transcript(path)[0]]


def suggested_title(e: CandidateEnrichment | None, yt: PostDraft | None) -> str | None:
    if e and e.status == EnrichStatus.OK and e.texto_de_abertura:
        return truncate_words(e.texto_de_abertura, TITLE_MAX)
    if yt and yt.title:
        return truncate_words(yt.title, TITLE_MAX)
    return None


@dataclass
class RenderPlan:
    candidate_id: int
    source_id: int
    video: Path
    start: float
    end: float
    src_w: int
    src_h: int
    layout: str  # effective layout
    style: str
    title: str | None
    words: list[SubWord]  # relative to the clip start
    facecam_rect: tuple[int, int, int, int] | None
    zoom_peak: float | None
    phash: str
    clip_name: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def plan_render(session: Session, engine: Engine, settings: Settings, c: Candidate) -> RenderPlan:
    source = session.get(Source, c.source_id)
    if source is None or not source.file_path:
        raise ClipadorError("O vídeo desta fonte não está baixado.")
    video = settings.abs_path(source.file_path)
    if not video.is_file():
        raise ClipadorError("Arquivo de vídeo da fonte não encontrado; baixe a fonte de novo.")
    streamer = session.get(Streamer, source.streamer_id)
    info = source.media_info or {}
    src_w, src_h = int(info.get("width") or 1920), int(info.get("height") or 1080)

    start, end = c.eff_start, c.eff_end
    raw = words_in_window(transcript_words(engine, settings, source.id), start, end)
    words = align_edited_text(raw, c.subtitle_text) if c.subtitle_text is not None else raw

    rect = streamer.facecam_rect if streamer else None
    wanted = c.render_layout or (streamer.default_layout if streamer else "blur")
    layout = "facecam" if wanted == "facecam" and rect else "blur"
    style = c.render_style if c.render_style in STYLES else settings.render_style

    if c.overlay_title is not None:
        title = c.overlay_title.strip() or None
    else:
        e = session.exec(
            select(CandidateEnrichment).where(CandidateEnrichment.candidate_id == c.id)
        ).first()
        yt = session.exec(
            select(PostDraft).where(PostDraft.candidate_id == c.id, PostDraft.platform == "youtube")
        ).first()
        title = suggested_title(e, yt)

    peak = c.peak_t - start if start <= c.peak_t <= end and settings.zoom_amount > 0 else None
    phash = params_hash(
        {
            "v": VERSION, "file": source.file_path, "start": start, "end": end, "layout": layout,
            "style": style, "title": title, "words": words, "rect": rect, "peak": peak,
            "zoom": settings.zoom_amount, "fps": settings.render_fps,
        }
    )  # fmt: skip
    name = f"{c.rank:02d}_{format_hms(start).replace(':', '')}"
    return RenderPlan(c.id, source.id, video, start, end, src_w, src_h, layout, style, title,
                      words, rect if layout == "facecam" else None, peak, phash, name)  # fmt: skip


def _drafts_for_package(session: Session, settings: Settings, c: Candidate) -> tuple[dict, list]:
    rows = session.exec(select(PostDraft).where(PostDraft.candidate_id == c.id)).all()
    drafts = {r.platform: DraftData(r.title, r.description, list(r.hashtags)) for r in rows}
    e = session.exec(
        select(CandidateEnrichment).where(CandidateEnrichment.candidate_id == c.id)
    ).first()
    if not drafts:  # never enriched: simple drafts so the credit is always there
        source = session.get(Source, c.source_id)
        streamer = session.get(Streamer, source.streamer_id)
        drafts = build_drafts(
            None, streamer=streamer.display_name, credit_text=streamer.credit_text,
            source_platform=source.platform, source_url=source.url,
            game=source.game or settings.default_game, snippet=c.transcript_snippet,
            start=c.eff_start,
        )  # fmt: skip
    alerts = list(e.alertas) if e and e.status == EnrichStatus.OK else []
    return drafts, alerts


def render_clip(
    session: Session, settings: Settings, c: Candidate, plan: RenderPlan, progress_cb, log
) -> RenderedClip:
    """Render (or reuse) the vertical clip of `c`. Idempotent by plan.phash."""
    existing = session.exec(select(RenderedClip).where(RenderedClip.candidate_id == c.id)).first()
    if (
        existing
        and existing.params_hash == plan.phash
        and settings.abs_path(existing.file_path).is_file()
    ):
        log(f"#{c.rank}: corte já renderizado com estes ajustes; nada a fazer.")
        return existing

    out_dir = settings.clips_dir / str(plan.source_id)
    work = settings.tmp_dir / f"render_{c.id}"
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = resolve_encoder(settings)
    graph, sub_layout = filters.vertical_filter(
        plan.layout, plan.src_w, plan.src_h, plan.duration,
        filters.Zoom(plan.zoom_peak, settings.zoom_amount),
        fps=settings.render_fps, facecam_rect=plan.facecam_rect,
    )  # fmt: skip
    (work / filters.SUBS_FILE).write_text(
        build_ass(plan.words, plan.duration, get_style(plan.style), sub_layout, plan.title),
        "utf-8",
    )
    tmp_out = work / "out.mp4"
    args = ["-ss", f"{plan.start:.3f}", "-t", f"{plan.duration:.3f}", "-i", str(plan.video),
            "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
            *filters.encoder_args(encoder), str(tmp_out)]  # fmt: skip
    t0 = time.monotonic()
    log(f"#{c.rank}: renderizando {plan.duration:.0f}s ({LAYOUT_LABELS[plan.layout]}, legenda "
        f"'{plan.style}', {encoder})…")  # fmt: skip
    try:
        ffmpeg.run_with_progress(
            args, duration=plan.duration, progress_cb=progress_cb, cwd=work,
            ffmpeg=settings.ffmpeg_path, error_message="Falha ao renderizar o corte.",
        )  # fmt: skip
    except ClipadorError as e:
        if encoder == ffmpeg.NVENC and "nvenc" in (e.detail or "").lower():
            log("NVENC falhou; tentando de novo com libx264 (CPU).")
            args = [*args[: args.index("-c:v")], *filters.encoder_args(ffmpeg.X264), str(tmp_out)]
            encoder = ffmpeg.X264
            ffmpeg.run_with_progress(
                args, duration=plan.duration, progress_cb=progress_cb, cwd=work,
                ffmpeg=settings.ffmpeg_path, error_message="Falha ao renderizar o corte.",
            )  # fmt: skip
        else:
            raise
    elapsed = time.monotonic() - t0

    final = out_dir / f"{plan.clip_name}_{plan.phash[:8]}.mp4"
    shutil.move(str(tmp_out), final)
    thumb = final.with_suffix(".jpg")
    with contextlib.suppress(ClipadorError):
        ffmpeg.make_thumbnail(
            final, thumb, plan.zoom_peak or plan.duration / 3, settings.ffmpeg_path
        )
    drafts, alerts = _drafts_for_package(session, settings, c)
    pkg = final.with_suffix(".txt")
    pkg.write_text(package_text(drafts, final.name, alerts), "utf-8")
    shutil.rmtree(work, ignore_errors=True)

    if existing is not None and existing.file_path != settings.rel_path(final):
        for old in (existing.file_path, existing.package_path, existing.thumbnail_path):
            if old:
                with contextlib.suppress(OSError):
                    settings.abs_path(old).unlink()
    row = existing or RenderedClip(candidate_id=c.id, source_id=plan.source_id, params_hash="",
                                   layout="", style="", file_path="", duration_seconds=0,
                                   encoder="")  # fmt: skip
    row.params_hash, row.layout, row.style = plan.phash, plan.layout, plan.style
    row.file_path = settings.rel_path(final)
    row.package_path = settings.rel_path(pkg)
    row.thumbnail_path = settings.rel_path(thumb) if thumb.is_file() else None
    row.duration_seconds, row.encoder = plan.duration, encoder
    row.render_seconds, row.updated_at = round(elapsed, 1), utcnow()
    session.add(row)
    session.commit()
    log(f"#{c.rank}: pronto em {elapsed:.0f}s → {final.name} "
        f"({plan.duration / max(elapsed, 0.01):.1f}x tempo real).")  # fmt: skip
    return row


# --- 16:9 compilation of the approved clips ----------------------------------------------------


def compile_long(
    session: Session, settings: Settings, plans: list[RenderPlan], progress_cb, log
) -> tuple[Path, Path, str]:
    """Concatenate approved moments (chronological) into a 1920x1080 video with subtitles and a
    chapters file for the YouTube description. Returns (video, chapters, hash)."""
    if not plans:
        raise ClipadorError("Aprove pelo menos um candidato para montar o vídeo longo.")
    plans = sorted(plans, key=lambda p: p.start)
    source_id = plans[0].source_id
    phash = params_hash({"v": VERSION, "long": [p.phash for p in plans]})
    out_dir = settings.clips_dir / str(source_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"longo_{phash[:8]}.mp4"
    chapters = final.with_suffix(".txt")
    if final.is_file() and chapters.is_file():
        log("Vídeo longo já montado com estes cortes; nada a fazer.")
        return final, chapters, phash

    work = settings.tmp_dir / f"long_{source_id}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    encoder = resolve_encoder(settings)
    total = sum(p.duration for p in plans)
    done = 0.0
    seg_files, chapter_lines, t_cursor = [], [], 0.0
    for i, p in enumerate(plans):
        seg_dir = work / f"seg{i:02d}"
        seg_dir.mkdir()
        (seg_dir / filters.SUBS_FILE).write_text(
            build_ass(p.words, p.duration, get_style(p.style), filters.LAYOUT_LONG, p.title),
            "utf-8",
        )
        seg = work / f"seg{i:02d}.mp4"
        graph = filters.horizontal_filter(p.src_w, p.src_h, p.duration, settings.render_fps)
        args = ["-ss", f"{p.start:.3f}", "-t", f"{p.duration:.3f}", "-i", str(p.video),
                "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
                *filters.encoder_args(encoder), str(seg)]  # fmt: skip
        log(f"Vídeo longo: trecho {i + 1}/{len(plans)} ({format_hms(p.start)}).")
        ffmpeg.run_with_progress(
            args, duration=p.duration, cwd=seg_dir, ffmpeg=settings.ffmpeg_path,
            progress_cb=lambda f, b=done, d=p.duration: progress_cb((b + f * d) / total * 0.95),
            error_message="Falha ao montar um trecho do vídeo longo.",
        )  # fmt: skip
        done += p.duration
        seg_files.append(seg)
        chapter_lines.append(f"{format_hms(t_cursor)} {p.title or f'Momento {i + 1}'}")
        t_cursor += p.duration
    lst = work / "list.txt"
    lst.write_text("".join(f"file '{s.name}'\n" for s in seg_files), "utf-8")
    ffmpeg.run_with_progress(
        ["-f", "concat", "-safe", "0", "-i", lst.name, "-c", "copy", "-movflags", "+faststart",
         str(final)],
        duration=total, cwd=work, ffmpeg=settings.ffmpeg_path, progress_cb=lambda f: None,
        error_message="Falha ao juntar os trechos do vídeo longo.",
    )  # fmt: skip
    if chapter_lines and not chapter_lines[0].startswith("00:00:00"):
        chapter_lines[0] = "00:00:00 " + chapter_lines[0].split(" ", 1)[1]
    chapters.write_text(
        "Capítulos (cole na descrição do YouTube):\n"
        + "\n".join(line.replace("00:", "", 1) if line.startswith("00:") else line
                    for line in chapter_lines)
        + "\n",
        "utf-8",
    )  # fmt: skip
    shutil.rmtree(work, ignore_errors=True)
    progress_cb(1.0)
    return final, chapters, phash
