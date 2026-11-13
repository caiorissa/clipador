"""Render planning + execution (DB/files side). Graphs, ASS and audio filters are pure functions
in clipador.render.*; this module wires them to the database, ffmpeg and the file system."""

from __future__ import annotations

import contextlib
import shutil
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlmodel import Session, col, select

from clipador.config import Settings
from clipador.errors import ClipadorError, ToolError
from clipador.llm.drafts import DraftData, build_drafts
from clipador.llm.schema import truncate_words
from clipador.models import (
    Candidate,
    CandidateEnrichment,
    CandidateStatus,
    EnrichStatus,
    Job,
    PostDraft,
    Render,
    RenderKind,
    RenderStatus,
    Source,
    Streamer,
)
from clipador.render import audio as raudio
from clipador.render import filters
from clipador.render.package import package_text
from clipador.render.styles import DEFAULT_FONTS_DIR, get_style
from clipador.render.subtitles import (
    IWord,
    Overlays,
    apply_overrides,
    build_ass,
    credit_overlay,
    safe_zone,
    words_in_window,
)
from clipador.render.validate import validate_render
from clipador.services import ffmpeg
from clipador.services.artifacts import get_artifact, params_hash
from clipador.services.transcriber import load_transcript
from clipador.utils import format_hms, utcnow

VERSION = 2
TITLE_MAX = 42
LAYOUT_LABELS = filters.LAYOUT_LABELS


@lru_cache(maxsize=4)
def _auto_encoder(ffmpeg_path: str) -> str:
    return ffmpeg.pick_encoder(ffmpeg_path)


def resolve_encoder(settings: Settings) -> str:
    if settings.render_encoder == "x264":
        return ffmpeg.X264
    if settings.render_encoder == "nvenc":
        return ffmpeg.NVENC
    return _auto_encoder(settings.ffmpeg_path)


def fonts_dir(settings: Settings) -> Path:
    return settings.fonts_dir or DEFAULT_FONTS_DIR


def transcript_words(engine: Engine, settings: Settings, source_id: int) -> list:
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


# --- planning ---------------------------------------------------------------------------------


@dataclass
class RenderPlan:
    candidate_id: int
    source_id: int
    kind: str
    video: Path
    start: float
    end: float
    src_w: int
    src_h: int
    fps: float
    layout: str
    style_key: str
    words: list[IWord]
    rect: tuple[float, float, float, float] | None
    opts: filters.LayoutOptions
    zoom_peak: float | None
    overlays: Overlays
    clip_name: str
    warnings: list[str] = field(default_factory=list)
    phash: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


def out_size(settings: Settings, kind: str) -> tuple[int, int]:
    if kind == RenderKind.PREVIEW:
        return settings.preview_width, settings.preview_height
    return filters.OUT_W, filters.OUT_H


def plan_render(
    session: Session, engine: Engine, settings: Settings, c: Candidate, kind: str = "final"
) -> RenderPlan:
    source = session.get(Source, c.source_id)
    if source is None or not source.file_path:
        raise ClipadorError("O vídeo desta fonte não está baixado.")
    video = settings.abs_path(source.file_path)
    if not video.is_file():
        raise ClipadorError("Arquivo de vídeo da fonte não encontrado; baixe a fonte de novo.")
    streamer = session.get(Streamer, source.streamer_id)
    info = source.media_info or {}
    src_w, src_h = int(info.get("width") or 1920), int(info.get("height") or 1080)
    fps = float(info.get("fps") or 30)

    start, end = c.eff_start, c.eff_end
    words = apply_overrides(
        words_in_window(transcript_words(engine, settings, source.id), start, end),
        c.caption_overrides,
    )
    rect = streamer.facecam_rect if streamer else None
    wanted = c.render_layout or (streamer.default_layout if streamer else "blur")
    layout, layout_warning = filters.resolve_layout(wanted, rect)
    style_key = (
        c.render_style
        or (streamer.default_caption_style if streamer else None)
        or settings.render_style
    )
    style = get_style(style_key, settings.caption_styles_path)
    o = c.render_options or {}
    opts = filters.LayoutOptions(zoom=float(o.get("zoom", 1.0)), foco_x=float(o.get("foco_x", 0.5)))

    show_opening = o.get("opening", streamer.show_opening_text if streamer else True)
    show_credit = o.get("credit", streamer.show_credit if streamer else True)
    opening = None
    if show_opening:
        if c.overlay_title is not None:
            opening = c.overlay_title.strip() or None
        else:
            e = session.exec(
                select(CandidateEnrichment).where(CandidateEnrichment.candidate_id == c.id)
            ).first()
            yt = session.exec(
                select(PostDraft).where(
                    PostDraft.candidate_id == c.id, PostDraft.platform == "youtube"
                )
            ).first()
            opening = suggested_title(e, yt)
    credit = (
        credit_overlay(streamer.credit_text, streamer.display_name)
        if show_credit and streamer
        else None
    )
    peak_on = o.get("peak_zoom", True) and settings.zoom_amount > 0
    peak = c.peak_t - start if peak_on and start <= c.peak_t <= end else None
    overlays = Overlays(opening, settings.opening_text_seconds, credit)

    warnings = [layout_warning] if layout_warning else []
    if end - start > settings.platform_max_seconds:
        warnings.append(
            f"O clip tem {end - start:.0f}s: passa de {settings.platform_max_seconds:.0f}s "
            "(limite comum de Shorts/Reels)."
        )
    params = {
        "v": VERSION, "kind": kind, "file": source.file_path, "start": start, "end": end,
        "layout": layout, "style": asdict(style), "words": [asdict(w) for w in words],
        "rect": rect, "opts": asdict(opts), "peak": peak, "zoom": settings.zoom_amount,
        "overlays": asdict(overlays), "size": out_size(settings, kind),
        "loudness": [settings.loudness_target, settings.loudness_true_peak],
        "fade": settings.audio_fade_seconds, "min_event": settings.min_caption_event,
        "safe": [settings.safe_bottom, settings.safe_top, settings.caption_lift],
        "max_fps": settings.render_max_fps,
    }  # fmt: skip
    name = f"{c.rank:02d}_{format_hms(start).replace(':', '')}"
    return RenderPlan(
        c.id, source.id, kind, video, start, end, src_w, src_h, fps, layout, style.key, words,
        rect if layout == "facecam_embaixo" else None, opts, peak, overlays, name, warnings,
        params_hash(params), params,
    )  # fmt: skip


# --- requesting (web) -------------------------------------------------------------------------


def request_renders(
    session: Session, engine: Engine, settings: Settings, candidate_ids: list[int], kind: str
) -> tuple[list[Render], list[Render]]:
    """Create (or reuse) Render rows. Returns (to_render, reused_done)."""
    todo, reused = [], []
    for cid in candidate_ids:
        c = session.get(Candidate, cid)
        if c is None:
            continue
        plan = plan_render(session, engine, settings, c, kind)
        existing = session.exec(
            select(Render).where(
                Render.candidate_id == cid, Render.kind == kind, Render.params_hash == plan.phash
            )
        ).all()
        done = next(
            (r for r in existing
             if r.status == RenderStatus.DONE and r.path and settings.abs_path(r.path).is_file()),
            None,
        )  # fmt: skip
        if done is not None:
            reused.append(done)
            continue
        pending = next(
            (r for r in existing if r.status in (RenderStatus.QUEUED, RenderStatus.RUNNING)), None
        )
        if pending is not None:
            todo.append(pending)
            continue
        row = Render(candidate_id=cid, source_id=c.source_id, kind=kind, params_hash=plan.phash,
                     params=plan.params, warning=" ".join(plan.warnings) or None)  # fmt: skip
        session.add(row)
        session.flush()
        todo.append(row)
    return todo, reused


# --- executing (worker) --------------------------------------------------------------------------


def _drafts_for_package(session: Session, settings: Settings, c: Candidate) -> tuple[dict, list]:
    rows = session.exec(select(PostDraft).where(PostDraft.candidate_id == c.id)).all()
    drafts = {r.platform: DraftData(r.title, r.description, list(r.hashtags)) for r in rows}
    e = session.exec(
        select(CandidateEnrichment).where(CandidateEnrichment.candidate_id == c.id)
    ).first()
    if not drafts:
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


def prepare_workdir(work: Path, settings: Settings, ass: str) -> None:
    """subs.ass + fonts/ inside ffmpeg's cwd, referenced by relative names only."""
    shutil.rmtree(work, ignore_errors=True)
    (work / filters.FONTS_SUBDIR).mkdir(parents=True)
    (work / filters.SUBS_FILE).write_text(ass, "utf-8")
    for f in fonts_dir(settings).glob("*.[ot]tf"):
        shutil.copy2(f, work / filters.FONTS_SUBDIR / f.name)


def build_args(plan: RenderPlan, settings: Settings, encoder: str, audio_filter: str,
               out: Path) -> list[str]:  # fmt: skip
    graph = filters.vertical_graph(
        plan.layout, plan.src_w, plan.src_h, plan.duration,
        filters.Zoom(plan.zoom_peak, settings.zoom_amount), audio_filter,
        rect=plan.rect, opts=plan.opts, out_size=out_size(settings, plan.kind),
    )  # fmt: skip
    return ["-ss", f"{plan.start:.3f}", "-t", f"{plan.duration:.3f}", "-i", str(plan.video),
            "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
            *filters.output_rate_args(plan.fps, settings.render_max_fps),
            *filters.encoder_args(encoder, preview=plan.kind == RenderKind.PREVIEW),
            str(out)]  # fmt: skip


def run_encode(args_for, settings: Settings, encoder: str, plan: RenderPlan, work: Path,
               progress_cb, log) -> str:  # fmt: skip
    """Run ffmpeg; on an NVENC failure retry once with libx264. Returns the encoder used."""
    try:
        ffmpeg.run_with_progress(
            args_for(encoder), duration=plan.duration, progress_cb=progress_cb, cwd=work,
            ffmpeg=settings.ffmpeg_path, error_message="Falha ao renderizar o corte.",
        )  # fmt: skip
        return encoder
    except ToolError as e:
        if encoder != ffmpeg.NVENC:
            raise
        log(f"NVENC falhou ({(e.detail or '')[-160:]}); tentando de novo com libx264 (CPU).")
        ffmpeg.run_with_progress(
            args_for(ffmpeg.X264), duration=plan.duration, progress_cb=progress_cb, cwd=work,
            ffmpeg=settings.ffmpeg_path, error_message="Falha ao renderizar o corte.",
        )  # fmt: skip
        return ffmpeg.X264


def execute_render(
    session: Session, settings: Settings, row: Render, plan: RenderPlan, progress_cb, log,
    job_id: int | None = None,
) -> Render:  # fmt: skip
    """Render `row`. Writes to a temp name and renames at the end: a failed or canceled render
    never leaves a broken file with the final name."""
    c = session.get(Candidate, row.candidate_id)
    row.status, row.job_id, row.error, row.updated_at = RenderStatus.RUNNING, job_id, None, utcnow()
    session.add(row)
    session.commit()

    kind_label = "prévia" if plan.kind == RenderKind.PREVIEW else "final"
    work = settings.tmp_dir / f"render_{row.id}"
    tmp_out = work / "out.tmp.mp4"
    t0 = time.monotonic()
    try:
        encoder = resolve_encoder(settings)
        gain = None
        if plan.kind == RenderKind.FINAL:
            gain = raudio.final_gain(
                plan.video, plan.start, plan.duration, settings.loudness_target,
                settings.loudness_true_peak, settings.ffmpeg_path,
            )  # fmt: skip
        audio_filter = raudio.loudnorm_filter(
            settings.loudness_target, settings.loudness_true_peak, gain, plan.duration,
            settings.audio_fade_seconds,
        )  # fmt: skip
        zone = safe_zone(filters.OUT_W, filters.OUT_H, settings.safe_bottom, settings.safe_top,
                         settings.caption_lift)  # fmt: skip
        style = get_style(plan.style_key, settings.caption_styles_path)
        ass = build_ass(
            plan.words, plan.duration, style, zone, plan.overlays, settings.min_caption_event
        )
        prepare_workdir(work, settings, ass)
        log(f"#{c.rank}: {kind_label} de {plan.duration:.0f}s ({LAYOUT_LABELS[plan.layout]}, "
            f"legenda '{style.key}', {encoder})…")  # fmt: skip
        used = run_encode(
            lambda enc: build_args(plan, settings, enc, audio_filter, tmp_out),
            settings, encoder, plan, work, progress_cb, log,
        )  # fmt: skip
        w, h = out_size(settings, plan.kind)
        probe, problems = validate_render(
            tmp_out, w, h, plan.duration, ffprobe=settings.ffprobe_path
        )
        if problems:
            raise ClipadorError("O arquivo renderizado não passou na verificação: "
                                + "; ".join(problems) + ".")  # fmt: skip
        out_dir = settings.clips_dir / str(plan.source_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        final = out_dir / f"{plan.clip_name}_{plan.kind}_{plan.phash[:8]}.mp4"
        tmp_out.replace(final)
        loud = (
            raudio.integrated_loudness(final, settings.ffmpeg_path)
            if plan.kind == RenderKind.FINAL
            else None
        )
        if plan.kind == RenderKind.FINAL:
            thumb = final.with_suffix(".jpg")
            with contextlib.suppress(ToolError):
                ffmpeg.make_thumbnail(final, thumb, plan.zoom_peak or plan.duration / 3,
                                      settings.ffmpeg_path)  # fmt: skip
            row.thumbnail_path = settings.rel_path(thumb) if thumb.is_file() else None
            drafts, alerts = _drafts_for_package(session, settings, c)
            pkg = final.with_suffix(".txt")
            pkg.write_text(package_text(drafts, final.name, alerts), "utf-8")
            row.package_path = settings.rel_path(pkg)
        elapsed = time.monotonic() - t0
        row.status, row.path = RenderStatus.DONE, settings.rel_path(final)
        row.duration, row.size_bytes, row.encoder = probe.duration, final.stat().st_size, used
        row.render_seconds, row.loudness_lufs = round(elapsed, 2), loud
        row.warning = " ".join(plan.warnings) or None
        row.updated_at = utcnow()
        session.add(row)
        if plan.kind == RenderKind.FINAL and c.status != CandidateStatus.REJECTED:
            c.status = CandidateStatus.RENDERED
            session.add(c)
        _drop_older(session, settings, row)
        session.commit()
        speed = plan.duration / max(elapsed, 0.01)
        log(f"#{c.rank}: {kind_label} pronta em {elapsed:.1f}s ({speed:.1f}x tempo real, "
            f"{row.size_bytes / 1e6:.1f} MB" + (f", {loud:.1f} LUFS" if loud is not None else "")
            + ").")  # fmt: skip
        for wmsg in plan.warnings:
            log(f"AVISO: {wmsg}")
        return row
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _drop_older(session: Session, settings: Settings, keep: Render) -> None:
    """Older finished renders of the same candidate+kind are replaced (files removed)."""
    old = session.exec(
        select(Render).where(
            Render.candidate_id == keep.candidate_id,
            Render.kind == keep.kind,
            Render.id != keep.id,
            col(Render.status).in_([RenderStatus.DONE, RenderStatus.FAILED, RenderStatus.CANCELED]),
        )
    ).all()
    for r in old:
        for rel in (r.path, r.package_path, r.thumbnail_path):
            if rel and rel != keep.path:
                with contextlib.suppress(OSError):
                    settings.abs_path(rel).unlink()
        session.delete(r)


def mark_unfinished(session: Session, job_id: int, status: str, message: str) -> None:
    for r in session.exec(
        select(Render).where(
            Render.job_id == job_id,
            col(Render.status).in_([RenderStatus.QUEUED, RenderStatus.RUNNING]),
        )
    ).all():
        r.status, r.error, r.updated_at = status, message, utcnow()
        session.add(r)
    session.commit()


def latest_renders(session: Session, candidate_ids: list[int]) -> dict[int, dict[str, Render]]:
    out: dict[int, dict[str, Render]] = {}
    if not candidate_ids:
        return out
    rows = session.exec(
        select(Render).where(col(Render.candidate_id).in_(candidate_ids)).order_by(Render.id)
    ).all()
    for r in rows:
        out.setdefault(r.candidate_id, {})[r.kind] = r  # later rows win
    return out


def job_progress(session: Session, render: Render) -> float | None:
    if render.job_id is None:
        return None
    job = session.get(Job, render.job_id)
    return job.progress if job else None


# --- 16:9 compilation of the approved clips ----------------------------------------------------


def _mmss(t: float) -> str:
    t = int(t)
    if t >= 3600:
        return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"
    return f"{t // 60:02d}:{t % 60:02d}"


def compile_long(
    settings: Settings, plans: list[RenderPlan], progress_cb, log
) -> tuple[Path, Path, str]:
    """Approved moments (chronological) -> 1920x1080 video with captions + chapters file."""
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
    zone = safe_zone(1920, 1080, 0.06, 0.05, 0.0, font_scale=0.6)
    total = sum(p.duration for p in plans)
    done = 0.0
    seg_files, chapter_lines, t_cursor = [], [], 0.0
    try:
        for i, p in enumerate(plans):
            seg_dir = work / f"seg{i:02d}"
            style = get_style(p.style_key, settings.caption_styles_path)
            ov = Overlays(None, 0, p.overlays.credit)
            prepare_workdir(seg_dir, settings, build_ass(p.words, p.duration, style, zone, ov,
                                                         settings.min_caption_event))  # fmt: skip
            seg = work / f"seg{i:02d}.mp4"
            af = raudio.loudnorm_filter(settings.loudness_target, settings.loudness_true_peak,
                                        None, p.duration, 0.3)  # fmt: skip
            graph = filters.horizontal_graph(p.src_w, p.src_h, p.duration, af)
            args = ["-ss", f"{p.start:.3f}", "-t", f"{p.duration:.3f}", "-i", str(p.video),
                    "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
                    *filters.output_rate_args(min(p.fps, 30), 30),
                    *filters.encoder_args(encoder), str(seg)]  # fmt: skip
            log(f"Vídeo longo: trecho {i + 1}/{len(plans)} ({format_hms(p.start)}).")
            ffmpeg.run_with_progress(
                args, duration=p.duration, cwd=seg_dir, ffmpeg=settings.ffmpeg_path,
                progress_cb=lambda f, b=done, d=p.duration: progress_cb((b + f * d) / total * 0.95),
                error_message="Falha ao montar um trecho do vídeo longo.",
            )  # fmt: skip
            done += p.duration
            seg_files.append(seg)
            chapter_lines.append((t_cursor, p.overlays.opening_text or f"Momento {i + 1}"))
            t_cursor += p.duration
        lst = work / "list.txt"
        lst.write_text("".join(f"file '{s.name}'\n" for s in seg_files), "utf-8")
        tmp_final = work / "longo.tmp.mp4"
        ffmpeg.run_with_progress(
            ["-f", "concat", "-safe", "0", "-i", lst.name, "-c", "copy", "-movflags", "+faststart",
             str(tmp_final)],
            duration=total, cwd=work, ffmpeg=settings.ffmpeg_path, progress_cb=lambda f: None,
            error_message="Falha ao juntar os trechos do vídeo longo.",
        )  # fmt: skip
        tmp_final.replace(final)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    chapters.write_text(
        "Capítulos (cole na descrição do YouTube):\n"
        + "\n".join(f"{_mmss(t)} {title}" for t, title in chapter_lines)
        + "\n",
        "utf-8",
    )
    progress_cb(1.0)
    return final, chapters, phash
