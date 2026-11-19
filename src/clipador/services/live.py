"""Live sessions (step 5E): create/start/stop, the separate capture process, heartbeat/dead
process detection, stats for the UI, the segment window used by renders and the retention.

The capture itself runs in `clipador live-capture <id>` (services/live_capture.py): it never
takes a worker slot. Analysis of each closed segment is a normal worker job (`live_analyze`,
pipeline/stages/live.py).
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, func
from sqlmodel import Session, col, select

from clipador.config import Settings
from clipador.detection.params import DetectParams
from clipador.errors import ClipadorError, ValidationError
from clipador.models import (
    ACTIVE_LIVE_STATUSES,
    AnalysisRun,
    Candidate,
    CandidateStatus,
    LiveSegment,
    LiveSession,
    LiveStatus,
    SegmentStatus,
    Source,
    SourceKind,
    SourceStatus,
    Streamer,
)
from clipador.platforms.base import ChatMessage
from clipador.services.sources import PERMISSION_REQUIRED_MESSAGE, channel_warning
from clipador.utils import utcnow

log = logging.getLogger(__name__)

ANALYZE_PRIORITY = 5  # live analysis goes ahead of VOD work and of the automatic previews
SEGMENT_PATTERN = "seg_%05d.ts"


def aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def session_dir(settings: Settings, live_id: int) -> Path:
    return settings.data_dir / "live" / str(live_id)


# --- per-segment files (data/live/<id>/) -----------------------------------------------------


def segment_wav(settings: Settings, live_id: int, index: int) -> Path:
    return session_dir(settings, live_id) / f"seg_{index:05d}.wav"


def segment_words_path(settings: Settings, live_id: int, index: int) -> Path:
    return session_dir(settings, live_id) / f"seg_{index:05d}.words.json"


def save_segment_words(path: Path, words: list[tuple[float, float, str, float]]) -> None:
    """Words of one segment, times on the session's global timeline. Atomic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([[round(a, 3), round(b, 3), w, round(p, 3)] for a, b, w, p in words],
                              ensure_ascii=False), "utf-8")  # fmt: skip
    tmp.replace(path)


def load_words(
    settings: Settings, live_id: int, indexes: list[int] | None = None
) -> list[tuple[float, float, str, float]]:
    """Words of the given segments (all transcribed ones when None), sorted by time."""
    d = session_dir(settings, live_id)
    if indexes is None:
        files = sorted(d.glob("seg_*.words.json"))
    else:
        files = [segment_words_path(settings, live_id, i) for i in indexes]
    out: list[tuple[float, float, str, float]] = []
    for f in files:
        with contextlib.suppress(OSError, ValueError):
            out.extend((float(a), float(b), str(w), float(p))
                       for a, b, w, p in json.loads(f.read_text("utf-8")))  # fmt: skip
    return sorted(out)


def session_language(settings: Settings, live_id: int) -> str | None:
    """Language for Whisper: the configured one, or the one detected on the first segment
    (kept for the whole live so segments don't flip between languages)."""
    if settings.language != "auto":
        return settings.language
    f = session_dir(settings, live_id) / "language.txt"
    return f.read_text("utf-8").strip() or None if f.is_file() else None


def save_session_language(settings: Settings, live_id: int, lang: str) -> None:
    f = session_dir(settings, live_id) / "language.txt"
    if settings.language == "auto" and not f.is_file():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(lang, "utf-8")


def session_chat(settings: Settings, live_id: int) -> list[ChatMessage]:
    """Chat read live by the capture process (chat.jsonl, global times)."""
    f = session_dir(settings, live_id) / "chat.jsonl"
    if not f.is_file():
        return []
    out = []
    with f.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            with contextlib.suppress(ValueError, KeyError, TypeError):
                r = json.loads(line)
                out.append(ChatMessage(float(r["t"]), str(r.get("author", "")),
                                       str(r.get("text", ""))))  # fmt: skip
    return out


def source_words(session: Session, settings: Settings,
                 source_id: int) -> list[tuple[float, float, str]] | None:  # fmt: skip
    """All transcribed words of a live source (captions, AI), or None if it is not a live."""
    ls = live_session_of_source(session, source_id)
    if ls is None:
        return None
    return [(a, b, w) for a, b, w, _p in load_words(settings, ls.id)]


# --- creating / starting ----------------------------------------------------------------------


def _check_streamer(session: Session, streamer_id: int | None) -> Streamer:
    st = session.get(Streamer, streamer_id) if streamer_id else None
    if st is None:
        raise ValidationError("Escolha o streamer dono da live.")
    if not st.active:
        raise ValidationError(f"{st.display_name} está desativado. Reative-o antes.")
    if not st.permission_confirmed:
        raise ValidationError(PERMISSION_REQUIRED_MESSAGE.format(name=st.display_name))
    return st


def active_session_for_streamer(session: Session, streamer_id: int) -> LiveSession | None:
    return session.exec(
        select(LiveSession)
        .join(Source, Source.id == LiveSession.source_id)
        .where(Source.streamer_id == streamer_id, col(LiveSession.status).in_(ACTIVE_LIVE_STATUSES))
    ).first()


def streamer_for_channel(session: Session, platform: str, handle: str | None) -> Streamer | None:
    from clipador.models import StreamerChannel

    if not handle:
        return None
    ch = session.exec(
        select(StreamerChannel).where(
            StreamerChannel.platform == platform,
            func.lower(StreamerChannel.channel_handle) == handle.lower(),
        )
    ).first()
    return session.get(Streamer, ch.streamer_id) if ch else None


@dataclass
class LiveOptions:
    segment_seconds: int
    auto_clip: bool = True
    min_score: float = 70.0
    live_from_start: bool = False

    @classmethod
    def defaults(cls, settings: Settings) -> LiveOptions:
        return cls(settings.live_segment_seconds, True, settings.live_min_score, False)

    def validated(self) -> LiveOptions:
        if not 10 <= int(self.segment_seconds) <= 1800:
            raise ValidationError("Tamanho do segmento: entre 10 s e 30 min.")
        if not 0 <= float(self.min_score) <= 100:
            raise ValidationError("Nota mínima: entre 0 e 100.")
        return LiveOptions(int(self.segment_seconds), bool(self.auto_clip),
                           float(self.min_score), bool(self.live_from_start))  # fmt: skip


def create_session(
    session: Session,
    settings: Settings,
    *,
    streamer_id: int | None,
    url: str | None = None,
    sim_file: Path | None = None,
    sim_speed: float | None = None,
    options: LiveOptions | None = None,
) -> LiveSession:
    """Validate (authorized streamer, one live per streamer) and create Source + LiveSession +
    the live AnalysisRun. Does not start the capture process. Does not commit."""
    from clipador.platforms.live import parse_live_url

    opts = (options or LiveOptions.defaults(settings)).validated()
    now = utcnow()
    stamp = now.strftime("%Y%m%d%H%M%S")
    warning = None
    if sim_file is not None:
        if not sim_file.is_file():
            raise ValidationError(f"Arquivo não encontrado: {sim_file}")
        st = _check_streamer(session, streamer_id)
        platform = st.channels[0].platform if st.channels else "twitch"
        ext_id, src_url = f"sim-{stamp}", sim_file.resolve().as_uri()
        title = f"Simulação de live: {sim_file.name}"
        opts.live_from_start = False
    else:
        target = parse_live_url(url or "")
        if streamer_id is None:
            match = streamer_for_channel(session, target.platform, target.channel_handle)
            streamer_id = match.id if match else None
        st = _check_streamer(session, streamer_id)
        if target.channel_handle:
            warning = channel_warning(st, target.platform, [target.channel_handle])
        platform, ext_id, src_url = target.platform, f"{target.external_id}@{stamp}", target.url
        title = f"Live de {st.display_name} ({now.astimezone().strftime('%d/%m %H:%M')})"
        if opts.live_from_start and not target.supports_from_start:
            opts.live_from_start = False
            note = "gravando a partir de agora (essa plataforma não permite pegar desde o início)"
            warning = f"{warning} {note}".strip() if warning else note.capitalize()
    if active_session_for_streamer(session, st.id) is not None:
        raise ValidationError(f"Já existe uma live de {st.display_name} sendo gravada.")
    src = Source(
        streamer_id=st.id, platform=platform, kind=SourceKind.LIVE, external_id=ext_id,
        url=src_url, title=title, channel_name=st.display_name, status=SourceStatus.LIVE,
        is_live=True, warning=warning, published_at=now,
    )  # fmt: skip
    session.add(src)
    session.flush()
    params = DetectParams.from_preset(settings.detect_preset).model_dump()
    params["live"] = True
    run = AnalysisRun(source_id=src.id, params_hash=f"live-{src.id}", params=params)
    session.add(run)
    session.flush()
    ls = LiveSession(
        source_id=src.id, segment_seconds=opts.segment_seconds, auto_clip=opts.auto_clip,
        min_score=opts.min_score, live_from_start=opts.live_from_start, warning=warning,
        sim_file=str(sim_file.resolve()) if sim_file else None,
        sim_speed=sim_speed if sim_file else None, run_id=run.id,
    )  # fmt: skip
    session.add(ls)
    session.flush()
    return ls


def capture_command(live_id: int) -> list[str]:
    return [sys.executable, "-m", "clipador.cli", "live-capture", str(live_id)]


def spawn_capture(settings: Settings, live_id: int) -> int:
    """Start `clipador live-capture <id>` as a separate, detached process (output in
    data/logs/live_<id>.log). Returns its pid."""
    settings.ensure_dirs()
    out = open(settings.logs_dir / f"live_{live_id}.log", "a", encoding="utf-8")  # noqa: SIM115
    kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": out,
                              "stderr": subprocess.STDOUT, "cwd": str(Path.cwd())}  # fmt: skip
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        breakaway = 0x01000000  # CREATE_BREAKAWAY_FROM_JOB: survive the worker's job object
        try:
            proc = subprocess.Popen(capture_command(live_id), creationflags=flags | breakaway,
                                    **kwargs)  # fmt: skip
        except OSError:
            proc = subprocess.Popen(capture_command(live_id), creationflags=flags, **kwargs)
    else:
        proc = subprocess.Popen(capture_command(live_id), start_new_session=True, **kwargs)
    out.close()
    return proc.pid


def start_session(engine: Engine, settings: Settings, ls_id: int, spawn=None) -> LiveSession:
    """Spawn the capture process of a created session (web / auto_watch)."""
    spawn = spawn or spawn_capture
    with Session(engine, expire_on_commit=False) as s:
        ls = s.get(LiveSession, ls_id)
        ls.heartbeat_at = ls.updated_at = utcnow()  # grace period while the process starts
        s.add(ls)
        s.commit()
        ls.pid = spawn(settings, ls.id)
        s.add(ls)
        s.commit()
        return ls


def auto_watch(engine: Engine, settings: Settings, check=None, spawn=None) -> list[str]:
    """Streamers with auto_watch on (and confirmed permission): if a registered channel is live
    and there is no session for that streamer yet, create one and start capturing. Returns
    log lines. `check(url) -> bool` is injectable for tests (default: yt-dlp --simulate)."""
    from clipador.models import StreamerChannel
    from clipador.platforms.live import channel_live_url

    if check is None:
        from clipador.platforms.live import check_live
        from clipador.services.downloader import DownloadError, YtDlp

        yt = YtDlp(settings)

        def check(url: str) -> bool:
            try:
                return check_live(yt, url).live
            except DownloadError:
                return False

    notes = []
    with Session(engine, expire_on_commit=False) as s:
        streamers = s.exec(
            select(Streamer).where(
                Streamer.auto_watch == True,  # noqa: E712
                Streamer.active == True,  # noqa: E712
                Streamer.permission_confirmed == True,  # noqa: E712
            )
        ).all()
        todo = []
        for st in streamers:
            if active_session_for_streamer(s, st.id) is not None:
                continue
            chans = s.exec(
                select(StreamerChannel).where(StreamerChannel.streamer_id == st.id)
            ).all()
            todo.append((st.id, st.display_name, [(c.platform, c.channel_handle) for c in chans]))
    for st_id, name, chans in todo:
        for platform, handle in chans:
            url = channel_live_url(platform, handle)
            if not url or not check(url):
                continue
            try:
                with Session(engine, expire_on_commit=False) as s:
                    ls = create_session(s, settings, streamer_id=st_id, url=url)
                    s.commit()
                start_session(engine, settings, ls.id, spawn)
                notes.append(
                    f"{name} entrou ao vivo ({platform}): captura iniciada (live #{ls.id})."
                )
            except (ValidationError, OSError) as e:
                notes.append(f"{name} está ao vivo, mas não consegui iniciar: {e}")
            break  # one session per streamer
    return notes


def request_stop(session: Session, live_id: int) -> LiveSession:
    ls = session.get(LiveSession, live_id)
    if ls is None:
        raise ValidationError("Sessão de live não encontrada.")
    if not ls.active:
        raise ValidationError("Esta live já foi encerrada.")
    ls.stop_requested = True
    ls.updated_at = utcnow()
    session.add(ls)
    return ls


# --- health -------------------------------------------------------------------------------------


def is_dead(ls: LiveSession, settings: Settings, now: datetime | None = None) -> bool:
    """An active session whose capture process stopped sending heartbeats."""
    if not ls.active:
        return False
    now = now or utcnow()
    last = aware(ls.heartbeat_at) or aware(ls.started_at)
    return (now - last).total_seconds() > settings.live_stale_seconds


def finish_session(
    session: Session, ls: LiveSession, status: str = LiveStatus.STOPPED, error: str | None = None
) -> None:
    """Final state + queue the VOD assembly (worker job). Does not commit."""
    from clipador.worker import queue

    ls.status = status
    ls.stopped_at = ls.updated_at = utcnow()
    if error:
        ls.error = error
    session.add(ls)
    queue.enqueue(session, "live_finalize", ls.source_id, {"session_id": ls.id},
                  priority=ANALYZE_PRIORITY)  # fmt: skip


def supervise(engine: Engine, settings: Settings, spawn=spawn_capture) -> list[str]:
    """Worker side, every few seconds: restart dead capture processes (up to
    live_max_restarts) or close their session. Returns log lines."""
    notes = []
    with Session(engine, expire_on_commit=False) as s:
        rows = s.exec(
            select(LiveSession).where(col(LiveSession.status).in_(ACTIVE_LIVE_STATUSES))
        ).all()
        for ls in rows:
            if not is_dead(ls, settings):
                continue
            if ls.stop_requested:
                finish_session(s, ls)
                notes.append(f"Live #{ls.id}: processo parado; sessão encerrada.")
            elif ls.restarts >= settings.live_max_restarts:
                msg = f"O processo de captura caiu {ls.restarts + 1} vezes; desisti."
                finish_session(s, ls, LiveStatus.FAILED, msg)
                notes.append(f"Live #{ls.id}: captura caiu demais; sessão encerrada.")
            else:
                ls.restarts += 1
                ls.heartbeat_at = ls.updated_at = utcnow()  # grace period while it starts
                s.add(ls)
                s.commit()
                try:
                    ls.pid = spawn(settings, ls.id)
                except OSError as e:
                    ls.error = f"Não consegui reiniciar a captura: {e}"
                notes.append(f"Live #{ls.id}: processo de captura sem sinal; reiniciado.")
                s.add(ls)
            s.commit()
    return notes


# --- stats for the UI -------------------------------------------------------------------------


def segments(session: Session, live_id: int) -> list[LiveSegment]:
    return list(
        session.exec(
            select(LiveSegment).where(LiveSegment.session_id == live_id).order_by(LiveSegment.index)
        ).all()
    )


def stats(session: Session, settings: Settings, ls: LiveSession) -> dict[str, Any]:
    segs = segments(session, ls.id)
    done = [g for g in segs if g.status != SegmentStatus.RECORDING]
    captured = sum(g.duration_s for g in done)
    cands = session.exec(select(Candidate).where(Candidate.live_session_id == ls.id)).all()
    hb = aware(ls.heartbeat_at)
    return {
        "captured_s": captured,
        "segments": len(done),
        "analyzed": sum(g.status == SegmentStatus.ANALYZED for g in segs),
        "pending": sum(g.status == SegmentStatus.CLOSED for g in segs),
        "failed": sum(g.status == SegmentStatus.FAILED for g in segs),
        "expired": sum(g.status == SegmentStatus.EXPIRED for g in segs),
        "recording": next((g for g in segs if g.status == SegmentStatus.RECORDING), None),
        "candidates": len(cands),
        "in_review": _in_review(cands, ls),
        "ai_scored": _ai_scored(session, [c.id for c in cands]),
        "auto_clipped": sum(c.auto_clipped for c in cands),
        "heartbeat_age": (utcnow() - hb).total_seconds() if hb else None,
        "dead": is_dead(ls, settings),
    }


def _in_review(cands: list[Candidate], ls: LiveSession) -> int:
    """How many of the session's clips are waiting in the review (all pending ones)."""
    from clipador.models import PENDING_STATUSES

    return sum(1 for c in cands if c.status in PENDING_STATUSES)


def _ai_scored(session: Session, ids: list[int]) -> int:
    from clipador.models import CandidateEnrichment, EnrichStatus

    if not ids:
        return 0
    return len(session.exec(select(CandidateEnrichment.id).where(
        col(CandidateEnrichment.candidate_id).in_(ids),
        CandidateEnrichment.status == EnrichStatus.OK)).all())  # fmt: skip


def pending_auto_clips(session: Session) -> list[Candidate]:
    """Auto-clipped candidates still waiting for a decision (badge in the menu, notifications)."""
    return list(
        session.exec(
            select(Candidate)
            .where(
                Candidate.auto_clipped == True,  # noqa: E712
                col(Candidate.status).in_([CandidateStatus.SUGGESTED, CandidateStatus.SKIPPED]),
            )
            .order_by(col(Candidate.id).desc())
        ).all()
    )


# --- segment window for renders ---------------------------------------------------------------


def live_session_of_source(session: Session, source_id: int) -> LiveSession | None:
    return session.exec(
        select(LiveSession)
        .where(LiveSession.source_id == source_id)
        .order_by(col(LiveSession.id).desc())
    ).first()


@dataclass
class SegmentWindow:
    parts: list[Path]  # segment files, in order
    offset: float  # global time of the first part's start
    key: str  # stable id of the window (render params hash)
    video: Path  # the file to read: the single part, or the cached concat of all parts


def segment_window(
    session: Session, settings: Settings, source_id: int, start: float, end: float
) -> SegmentWindow:
    """Segments covering [start, end] of a live source (no single video file yet)."""
    ls = live_session_of_source(session, source_id)
    if ls is None:
        raise ClipadorError("O vídeo desta fonte não está baixado.")
    segs = [
        g for g in segments(session, ls.id)
        if g.status in (SegmentStatus.CLOSED, SegmentStatus.ANALYZED, SegmentStatus.FAILED)
        and g.path and g.start_offset_s < end and g.end_offset_s > start
    ]  # fmt: skip
    if not segs or segs[0].start_offset_s > start + 0.5 or segs[-1].end_offset_s < end - 0.5:
        raise ClipadorError("Os segmentos da live deste trecho não estão mais no disco.")
    if any(b.index != a.index + 1 for a, b in itertools.pairwise(segs)):
        raise ClipadorError("Falta um segmento da live no meio deste trecho.")
    parts = [settings.abs_path(g.path) for g in segs]
    if not all(p.is_file() for p in parts):
        raise ClipadorError("Os segmentos da live deste trecho não estão mais no disco.")
    key = f"live:{ls.id}:{segs[0].index}-{segs[-1].index}"
    if len(parts) == 1:
        video = parts[0]
    else:
        name = f"cat_{segs[0].index:05d}_{segs[-1].index:05d}.mp4"
        video = settings.tmp_dir / f"live_{ls.id}" / name
    return SegmentWindow(parts, segs[0].start_offset_s, key, video)


def concat_files(parts: list[Path], out: Path, ffmpeg: str = "ffmpeg") -> Path:
    """Join segment files without re-encoding (concat demuxer, -c copy). Atomic."""
    from clipador.errors import ToolError

    out.parent.mkdir(parents=True, exist_ok=True)
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts), "utf-8")
    tmp = out.with_name(out.stem + ".tmp" + out.suffix)
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-f", "concat",
           "-safe", "0", "-i", str(lst), "-map", "0:v:0?", "-map", "0:a:0?", "-c", "copy",
           "-movflags", "+faststart", str(tmp)]  # fmt: skip
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=3600)  # fmt: skip
    except FileNotFoundError as e:
        raise ToolError("ffmpeg não encontrado (rode `clipador doctor`).", detail=str(e)) from e
    finally:
        with contextlib.suppress(OSError):
            lst.unlink()
    if res.returncode != 0 or not tmp.is_file():
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise ToolError("Falha ao juntar os segmentos da live.", detail=res.stderr[-3000:])
    tmp.replace(out)
    return out


def ensure_window_video(win: SegmentWindow, ffmpeg: str = "ffmpeg") -> Path:
    """The concat is built once and cached (tmp/live_<id>/cat_<a>_<b>.mp4)."""
    if not win.video.is_file():
        concat_files(win.parts, win.video, ffmpeg)
    return win.video


# --- retention ------------------------------------------------------------------------------------


def retention_sweep(session: Session, settings: Settings, ls: LiveSession,
                    now: datetime | None = None) -> list[int]:  # fmt: skip
    """Delete segment files that are no longer needed (status -> expired). Returns indexes.
    - the live's VOD was assembled: every analyzed segment is inside it;
    - otherwise: analyzed segments older than live_retention_hours (0 = never) that don't touch
      any candidate of the session (with the analysis context as margin)."""
    now = now or utcnow()
    src = session.get(Source, ls.source_id)
    has_vod = bool(src and src.file_path and settings.abs_path(src.file_path).is_file())
    hours = settings.live_retention_hours
    if not has_vod and hours <= 0:
        return []
    cands = session.exec(select(Candidate).where(Candidate.live_session_id == ls.id)).all()
    margin = settings.live_context_seconds
    spans = [(c.eff_start - margin, c.eff_end + margin) for c in cands]
    expired = []
    for g in segments(session, ls.id):
        if g.status not in (SegmentStatus.ANALYZED, SegmentStatus.FAILED) or not g.path:
            continue
        if not has_vod:
            age = (now - aware(g.analyzed_at or g.updated_at)).total_seconds()
            if age < hours * 3600:
                continue
            if any(a < g.end_offset_s and b > g.start_offset_s for a, b in spans):
                continue
        with contextlib.suppress(OSError):
            settings.abs_path(g.path).unlink()
        wav = session_dir(settings, ls.id) / f"seg_{g.index:05d}.wav"
        with contextlib.suppress(OSError):
            wav.unlink()
        g.status, g.path, g.updated_at = SegmentStatus.EXPIRED, None, now
        session.add(g)
        expired.append(g.index)
    if expired and has_vod:
        import shutil

        shutil.rmtree(settings.tmp_dir / f"live_{ls.id}", ignore_errors=True)
    return expired


def wall_seconds(ls: LiveSession) -> float:
    end = aware(ls.stopped_at) or utcnow()
    return max(0.0, (end - aware(ls.started_at)).total_seconds())


def disk_free_gb(path: Path) -> float:
    import shutil

    return shutil.disk_usage(path).free / 1e9


def mark_heartbeat(engine: Engine, live_id: int, status: str | None = None) -> LiveSession | None:
    with Session(engine, expire_on_commit=False) as s:
        ls = s.get(LiveSession, live_id)
        if ls is None:
            return None
        ls.heartbeat_at = ls.updated_at = utcnow()
        ls.pid = os.getpid()
        if status and ls.active:
            ls.status = status
        s.add(ls)
        s.commit()
        return ls
