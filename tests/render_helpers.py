"""Shared synthetic media + DB setup for render tests (ffmpeg only, no network)."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import date
from pathlib import Path

import numpy as np
from sqlmodel import Session, select

from clipador.detection.params import DetectParams
from clipador.models import (
    AnalysisRun,
    Candidate,
    CandidateStatus,
    Job,
    Source,
    SourceStatus,
    Streamer,
)
from clipador.services.artifacts import save_artifact
from clipador.services.ffmpeg import probe
from clipador.services.transcriber import Word, save_transcript

FLASH_T = 2.0  # white flash + 1 kHz beep in the source (A/V sync reference)
WORDS = [(1.0, 1.3, "olha"), (1.35, 1.6, "a"), (1.65, 2.2, "polícia!"), (3.5, 3.8, "corre"),
         (3.85, 4.2, "{corre}"), (4.25, 4.9, "\\Ncorre 😂"), (6.0, 6.4, "meu"), (6.45, 7.0, "deus"),
         (8.0, 8.5, "capotou"), (8.6, 9.4, "o"), (9.45, 10.5, "carro")]  # fmt: skip


def make_video(path: Path, size: str = "1280x720", seconds: float = 12, vfr: bool = False) -> Path:
    """testsrc2 with a white flash at FLASH_T; audio: quiet tone, a 1 kHz beep at FLASH_T and a
    silent gap (tones + silences, so loudness normalization has real work to do)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size.split("x")
    vf = (f"drawbox=x=0:y=0:w={w}:h={h}:color=white:t=fill:enable='between(t,{FLASH_T},"
          f"{FLASH_T + 0.1})'")  # fmt: skip
    aexpr = (f"0.02*sin(2*PI*220*t)*not(between(t,5,6))"
             f"+0.5*sin(2*PI*1000*t)*between(t,{FLASH_T},{FLASH_T + 0.1})")  # fmt: skip
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30",
           "-f", "lavfi", "-i", f"aevalsrc='{aexpr}':s=48000:c=mono",
           "-t", str(seconds), "-vf", vf + (",select='not(eq(mod(n\\,4)\\,3))'" if vfr else "")]  # fmt: skip
    if vfr:
        cmd += ["-fps_mode", "vfr"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", str(path)]  # fmt: skip
    subprocess.run(cmd, check=True)
    return path


def seed(settings, engine, video: Path, *, rect=None, layout="blur",
         candidates=((0.5, 5.5, 2.0), (6.0, 11.0, 8.2))) -> tuple[int, list[int], int]:  # fmt: skip
    """Streamer + ready source (+transcript) + approved candidates. Returns (src, cands, streamer)."""
    info = probe(video)
    tr = settings.work_dir / "1" / "transcript.json.gz"
    save_transcript(tr, [Word(s, e, w) for s, e, w in WORDS], {"language": "pt"})
    with Session(engine, expire_on_commit=False) as s:
        st = Streamer(display_name="Fulano", permission_confirmed=True, permission_date=date.today(),
                      credit_text="Clip de @fulano na Twitch", facecam_box=list(rect) if rect else None,
                      default_layout=layout)  # fmt: skip
        s.add(st)
        s.flush()
        src = Source(streamer_id=st.id, platform="twitch", kind="vod", external_id="1",
                     url="https://www.twitch.tv/videos/1", status=SourceStatus.READY,
                     file_path=settings.rel_path(video), duration_seconds=info.duration,
                     media_info=info.to_dict())  # fmt: skip
        s.add(src)
        s.flush()
        run = AnalysisRun(source_id=src.id, params_hash="x", params=DetectParams().model_dump())
        s.add(run)
        s.flush()
        ids = []
        for i, (a, b, peak) in enumerate(candidates, 1):
            c = Candidate(source_id=src.id, analysis_run_id=run.id, rank=i, start=a, end=b,
                          peak_t=peak, score=80 - i, status=CandidateStatus.APPROVED)  # fmt: skip
            s.add(c)
            s.flush()
            ids.append(c.id)
        s.commit()
        sid, stid = src.id, st.id
    save_artifact(engine, sid, "transcribe", "h", files={"transcript": settings.rel_path(tr)})
    return sid, ids, stid


def drain(worker, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        worker.tick()
        time.sleep(0.05)
        if worker.active_job_ids():
            continue
        with Session(worker.engine) as s:
            if not s.exec(select(Job).where(Job.status.in_(["queued", "running"]))).first():
                return
    raise AssertionError("timeout")


def ffprobe_json(path: Path, entries: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", entries, "-of", "json",
                          str(path)], capture_output=True, text=True, check=True).stdout  # fmt: skip
    return json.loads(out)


def frame_at(path: Path, t: float, w: int, h: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path),
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout  # fmt: skip
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w)


def brightest_frame_time(path: Path, fps: float = 30.0) -> float:
    """Timestamp (s) of the brightest video frame (the flash), via tiny gray frames on a pipe."""
    w, h = 16, 28
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
                          f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout  # fmt: skip
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h * w).mean(axis=1)
    return float(np.argmax(frames) / fps)


def loudest_audio_time(path: Path) -> float:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar",
                          "8000", "-f", "s16le", "-"], capture_output=True, check=True).stdout  # fmt: skip
    a = np.abs(np.frombuffer(raw, dtype="<i2").astype(np.float32))
    win = 80  # 10 ms
    env = np.convolve(a, np.ones(win) / win, mode="same")
    return float(np.argmax(env) / 8000)
