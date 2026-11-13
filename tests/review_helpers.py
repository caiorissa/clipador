"""DB seed for review tests: no ffmpeg, no network. The 'video' is a dummy file (plan_render only
checks that it exists), so renders can be planned/queued but are never executed here."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from datetime import date

import numpy as np
from sqlmodel import Session

from clipador.detection.params import DetectParams
from clipador.models import (
    AnalysisRun,
    Candidate,
    CandidateEnrichment,
    Source,
    SourceStatus,
    Streamer,
)
from clipador.services.artifacts import save_artifact
from clipador.services.transcriber import Word, save_transcript

# absolute times in the original video
WORDS = [(10.0, 10.4, "olha"), (10.5, 10.8, "a"), (10.9, 11.6, "polícia!"), (13.0, 13.5, "corre"),
         (13.6, 14.2, "corre"), (20.0, 20.5, "meu"), (20.6, 21.2, "deus")]  # fmt: skip


@dataclass
class Seeded:
    sources: list[int]
    streamers: list[int]
    cands: dict[int, list[int]]  # source id -> candidate ids (rank order)
    runs: dict[int, int]


def seed_review(settings, engine, *, video=True, transcript=True) -> Seeded:
    """Source 1 (Twitch, streamer A, has a facecam): 3 candidates; source 2 (YouTube,
    streamer B): 2 candidates. final_score decreases with rank inside each source."""
    out = Seeded([], [], {}, {})
    specs = [("twitch", "A", [(8, 22, 90, 85), (30, 50, 70, 80), (60, 75, 50, 40)]),
             ("youtube", "B", [(5, 20, 88, 88), (40, 55, 60, 30)])]  # fmt: skip
    with Session(engine, expire_on_commit=False) as s:
        for n, (platform, name, cands) in enumerate(specs, 1):
            st = Streamer(display_name=name, permission_confirmed=True, permission_date=date.today(),
                          credit_text=f"Clip de @{name.lower()}",
                          facecam_box=[0.7, 0.6, 0.25, 0.35] if n == 1 else None)  # fmt: skip
            s.add(st)
            s.flush()
            path = settings.vods_dir / str(n) / "video.mp4"
            if video:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"not really a video")
            src = Source(streamer_id=st.id, platform=platform, kind="vod", external_id=str(n),
                         url=f"https://example.invalid/{n}", status=SourceStatus.READY,
                         file_path=settings.rel_path(path) if video else None,
                         duration_seconds=600.0,
                         media_info={"width": 1920, "height": 1080, "fps": 30})  # fmt: skip
            s.add(src)
            s.flush()
            run = AnalysisRun(source_id=src.id, params_hash="x", params=DetectParams().model_dump())
            s.add(run)
            s.flush()
            ids = []
            for rank, (a, b, score, final) in enumerate(cands, 1):
                c = Candidate(source_id=src.id, analysis_run_id=run.id, rank=rank, start=a, end=b,
                              peak_t=(a + b) / 2, score=score, final_score=final,
                              breakdown={"audio": 50, "speech": 70},
                              transcript_snippet="olha a polícia")  # fmt: skip
                s.add(c)
                s.flush()
                ids.append(c.id)
            out.sources.append(src.id)
            out.streamers.append(st.id)
            out.cands[src.id] = ids
            out.runs[src.id] = run.id
        # final_rank inside each source follows final_score
        for ids in out.cands.values():
            rows = sorted((s.get(Candidate, i) for i in ids), key=lambda c: -c.final_score)
            for r, c in enumerate(rows, 1):
                c.final_rank = r
                s.add(c)
        first = out.cands[out.sources[0]][0]
        s.add(CandidateEnrichment(candidate_id=first, source_id=out.sources[0], prompt_version="v1",
                                  model="fake", input_hash="h", motivo="perseguição engraçada",
                                  scores={"nota_geral": 8}, titulos=["FUGIU DA POLÍCIA"],
                                  texto_de_abertura="ELE FUGIU",
                                  alertas=[{"tipo": "xingamento_pesado", "trecho": "p*rra"}]))  # fmt: skip
        s.commit()
    if transcript:
        sid = out.sources[0]
        tr = settings.work_dir / str(sid) / "transcript.json.gz"
        save_transcript(tr, [Word(a, b, w) for a, b, w in WORDS], {"language": "pt"})
        save_artifact(engine, sid, "transcribe", "h", files={"transcript": settings.rel_path(tr)})
    return out


def write_wav(path, seconds: float, bursts: list[float], sr: int = 16000, amp: float = 0.8):
    """Quiet noise with 50 ms loud bursts at the given times (seconds into the WAV)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, int(seconds * sr)).astype(np.float32)
    for t in bursts:
        a = int(t * sr)
        x[a : a + int(0.05 * sr)] = amp
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return path
