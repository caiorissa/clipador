"""Share the work with another computer: a small HTTP server that runs Whisper on ITS GPU
(e.g. the Windows PC's RTX), used by the main Clipador (e.g. the Mac) through
CLIPADOR_REMOTE_WHISPER_URL. Only transcription travels: the main machine sends a 16 kHz mono
WAV of a chunk (~10-20 MB) and gets the words back.

    PC:   uv run clipador servir-transcricao        (prints the address and the token)
    Mac:  CLIPADOR_REMOTE_WHISPER_URL=http://<ip-do-pc>:8765
          CLIPADOR_REMOTE_WHISPER_TOKEN=<token>

The model stays loaded; one transcription runs at a time on the GPU (the others wait in line).
Every request needs the token (Authorization: Bearer ...). Meant for the home network only.
"""

from __future__ import annotations

import secrets
import socket
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request

from clipador.config import Settings

TOKEN_FILE = "transcription_server_token.txt"


def server_token(settings: Settings) -> str:
    """CLIPADOR_TRANSCRIPTION_SERVER_TOKEN, or one generated once and kept in data/."""
    if settings.transcription_server_token:
        return settings.transcription_server_token
    f = settings.data_dir / TOKEN_FILE
    if f.is_file() and f.read_text("utf-8").strip():
        return f.read_text("utf-8").strip()
    f.parent.mkdir(parents=True, exist_ok=True)
    tok = secrets.token_urlsafe(24)
    f.write_text(tok, "utf-8")
    return tok


def lan_ip() -> str:
    """This computer's address on the home network (no packet is actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.0.1", 9))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def create_app(settings: Settings, transcriber=None):
    """FastAPI app. `transcriber` is injectable for tests (default: this machine's Whisper)."""
    from clipador.services import transcriber as tx

    token = server_token(settings)
    gpu = threading.Lock()
    state: dict = {"t": transcriber, "busy": 0, "served": 0, "audio_s": 0.0, "work_s": 0.0}
    app = FastAPI(title="Clipador · transcrição", docs_url=None, redoc_url=None)

    def check(auth: str | None) -> None:
        if not auth or not secrets.compare_digest(auth.removeprefix("Bearer ").strip(), token):
            raise HTTPException(status_code=401, detail="token inválido")

    def model():
        if state["t"] is None:
            state["t"] = tx.make_transcriber(settings, lambda m: print(m, flush=True))
        return state["t"]

    @app.get("/health")
    def health(authorization: str | None = Header(default=None)):
        check(authorization)
        speed = state["audio_s"] / state["work_s"] if state["work_s"] else None
        return {"ok": True, "backend": tx.resolve_backend(settings),
                "model": tx.resolve_model(settings), "busy": state["busy"],
                "served": state["served"], "speed": round(speed, 1) if speed else None}  # fmt: skip

    @app.post("/transcribe")
    async def transcribe(request: Request, language: str = "",
                         authorization: str | None = Header(default=None)):  # fmt: skip
        check(authorization)
        body = await request.body()
        if not body.startswith(b"RIFF"):
            raise HTTPException(status_code=400, detail="envie um WAV")
        from starlette.concurrency import run_in_threadpool

        from clipador.services.audio import WavReader

        def work(wav: Path):
            with gpu:  # one at a time on the GPU (the others wait here)
                t0 = time.monotonic()
                t = model()
                words = t.transcribe(wav, language or None, lambda f: None)
                return t, words, time.monotonic() - t0

        with tempfile.TemporaryDirectory() as d:
            wav = Path(d) / "chunk.wav"
            wav.write_bytes(body)
            dur = WavReader(wav).duration
            state["busy"] += 1
            try:
                t, words, dt = await run_in_threadpool(work, wav)
            finally:
                state["busy"] -= 1
        state["served"] += 1
        state["audio_s"] += dur
        state["work_s"] += dt
        print(f"[{time.strftime('%H:%M:%S')}] {dur / 60:.1f} min transcritos em {dt:.0f}s "
              f"({dur / max(dt, 1e-6):.0f}x) · {len(words)} palavras", flush=True)  # fmt: skip
        return {"language": getattr(t, "detected_language", None),
                "words": [[w.start, w.end, w.word, w.prob] for w in words]}  # fmt: skip

    return app


def serve(settings: Settings, port: int) -> int:
    import uvicorn

    from clipador.services import transcriber as tx

    tok = server_token(settings)
    ip = lan_ip()
    print("\n=== Clipador: servidor de transcrição ===")
    print(f"Whisper: {tx.resolve_model(settings)} ({tx.resolve_backend(settings)})")
    print("\nNo .env do Mac, coloque estas duas linhas e reinicie o worker de lá:\n")
    print(f"CLIPADOR_REMOTE_WHISPER_URL=http://{ip}:{port}")
    print(f"CLIPADOR_REMOTE_WHISPER_TOKEN={tok}\n")
    print("Se o Windows perguntar sobre o firewall, permita em 'Redes privadas'.")
    print("Deixe esta janela aberta. Ctrl+C para parar.\n", flush=True)
    uvicorn.run(create_app(settings), host="0.0.0.0", port=port, log_level="warning")
    return 0
