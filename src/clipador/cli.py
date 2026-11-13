"""Command line: `clipador web`, `clipador worker`, `clipador doctor`, `clipador migrate`."""

from __future__ import annotations

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler

from clipador.config import Settings, get_settings


def setup_logging(settings: Settings, name: str) -> None:
    settings.ensure_dirs()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    file = RotatingFileHandler(
        settings.logs_dir / f"{name}.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file.setFormatter(fmt)
    root.handlers[:] = [console, file]


def _utf8_stdout() -> None:
    # Windows consoles may default to a legacy code page; PT-BR text needs UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def cmd_doctor(settings: Settings) -> int:
    from clipador.services.doctor import CheckStatus, run_all

    icons = {CheckStatus.OK: "[ OK ]", CheckStatus.WARN: "[AVISO]", CheckStatus.FAIL: "[FALHA]"}
    print("Clipador doctor: verificando o ambiente…\n")
    results = run_all(settings)
    for r in results:
        print(f"{icons[r.status]:8} {r.name}: {r.detail}")
        if (r.hint and r.status != CheckStatus.OK) or (r.hint and r.name == "yt-dlp"):
            print(f"         dica: {r.hint}")
    fails = sum(r.status == CheckStatus.FAIL for r in results)
    warns = sum(r.status == CheckStatus.WARN for r in results)
    print(f"\n{len(results)} verificações · {fails} falha(s) · {warns} aviso(s)")
    return 1 if fails else 0


def cmd_web(settings: Settings, host: str | None, port: int | None) -> int:
    import uvicorn

    from clipador.web.app import create_app

    setup_logging(settings, "web")
    app = create_app(settings)
    uvicorn.run(app, host=host or settings.web_host, port=port or settings.web_port)
    return 0


def cmd_worker(settings: Settings) -> int:
    from clipador.worker.runner import Worker, WorkerAlreadyRunning

    setup_logging(settings, "worker")
    try:
        Worker(settings).run_forever()
    except WorkerAlreadyRunning as e:
        print(e.message, file=sys.stderr)
        return 2
    return 0


def cmd_migrate(settings: Settings) -> int:
    from clipador.db import run_migrations

    run_migrations(settings)
    print(f"Banco atualizado: {settings.db_path}")
    return 0


def _engine(settings: Settings):
    from clipador.db import make_engine, run_migrations

    run_migrations(settings)
    return make_engine(settings)


def cmd_dump(settings: Settings, args) -> int:
    from pathlib import Path

    from clipador.errors import ClipadorError
    from clipador.services.calibration import dump

    try:
        print(dump(_engine(settings), args.source_id, args.top, Path(args.out), args.run))
    except ClipadorError as e:
        print(e.message, file=sys.stderr)
        return 1
    return 0


def cmd_eval(settings: Settings, args) -> int:
    from pathlib import Path

    from clipador.errors import ClipadorError
    from clipador.services.calibration import run_eval

    if args.tolerance is not None:
        settings = settings.model_copy(update={"eval_tolerance_seconds": args.tolerance})
    try:
        _res, text = run_eval(
            _engine(settings), settings, args.source_id, Path(args.golden), args.run
        )
    except ClipadorError as e:
        print(e.message, file=sys.stderr)
        return 1
    print(text)
    return 0


def cmd_bench_transcribe(settings: Settings, args) -> int:
    """Measure Whisper speed (x real time) and peak VRAM on a WAV/video, without the queue."""
    import shutil
    import subprocess
    import tempfile
    import threading
    import time
    from pathlib import Path

    from clipador.services import audio
    from clipador.services.transcriber import FasterWhisperTranscriber

    src = Path(args.audio)
    if not src.is_file():
        print(f"Arquivo não encontrado: {src}", file=sys.stderr)
        return 1
    overrides = {"language": args.language}
    if args.model:
        overrides["whisper_model"] = args.model
    if args.compute:
        overrides["whisper_compute_type"] = args.compute
    settings = settings.model_copy(update=overrides)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "bench.wav"
        audio.extract_wav(src, wav, start=0, end=args.seconds, duration_hint=None,
                          progress_cb=lambda p: None, ffmpeg=settings.ffmpeg_path)  # fmt: skip
        dur = audio.WavReader(wav).duration
        peak = {"mb": 0}
        stop = threading.Event()

        def poll_vram():
            while not stop.is_set() and shutil.which("nvidia-smi"):
                try:
                    out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5,
                    ).stdout  # fmt: skip
                    peak["mb"] = max(peak["mb"], int(out.split()[0]))
                except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                    pass
                stop.wait(0.5)

        threading.Thread(target=poll_vram, daemon=True).start()
        tr = FasterWhisperTranscriber(settings, print)
        lang = None if settings.language == "auto" else settings.language
        t0 = time.monotonic()
        tr._load()
        t_load = time.monotonic() - t0
        t1 = time.monotonic()
        words = tr.transcribe(wav, lang, lambda f: None)
        t_tx = max(time.monotonic() - t1, 1e-3)
        stop.set()
        tr.close()
    print(
        f"\nmodelo={settings.whisper_model} compute={tr.compute_type} device={tr.device} "
        f"idioma={tr.detected_language}\n"
        f"áudio: {dur / 60:.1f} min · carga do modelo: {t_load:.1f}s · transcrição: {t_tx:.1f}s · "
        f"{len(words)} palavras\n"
        f"velocidade: {dur / t_tx:.1f}x tempo real · ~{3600 / (dur / t_tx) / 60:.1f} min por hora "
        f"de VOD · pico de VRAM (GPU inteira): {peak['mb']} MiB"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    parser = argparse.ArgumentParser(prog="clipador", description="Clipador: VODs -> clips 9:16")
    sub = parser.add_subparsers(dest="command", required=True)
    web = sub.add_parser("web", help="sobe a interface web")
    web.add_argument("--host")
    web.add_argument("--port", type=int)
    sub.add_parser("worker", help="processa a fila de jobs")
    sub.add_parser("doctor", help="verifica o ambiente")
    sub.add_parser("migrate", help="aplica as migrações do banco")
    d = sub.add_parser("dump", help="mostra os melhores candidatos e salva candidates.csv")
    d.add_argument("source_id", type=int)
    d.add_argument("--top", type=int, default=20)
    d.add_argument("--run", type=int, help="id da análise (padrão: a mais recente)")
    d.add_argument("--out", default="candidates.csv")
    ev = sub.add_parser("eval", help="mede recall@N contra momentos marcados à mão")
    ev.add_argument("source_id", type=int)
    ev.add_argument("--golden", required=True, help="arquivo com um HH:MM:SS por linha")
    ev.add_argument("--run", type=int, help="id da análise (padrão: a mais recente)")
    ev.add_argument("--tolerance", type=float, help="folga em segundos (padrão 2)")
    b = sub.add_parser("bench-transcribe", help="mede a velocidade do Whisper neste PC")
    b.add_argument("audio", help="vídeo ou áudio (ex.: data/vods/1/video.mp4)")
    b.add_argument("--seconds", type=float, default=600, help="quanto do início usar (padrão 600)")
    b.add_argument("--model")
    b.add_argument("--compute")
    b.add_argument("--language", default="auto")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.command == "bench-transcribe":
        return cmd_bench_transcribe(settings, args)
    if args.command == "dump":
        return cmd_dump(settings, args)
    if args.command == "eval":
        return cmd_eval(settings, args)
    if args.command == "doctor":
        return cmd_doctor(settings)
    if args.command == "web":
        return cmd_web(settings, args.host, args.port)
    if args.command == "worker":
        return cmd_worker(settings)
    if args.command == "migrate":
        return cmd_migrate(settings)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
