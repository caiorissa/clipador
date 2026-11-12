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
    args = parser.parse_args(argv)

    settings = get_settings()
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
