"""Engine, sessions and migrations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, event
from sqlmodel import Session, create_engine

from clipador.config import Settings

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def make_engine(settings: Settings) -> Engine:
    settings.ensure_dirs()
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False, "timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - trivial
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")  # web + worker processes read/write concurrently
        cur.execute("PRAGMA busy_timeout=30000")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    return engine


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    with Session(engine, expire_on_commit=False) as session:
        yield session


def alembic_config(settings: Settings):
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", settings.database_url)
    return cfg


def run_migrations(settings: Settings) -> None:
    """Bring the database schema to the latest revision (safe to call on every start)."""
    from alembic import command

    settings.ensure_dirs()
    command.upgrade(alembic_config(settings), "head")
