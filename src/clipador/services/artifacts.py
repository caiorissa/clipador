"""Per-stage output cache: a StageArtifact row is valid while its params hash matches and its
files exist. Each stage hashes its own params plus the hash of its inputs, so changing e.g. only
detection weights re-runs only `detect`."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy import Engine
from sqlmodel import Session, select

from clipador.config import Settings
from clipador.models import ArtifactStatus, StageArtifact
from clipador.utils import utcnow


def params_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def file_hash(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()[:16]


def get_artifact(engine: Engine, source_id: int, stage: str) -> StageArtifact | None:
    with Session(engine) as s:
        return s.exec(
            select(StageArtifact).where(
                StageArtifact.source_id == source_id, StageArtifact.stage == stage
            )
        ).first()


def valid_artifact(
    engine: Engine, settings: Settings, source_id: int, stage: str, phash: str
) -> StageArtifact | None:
    art = get_artifact(engine, source_id, stage)
    if art is None or art.params_hash != phash:
        return None
    if any(not settings.abs_path(rel).is_file() for rel in art.files.values()):
        return None
    return art


def save_artifact(
    engine: Engine,
    source_id: int,
    stage: str,
    phash: str,
    *,
    files: dict[str, str] | None = None,
    summary: dict[str, Any] | None = None,
    status: str = ArtifactStatus.OK,
    message: str | None = None,
    elapsed: float | None = None,
) -> StageArtifact:
    with Session(engine, expire_on_commit=False) as s:
        art = s.exec(
            select(StageArtifact).where(
                StageArtifact.source_id == source_id, StageArtifact.stage == stage
            )
        ).first()
        if art is None:
            art = StageArtifact(source_id=source_id, stage=stage, params_hash=phash)
        art.params_hash = phash
        art.files = files or {}
        art.summary = summary or {}
        art.status = status
        art.message = message
        art.elapsed_seconds = elapsed
        art.updated_at = utcnow()
        s.add(art)
        s.commit()
        return art
