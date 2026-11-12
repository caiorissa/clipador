"""Stage registry: job.type -> Stage class."""

from __future__ import annotations

from clipador.errors import ClipadorError
from clipador.pipeline.base import Stage
from clipador.pipeline.stages.detect import DetectStage
from clipador.pipeline.stages.ingest import IngestStage
from clipador.pipeline.stages.render import RenderStage
from clipador.pipeline.stages.transcribe import TranscribeStage

STAGES: dict[str, type[Stage]] = {
    cls.name: cls for cls in (IngestStage, TranscribeStage, DetectStage, RenderStage)
}


def get_stage(name: str) -> type[Stage]:
    try:
        return STAGES[name]
    except KeyError:
        raise ClipadorError(f"Tipo de job desconhecido: '{name}'.") from None
