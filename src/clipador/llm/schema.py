"""Structured output of the enrich prompt: JSON Schema sent to Ollama + pydantic validation.

Validation is strict about structure (unknown keys are rejected, so anything outside the schema
fails) and lenient about sizes (long strings are cut at a word boundary, scores clamped to 0-10).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ALERT_TYPES = (
    "xingamento_pesado",
    "discurso_de_odio_ou_ofensa",
    "violencia_real_ou_autolesao",
    "conteudo_sexual",
    "dados_pessoais_ou_doxxing",
    "apostas_ou_promocao",
    "outro",
)
SCORE_KEYS = (
    "humor",
    "surpresa_choque",
    "habilidade",
    "emocao",
    "faz_sentido_sozinho",
    "forca_do_inicio",
    "nota_geral",
)

MAX_TITLE = 70
MAX_MOTIVO = 200
MAX_DESCRICAO = 300
MAX_ABERTURA = 60
MAX_HASHTAG = 30
MAX_ALERT_SNIPPET = 160


def truncate_words(text: str, limit: int, ellipsis: str = "…") -> str:
    """Cut to `limit` chars without breaking a word (adds an ellipsis when cut)."""
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    cut = text[: max(0, limit - len(ellipsis))]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:-–—") + ellipsis


class Scores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    humor: float
    surpresa_choque: float
    habilidade: float
    emocao: float
    faz_sentido_sozinho: float
    forca_do_inicio: float
    nota_geral: float

    @field_validator("*", mode="after")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return round(min(10.0, max(0.0, float(v))), 1)


class Alerta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tipo: Literal[ALERT_TYPES]  # type: ignore[valid-type]
    trecho: str = ""

    @field_validator("trecho", mode="after")
    @classmethod
    def _cut(cls, v: str) -> str:
        return truncate_words(v, MAX_ALERT_SNIPPET)


class EnrichOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: Scores
    motivo: str
    precisa_de_contexto: bool
    titulos: list[str] = Field(min_length=1)
    texto_de_abertura: str | None = None
    descricao: str
    hashtags: list[str] = Field(default_factory=list)
    alertas: list[Alerta] = Field(default_factory=list)

    @field_validator("motivo", mode="after")
    @classmethod
    def _motivo(cls, v: str) -> str:
        return truncate_words(v, MAX_MOTIVO)

    @field_validator("descricao", mode="after")
    @classmethod
    def _descricao(cls, v: str) -> str:
        return truncate_words(v, MAX_DESCRICAO)

    @field_validator("texto_de_abertura", mode="after")
    @classmethod
    def _abertura(cls, v: str | None) -> str | None:
        v = truncate_words(v, MAX_ABERTURA) if v else None
        return v or None

    @field_validator("titulos", mode="after")
    @classmethod
    def _titulos(cls, v: list[str]) -> list[str]:
        out = [truncate_words(t, MAX_TITLE) for t in v if str(t).strip()]
        if not out:
            raise ValueError("nenhum título")
        return out[:3]

    @field_validator("hashtags", mode="after")
    @classmethod
    def _hashtags(cls, v: list[str]) -> list[str]:
        return [h.strip()[:MAX_HASHTAG] for h in v if h.strip()][:6]

    @field_validator("alertas", mode="after")
    @classmethod
    def _alertas(cls, v: list[Alerta]) -> list[Alerta]:
        return v[:10]


def json_schema() -> dict[str, Any]:
    """Hand-written, inline (no $ref) schema for Ollama's `format` (grammar-constrained)."""
    score = {"type": "number", "minimum": 0, "maximum": 10}
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": {k: score for k in SCORE_KEYS},
                "required": list(SCORE_KEYS),
            },
            "motivo": {"type": "string"},
            "precisa_de_contexto": {"type": "boolean"},
            "titulos": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
            "texto_de_abertura": {"type": "string"},
            "descricao": {"type": "string"},
            "hashtags": {
                "type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 6,
            },
            "alertas": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tipo": {"type": "string", "enum": list(ALERT_TYPES)},
                        "trecho": {"type": "string"},
                    },
                    "required": ["tipo", "trecho"],
                },
            },
        },
        "required": [
            "scores", "motivo", "precisa_de_contexto", "titulos", "texto_de_abertura",
            "descricao", "hashtags", "alertas",
        ],
    }  # fmt: skip
