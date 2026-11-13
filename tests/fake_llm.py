"""Fake LLM clients for tests (no Ollama, no GPU, no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from clipador.llm.client import LLMNetworkError, LLMResponse, LLMUnavailable, install_hint


def good_answer(nota: float = 8, **over: Any) -> dict[str, Any]:
    data = {
        "scores": {
            "humor": 7, "surpresa_choque": 8, "habilidade": 5, "emocao": 9,
            "faz_sentido_sozinho": 8, "forca_do_inicio": 7, "nota_geral": nota,
        },
        "motivo": "Reação forte a uma perseguição com final inesperado.",
        "precisa_de_contexto": False,
        "titulos": ["Fugiu da polícia do jeito mais absurdo", "Ninguém esperava esse final",
                    "A perseguição mais caótica do GTA 6"],
        "texto_de_abertura": "OLHA ISSO",
        "descricao": "Perseguição no GTA 6 que terminou do jeito mais inesperado possível.",
        "hashtags": ["#GTA6", "#Perseguição Policial", "gta6", "#humor"],
        "alertas": [],
    }  # fmt: skip
    data.update(over)
    return data


class OfflineLLM:
    def list_models(self) -> list[str]:
        raise LLMUnavailable(
            "Ollama fora do ar em http://localhost:11434 (ConnectError).", install_hint()
        )

    def chat(self, *a, **k):
        raise LLMUnavailable("Ollama fora do ar.", install_hint())

    def unload(self, model: str) -> None:
        pass


class FakeLLM:
    """Answers come from `responder(messages, call_index)` -> str | Exception, or a fixed list."""

    def __init__(
        self,
        answers: list[Any] | Callable[[list[dict[str, str]], int], Any] | None = None,
        models: tuple[str, ...] = ("qwen2.5:7b",),
    ):
        self.answers = answers if answers is not None else [good_answer()]
        self.models = list(models)
        self.calls: list[list[dict[str, str]]] = []
        self.schemas: list[Any] = []
        self.unloaded: list[str] = []

    def list_models(self) -> list[str]:
        return self.models

    def chat(self, model, messages, *, schema, temperature, num_ctx, keep_alive, timeout,
             num_predict=None) -> LLMResponse:  # fmt: skip
        idx = len(self.calls)
        self.calls.append(messages)
        self.schemas.append(schema)
        if callable(self.answers):
            ans = self.answers(messages, idx)
        else:
            ans = self.answers[min(idx, len(self.answers) - 1)]
        if isinstance(ans, Exception):
            raise ans
        content = ans if isinstance(ans, str) else json.dumps(ans, ensure_ascii=False)
        return LLMResponse(content, latency_ms=1200, prompt_tokens=900, output_tokens=250,
                           output_tokens_per_s=42.0)  # fmt: skip

    def unload(self, model: str) -> None:
        self.unloaded.append(model)


TIMEOUT = LLMNetworkError("ReadTimeout")
