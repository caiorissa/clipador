"""Thin Ollama HTTP client (httpx) behind a Protocol so tests can use a fake."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from clipador.config import Settings


class LLMError(Exception):
    """Generic LLM failure for one call (the stage moves on to the next candidate)."""


class LLMNetworkError(LLMError):
    """Timeout / dropped connection during a call (retried once)."""


class LLMUnavailable(LLMError):
    """Ollama is not installed/running or the model is missing: the stage is skipped."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint


@dataclass
class LLMResponse:
    content: str
    latency_ms: int
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    output_tokens_per_s: float | None = None

    @property
    def tokens(self) -> int | None:
        if self.prompt_tokens is None and self.output_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.output_tokens or 0)


class LLMClient(Protocol):
    def list_models(self) -> list[str]: ...

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None,
        temperature: float,
        num_ctx: int,
        keep_alive: str | int,
        timeout: float,
        num_predict: int | None = None,
    ) -> LLMResponse: ...

    def unload(self, model: str) -> None: ...


def install_hint() -> str:
    return (
        "Instale o Ollama (https://ollama.com/download), abra-o (ou rode `ollama serve`) e baixe "
        "um modelo, ex.: `ollama pull qwen2.5:7b`."
    )


def model_installed(installed: list[str], model: str) -> bool:
    """'qwen2.5:7b' matches 'qwen2.5:7b'; 'llama3.1' matches 'llama3.1:latest'."""
    want = model if ":" in model else f"{model}:latest"
    return any(m in (want, model) for m in installed)


class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _unavailable(self, e: Exception) -> LLMUnavailable:
        return LLMUnavailable(
            f"Ollama fora do ar em {self.base_url} ({type(e).__name__}).", install_hint()
        )

    def list_models(self) -> list[str]:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            r.raise_for_status()
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise self._unavailable(e) from e
        except httpx.HTTPError as e:
            raise LLMUnavailable(f"Ollama respondeu com erro: {e}", install_hint()) from e
        return [m.get("name", "") for m in r.json().get("models", [])]

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None,
        temperature: float,
        num_ctx: int,
        keep_alive: str | int,
        timeout: float,
        num_predict: int | None = None,
    ) -> LLMResponse:
        options: dict[str, Any] = {"temperature": temperature, "num_ctx": num_ctx}
        if num_predict is not None:
            options["num_predict"] = num_predict
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
            "keep_alive": keep_alive,
            "think": False,  # reasoning models: answer directly (structured output)
        }
        if schema is not None:
            body["format"] = schema
        t0 = time.monotonic()
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=body, timeout=timeout)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise self._unavailable(e) from e
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as e:
            raise LLMNetworkError(f"{type(e).__name__}: {e}") from e
        if r.status_code == 404:
            raise LLMUnavailable(
                f"Modelo '{model}' não está instalado no Ollama.", f"Rode `ollama pull {model}`."
            )
        if r.status_code == 400 and "think" in r.text:
            # older Ollama or model without thinking support: retry without the flag
            body.pop("think", None)
            r = httpx.post(f"{self.base_url}/api/chat", json=body, timeout=timeout)
        if r.status_code >= 400:
            raise LLMError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        latency = int((time.monotonic() - t0) * 1000)
        out_tokens = data.get("eval_count")
        eval_ns = data.get("eval_duration")
        tps = out_tokens / (eval_ns / 1e9) if out_tokens and eval_ns else None
        return LLMResponse(
            content=(data.get("message") or {}).get("content", ""),
            latency_ms=latency,
            prompt_tokens=data.get("prompt_eval_count"),
            output_tokens=out_tokens,
            output_tokens_per_s=tps,
        )

    def unload(self, model: str) -> None:
        """Free VRAM now (keep_alive=0)."""
        with contextlib.suppress(httpx.HTTPError):
            httpx.post(
                f"{self.base_url}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30
            )


def make_llm_client(settings: Settings) -> LLMClient:
    """Factory used by the enrich stage, doctor and CLI (tests monkeypatch it)."""
    return OllamaClient(settings.llm_base_url)


@dataclass
class OllamaStatus:
    ok: bool
    models: list[str]
    configured: str
    configured_installed: bool
    message: str
    hint: str = ""


def ollama_status(settings: Settings) -> OllamaStatus:
    """For the UI/doctor: is Ollama up, which models, is the configured one installed."""
    client = make_llm_client(settings)
    try:
        models = client.list_models()
    except LLMUnavailable as e:
        return OllamaStatus(False, [], settings.llm_model, False, e.message, e.hint)
    installed = model_installed(models, settings.llm_model)
    if installed:
        return OllamaStatus(True, models, settings.llm_model, True, "Ollama ok.")
    return OllamaStatus(
        True,
        models,
        settings.llm_model,
        False,
        f"Ollama ok, mas o modelo '{settings.llm_model}' não está instalado.",
        f"Rode `ollama pull {settings.llm_model}` (ou escolha um modelo instalado).",
    )
