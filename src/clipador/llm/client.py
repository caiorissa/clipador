"""Thin Ollama HTTP client (httpx) behind a Protocol so tests can use a fake."""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
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


class ClaudeCodeClient:
    """The Claude Code CLI already installed on this machine, in non-interactive mode
    (`claude -p`), using the user's Claude subscription: no API key, no extra cost (it counts
    against the subscription's usage limits).

    Every call is isolated on purpose: own system prompt, NO tools (`--tools ""`: the
    transcript and chat are third-party text and must never be able to make it act), no MCP
    servers, no settings/plugins/skills and no saved session. That keeps each call around a
    thousand tokens instead of the ~240k of a normal Claude Code session.
    """

    def __init__(self, settings: Settings, binary: str | None = None):
        self.settings = settings
        self.binary = binary or settings.claude_binary

    def _exe(self) -> str:
        exe = shutil.which(self.binary) or (self.binary if Path(self.binary).is_file() else None)
        if not exe:
            raise LLMUnavailable(
                "Claude Code não encontrado neste computador.",
                "Instale o Claude Code (https://claude.com/download) e entre na sua conta, ou "
                "use CLAUDE_BINARY / CLIPADOR_LLM_BACKEND=ollama.",
            )
        return exe

    def list_models(self) -> list[str]:
        paused = claude_paused(self.settings)
        if paused:
            raise LLMUnavailable(paused, "A IA volta sozinha quando o limite renovar.")
        exe = self._exe()
        try:
            r = subprocess.run([exe, "auth", "status"], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=30)  # fmt: skip
            status = json.loads(r.stdout or "{}")
        except (OSError, subprocess.TimeoutExpired, ValueError) as e:
            raise LLMUnavailable(f"Não consegui falar com o Claude Code ({e}).") from e
        if not status.get("loggedIn"):
            raise LLMUnavailable("O Claude Code não está logado.", LOGIN_HINT)
        return [CLAUDE_PREFIX + m for m in CLAUDE_MODELS]

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
        exe = self._exe()
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        if len(convo) == 1:
            prompt = convo[0]["content"]
        else:  # a retry with a correction: replay the exchange as text
            prompt = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in convo)
        workdir = self.settings.tmp_dir / "claude"
        workdir.mkdir(parents=True, exist_ok=True)  # empty dir: no CLAUDE.md is picked up
        cmd = [
            exe, "-p", "--output-format", "json", "--tools", "", "--no-session-persistence",
            "--model", model.removeprefix(CLAUDE_PREFIX) or self.settings.claude_model,
            "--system-prompt", system or "Responda só no formato pedido.",
            "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
            "--setting-sources", "", "--disable-slash-commands",
        ]  # fmt: skip
        if schema is not None:
            cmd += ["--json-schema", json.dumps(schema)]
        t0 = time.monotonic()
        try:
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=timeout,
                               cwd=str(workdir))  # fmt: skip
        except subprocess.TimeoutExpired as e:
            raise LLMNetworkError(f"Claude Code demorou mais de {timeout:.0f}s") from e
        except OSError as e:
            raise LLMUnavailable(f"Não consegui rodar o Claude Code ({e}).") from e
        latency = int((time.monotonic() - t0) * 1000)
        try:
            data = json.loads(r.stdout)
        except ValueError as e:
            raise self._failure(r.stderr or r.stdout) from e
        if data.get("is_error") or data.get("subtype") != "success":
            raise self._failure(str(data.get("result") or data.get("api_error_status") or data))
        out = data.get("structured_output")
        content = json.dumps(out, ensure_ascii=False) if out is not None else data.get("result", "")
        usage = data.get("usage") or {}
        prompt_tokens = sum(int(usage.get(k) or 0) for k in (
            "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))  # fmt: skip
        return LLMResponse(content=content, latency_ms=latency, prompt_tokens=prompt_tokens,
                           output_tokens=usage.get("output_tokens"))  # fmt: skip

    def _failure(self, text: str) -> LLMError:
        if is_limit_message(text):
            when = _reset_hint(text)
            pause_claude(self.settings, text)
            return LLMUnavailable(
                "Limite de uso da sua assinatura do Claude atingido"
                + (f" (renova {when})" if when else " por agora")
                + ".",
                "Os clips sem nota da IA são avaliados quando o limite renovar; até lá vale a "
                "nota da detecção. Ou use CLIPADOR_LLM_BACKEND=ollama nesse meio-tempo.",
            )
        low = text.lower()
        if any(k in low for k in ("not logged", "log in", "login", "authentication", "401")):
            return LLMUnavailable("O Claude Code não está logado.", LOGIN_HINT)
        return LLMError(f"Claude Code: {text.strip()[:300]}")

    def unload(self, model: str) -> None:
        """Nothing to free: Claude runs in Anthropic's cloud."""


def is_limit_message(text: str) -> bool:
    """'You've hit your 5-hour limit · resets 2:40pm', 'usage limit reached', 429, overloaded."""
    low = text.lower()
    if any(k in low for k in ("429", "overloaded", "rate limit", "usage limit")):
        return True
    return "limit" in low and any(k in low for k in ("hit", "reset", "reached", "exceeded"))


def _reset_hint(text: str) -> str:
    import re

    m = re.search(r"resets?\s+(?:at\s+)?([0-9][0-9:]*\s*(?:am|pm)?)", text, re.IGNORECASE)
    return f"às {m.group(1).strip()}" if m else ""


PAUSE_MINUTES = 20  # after a limit: don't call Claude again for a while (no wasted calls)


def _pause_file(settings: Settings) -> Path:
    return settings.data_dir / "claude_limit.json"


def pause_claude(settings: Settings, text: str) -> None:
    with contextlib.suppress(OSError):
        _pause_file(settings).write_text(
            json.dumps({"until": time.time() + PAUSE_MINUTES * 60, "message": text.strip()[:300]}),
            "utf-8",
        )


def claude_paused(settings: Settings) -> str | None:
    """PT-BR message while the subscription limit pause is on, else None."""
    try:
        data = json.loads(_pause_file(settings).read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("until", 0) <= time.time():
        return None
    when = _reset_hint(data.get("message", ""))
    return (
        "Limite de uso da sua assinatura do Claude atingido"
        + (f" (renova {when})" if when else "")
        + "."
    )


CLAUDE_PREFIX = "claude:"
LOGIN_HINT = "Abra um terminal, rode `claude` e entre na sua conta."
CLAUDE_MODELS = ("sonnet", "opus", "haiku")


def default_model(settings: Settings) -> str:
    """Model the enrich stage uses when none is given."""
    if settings.llm_backend == "claude":
        return CLAUDE_PREFIX + settings.claude_model
    return settings.llm_model


def prompt_name(settings: Settings) -> str:
    """The enrich prompt to use (CLIPADOR_ENRICH_PROMPT, or by backend when 'auto')."""
    if settings.enrich_prompt != "auto":
        return settings.enrich_prompt
    return "enrich_v2" if settings.llm_backend == "claude" else "enrich_v1"


def make_llm_client(settings: Settings) -> LLMClient:
    """Factory used by the enrich stage, doctor and CLI (tests monkeypatch it)."""
    if settings.llm_backend == "claude":
        return ClaudeCodeClient(settings)
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
    """For the UI/doctor: is the AI backend up (Ollama, or Claude Code logged in), which
    models, is the configured one available."""
    client = make_llm_client(settings)
    if settings.llm_backend == "claude":
        try:
            models = client.list_models()
        except LLMUnavailable as e:
            return OllamaStatus(False, [], default_model(settings), False, e.message, e.hint)
        msg = f"Claude ({settings.claude_model}) pela sua assinatura: ok."
        return OllamaStatus(True, models, default_model(settings), True, msg)
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
