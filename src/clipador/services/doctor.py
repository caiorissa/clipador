"""Environment checks shown by `clipador doctor` and the /sistema page."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum

import httpx

from clipador.config import Settings
from clipador.services import ffmpeg
from clipador.services.downloader import YtDlp

MIN_PYTHON = (3, 11)
DISK_WARN_GB = 50
DISK_FAIL_GB = 10


class CheckStatus(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    detail: str
    hint: str = ""


def check_system() -> CheckResult:
    return CheckResult(
        "Sistema operacional",
        CheckStatus.OK,
        f"{platform.system()} {platform.release()} ({platform.version()}) · {platform.machine()}",
    )


def check_python() -> CheckResult:
    v = sys.version_info
    detail = f"Python {v.major}.{v.minor}.{v.micro} ({sys.executable})"
    if (v.major, v.minor) >= MIN_PYTHON:
        return CheckResult("Python", CheckStatus.OK, detail)
    return CheckResult(
        "Python", CheckStatus.FAIL, detail, "Instale o Python 3.11 ou mais novo e rode `uv sync`."
    )


def check_binary(label: str, binary: str) -> CheckResult:
    version = ffmpeg.tool_version(binary)
    if version:
        return CheckResult(label, CheckStatus.OK, version)
    return CheckResult(
        label,
        CheckStatus.FAIL,
        f"'{binary}' não encontrado ou não executa.",
        "Baixe o build 'essentials' em https://www.gyan.dev/ffmpeg/builds/, extraia e coloque a "
        "pasta bin no PATH (ou defina CLIPADOR_FFMPEG_PATH / CLIPADOR_FFPROBE_PATH).",
    )


def check_ytdlp(settings: Settings) -> CheckResult:
    yt = YtDlp(settings)
    version = yt.version()
    update_hint = (
        "Extratores quebram com frequência (principalmente Kick e YouTube). Se um download falhar, "
        "atualize: `uv lock --upgrade-package yt-dlp` e `uv sync` "
        "(ou `yt-dlp -U` se usar o executável avulso)."
    )
    if version:
        return CheckResult("yt-dlp", CheckStatus.OK, f"versão {version}", update_hint)
    return CheckResult(
        "yt-dlp",
        CheckStatus.FAIL,
        "yt-dlp não executa (" + " ".join(yt.base_cmd()) + ").",
        "Rode `uv sync` para instalar, ou ajuste CLIPADOR_YTDLP_COMMAND.",
    )


def check_js_runtime() -> CheckResult:
    """yt-dlp needs a JavaScript runtime (Deno by default) to solve YouTube's challenges."""
    name = "Runtime JS (YouTube)"
    deno = shutil.which("deno")
    if deno:
        try:
            res = subprocess.run(
                [deno, "--version"], capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=15,
            )  # fmt: skip
            version = (res.stdout.splitlines() or ["deno"])[0]
        except (OSError, subprocess.TimeoutExpired):
            version = deno
        return CheckResult(name, CheckStatus.OK, version)
    return CheckResult(
        name,
        CheckStatus.WARN,
        "Deno não encontrado: no YouTube o yt-dlp pode não achar todos os formatos (ex.: 1080p).",
        "Instale o Deno no PowerShell: `irm https://deno.land/install.ps1 | iex` e reabra o "
        "terminal. Twitch e Kick não precisam dele.",
    )


def check_cookies(settings: Settings) -> CheckResult:
    if settings.cookies_file:
        if settings.cookies_file.is_file():
            return CheckResult("Cookies", CheckStatus.OK, f"arquivo {settings.cookies_file}")
        return CheckResult(
            "Cookies",
            CheckStatus.FAIL,
            f"arquivo de cookies não existe: {settings.cookies_file}",
            "Corrija CLIPADOR_COOKIES_FILE ou remova a variável.",
        )
    if settings.cookies_from_browser:
        return CheckResult(
            "Cookies", CheckStatus.OK, f"do navegador '{settings.cookies_from_browser}'"
        )
    return CheckResult(
        "Cookies",
        CheckStatus.OK,
        "não configurados (opcional)",
        "Se YouTube/Kick pedirem login ou bloquearem, use CLIPADOR_COOKIES_FROM_BROWSER.",
    )


def check_gpu() -> CheckResult:
    if not shutil.which("nvidia-smi"):
        return CheckResult(
            "GPU NVIDIA",
            CheckStatus.WARN,
            "nvidia-smi não encontrado.",
            "Instale/atualize o driver NVIDIA. Sem GPU o app funciona, mas transcrição e "
            "renderização ficam bem mais lentas.",
        )
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired) as e:
        return CheckResult("GPU NVIDIA", CheckStatus.WARN, f"nvidia-smi falhou: {e}")
    if res.returncode != 0 or not res.stdout.strip():
        return CheckResult(
            "GPU NVIDIA",
            CheckStatus.WARN,
            "nvidia-smi não listou GPUs: " + (res.stderr.strip() or res.stdout.strip())[:300],
            "Reinstale o driver NVIDIA.",
        )
    gpus = []
    for line in res.stdout.strip().splitlines():
        name, driver, total, used = (p.strip() for p in line.split(","))
        gpus.append(f"{name} · driver {driver} · VRAM {used} usada de {total}")
    return CheckResult("GPU NVIDIA", CheckStatus.OK, "; ".join(gpus))


def check_nvenc(settings: Settings) -> CheckResult:
    ok, err = ffmpeg.try_encoder(ffmpeg.NVENC, settings.ffmpeg_path)
    if ok:
        return CheckResult(
            "Encoder NVENC", CheckStatus.OK, "h264_nvenc funcionando (teste de encode de 0,2 s)"
        )
    return CheckResult(
        "Encoder NVENC",
        CheckStatus.WARN,
        f"h264_nvenc indisponível: {err}. A renderização vai usar libx264 (CPU, mais lento).",
        "Atualize o driver NVIDIA e use um ffmpeg recente (build da gyan.dev).",
    )


def check_disk(settings: Settings) -> CheckResult:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(settings.data_dir)
    free_gb = usage.free / 1024**3
    detail = f"{free_gb:.1f} GB livres em {settings.data_dir}"
    hint = (
        "VODs longos ocupam vários GB. Libere espaço ou aponte CLIPADOR_DATA_DIR para outro disco."
    )
    if free_gb < DISK_FAIL_GB:
        return CheckResult("Espaço em disco", CheckStatus.FAIL, detail, hint)
    if free_gb < DISK_WARN_GB:
        return CheckResult("Espaço em disco", CheckStatus.WARN, detail, hint)
    return CheckResult("Espaço em disco", CheckStatus.OK, detail)


def check_ollama(settings: Settings) -> CheckResult:
    url = settings.ollama_url.rstrip("/")
    try:
        res = httpx.get(f"{url}/api/tags", timeout=2.0)
        res.raise_for_status()
        models = [m.get("name", "?") for m in res.json().get("models", [])]
    except (httpx.HTTPError, ValueError) as e:
        return CheckResult(
            "Ollama",
            CheckStatus.WARN,
            f"não acessível em {url} ({type(e).__name__}). Opcional por enquanto.",
            "Será usado a partir do dia 3. Instale em https://ollama.com/download.",
        )
    detail = f"acessível em {url}; modelos: {', '.join(models) if models else 'nenhum ainda'}"
    return CheckResult("Ollama", CheckStatus.OK, detail)


def run_all(settings: Settings) -> list[CheckResult]:
    return [
        check_system(),
        check_python(),
        check_binary("ffmpeg", settings.ffmpeg_path),
        check_binary("ffprobe", settings.ffprobe_path),
        check_ytdlp(settings),
        check_js_runtime(),
        check_cookies(settings),
        check_gpu(),
        check_nvenc(settings),
        check_disk(settings),
        check_ollama(settings),
    ]
