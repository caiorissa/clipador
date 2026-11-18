"""Environment checks shown by `clipador doctor` and the /sistema page."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from clipador.config import Settings
from clipador.services import ffmpeg
from clipador.services.downloader import YtDlp

MIN_PYTHON = (3, 11)
DISK_WARN_GB = 50
DISK_FAIL_GB = 10


IS_MAC = sys.platform == "darwin"
FFMPEG_HINT = (
    "No Mac: `brew tap homebrew-ffmpeg/ffmpeg && brew install homebrew-ffmpeg/ffmpeg/ffmpeg` "
    "(o ffmpeg do core do Homebrew vem sem libass, e as legendas falham)."
    if IS_MAC
    else "Baixe o build 'essentials' em https://www.gyan.dev/ffmpeg/builds/, extraia e coloque a "
    "pasta bin no PATH (ou defina CLIPADOR_FFMPEG_PATH / CLIPADOR_FFPROBE_PATH)."
)


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
        FFMPEG_HINT,
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
        (
            "Instale com `brew install deno`."
            if IS_MAC
            else "Instale o Deno no PowerShell: `irm https://deno.land/install.ps1 | iex` e "
            "reabra o terminal."
        )
        + " Twitch e Kick não precisam dele.",
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


def _sysctl(key: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True,
                              timeout=10).stdout.strip()  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return ""


def check_apple_gpu() -> CheckResult:
    chip = _sysctl("machdep.cpu.brand_string") or platform.machine()
    try:
        mem_gb = int(_sysctl("hw.memsize")) / 1024**3
    except ValueError:
        mem_gb = 0
    if platform.machine() != "arm64":
        return CheckResult(
            "GPU (Mac)", CheckStatus.WARN, f"{chip}: Mac Intel, sem GPU para o Whisper",
            "A transcrição roda na CPU (lenta). Um Mac M1 ou mais novo usa a GPU (MLX).",
        )  # fmt: skip
    detail = f"{chip} · {mem_gb:.0f} GB de memória unificada (GPU via Metal/MLX)"
    if mem_gb and mem_gb < 16:
        return CheckResult(
            "GPU (Mac)", CheckStatus.WARN, detail,
            "Com menos de 16 GB, use CLIPADOR_WHISPER_MODEL=small e um LLM de ~3-4B no Ollama.",
        )  # fmt: skip
    return CheckResult("GPU (Mac)", CheckStatus.OK, detail)


def check_gpu() -> CheckResult:
    if IS_MAC:
        return check_apple_gpu()
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


def check_hw_encoder(settings: Settings) -> CheckResult:
    """The hardware H.264 encoder of this machine: NVENC on the PC, VideoToolbox on a Mac."""
    enc = ffmpeg.hw_encoder_candidates()[0]
    label = "Encoder VideoToolbox" if enc == ffmpeg.VIDEOTOOLBOX else "Encoder NVENC"
    ok, err = ffmpeg.try_encoder(enc, settings.ffmpeg_path)
    if ok:
        return CheckResult(label, CheckStatus.OK, f"{enc} funcionando (teste de encode de 0,2 s)")
    return CheckResult(
        label,
        CheckStatus.WARN,
        f"{enc} indisponível: {err}. A renderização vai usar libx264 (CPU, mais lento).",
        FFMPEG_HINT if IS_MAC else
        "Atualize o driver NVIDIA e use um ffmpeg recente (build da gyan.dev).",
    )  # fmt: skip


check_nvenc = check_hw_encoder  # old name (tests / imports)


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


def check_ollama(settings: Settings) -> list[CheckResult]:
    """Reachable? configured model installed? quick generation latency. With
    CLIPADOR_LLM_BACKEND=claude: is the Claude Code CLI installed and logged in."""
    from clipador.llm.client import LLMError, make_llm_client, ollama_status

    if settings.llm_backend == "claude":
        st = ollama_status(settings)
        if st.ok:
            return [CheckResult("IA (Claude)", CheckStatus.OK, st.message + " Sem custo extra: "
                                "usa o limite da sua assinatura.")]  # fmt: skip
        return [CheckResult("IA (Claude)", CheckStatus.WARN, st.message, st.hint)]

    st = ollama_status(settings)
    if not st.ok:
        return [
            CheckResult(
                "Ollama",
                CheckStatus.WARN,
                f"{st.message} A IA local é opcional: sem ela o app usa só a nota de detecção.",
                st.hint,
            )
        ]
    out = [
        CheckResult(
            "Ollama",
            CheckStatus.OK,
            f"acessível em {settings.llm_base_url}; modelos: {', '.join(st.models) or 'nenhum'}",
        )
    ]
    if not st.configured_installed:
        out.append(CheckResult("Modelo da IA", CheckStatus.WARN, st.message, st.hint))
        return out
    client = make_llm_client(settings)
    try:
        resp = client.chat(
            settings.llm_model, [{"role": "user", "content": "Responda apenas: ok"}],
            schema=None, temperature=0, num_ctx=512, keep_alive=0, timeout=180, num_predict=5,
        )  # fmt: skip
    except LLMError as e:
        hint = f"Veja os logs do Ollama; tente `ollama run {settings.llm_model}`."
        fail = CheckResult("Modelo da IA", CheckStatus.FAIL, f"geração de teste falhou: {e}", hint)
        return [*out, fail]
    speed = f" · {resp.output_tokens_per_s:.0f} tokens/s" if resp.output_tokens_per_s else ""
    out.append(
        CheckResult(
            "Modelo da IA",
            CheckStatus.OK,
            f"{settings.llm_model}: geração de teste em {resp.latency_ms / 1000:.1f}s "
            f"(inclui carregar o modelo){speed}",
        )
    )
    return out


REQUIRED_FILTERS = ("subtitles", "loudnorm", "ebur128", "alimiter", "boxblur", "overlay")


def check_render_filters(settings: Settings) -> CheckResult:
    """ffmpeg must have libass (subtitles filter) and the filters the renderer uses."""
    name = "ffmpeg para renderizar"
    try:
        res = subprocess.run(
            [settings.ffmpeg_path, "-hide_banner", "-filters"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30,
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired) as e:
        return CheckResult(name, CheckStatus.FAIL, f"não foi possível listar os filtros: {e}")
    have = {ln.split()[1] for ln in res.stdout.splitlines() if len(ln.split()) > 2}
    missing = [f for f in REQUIRED_FILTERS if f not in have]
    if missing:
        return CheckResult(
            name,
            CheckStatus.FAIL,
            "faltam filtros: " + ", ".join(missing)
            + (" (sem libass: legendas não podem ser queimadas)" if "subtitles" in missing else ""),
            FFMPEG_HINT,
        )  # fmt: skip
    return CheckResult(
        name, CheckStatus.OK, "libass (legendas) e " + ", ".join(REQUIRED_FILTERS[1:])
    )


def check_fonts(settings: Settings) -> CheckResult:
    from clipador.render.styles import DEFAULT_FONTS_DIR, EMBEDDED_FONT, load_styles

    d = settings.fonts_dir or DEFAULT_FONTS_DIR
    fonts = sorted(p.name for p in d.glob("*.[ot]tf")) if d.is_dir() else []
    if not fonts:
        return CheckResult(
            "Fonte das legendas", CheckStatus.FAIL, f"nenhuma fonte em {d}",
            "A fonte embutida (Anton, OFL) deve estar em assets/fonts; restaure o repositório.",
        )  # fmt: skip
    try:
        styles = load_styles(settings.caption_styles_path)
    except (OSError, ValueError) as e:
        return CheckResult("Fonte das legendas", CheckStatus.FAIL,
                           f"estilos de legenda inválidos: {e}",
                           "Confira config/caption_styles.toml.")  # fmt: skip
    license_ok = any((d / n).is_file() for n in ("OFL.txt", "LICENSE.txt", "LICENSE"))
    detail = f"{', '.join(fonts)} em {d} · {len(styles)} estilos ({', '.join(styles)})"
    if EMBEDDED_FONT not in "".join(fonts):
        return CheckResult("Fonte das legendas", CheckStatus.WARN, detail,
                           f"A fonte padrão '{EMBEDDED_FONT}' não foi encontrada.")  # fmt: skip
    if not license_ok:
        return CheckResult("Fonte das legendas", CheckStatus.WARN, detail,
                           "Falta o arquivo de licença da fonte (OFL.txt).")  # fmt: skip
    return CheckResult("Fonte das legendas", CheckStatus.OK, detail)


def check_clips_space(settings: Settings) -> CheckResult:
    settings.clips_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(settings.clips_dir).free / 1024**3
    used = sum(f.stat().st_size for f in settings.clips_dir.rglob("*.mp4")) / 1024**3
    detail = f"{settings.clips_dir} · {used:.1f} GB em cortes · {free_gb:.1f} GB livres"
    if free_gb < 5:
        return CheckResult("Espaço para cortes", CheckStatus.WARN, detail,
                           "Cada corte final tem ~10-40 MB; libere espaço.")  # fmt: skip
    return CheckResult("Espaço para cortes", CheckStatus.OK, detail)


def check_faster_whisper() -> CheckResult:
    try:
        from importlib.metadata import version

        fw, ct2 = version("faster-whisper"), version("ctranslate2")
    except Exception as e:
        return CheckResult(
            "faster-whisper", CheckStatus.FAIL, f"não instalado ({e})", "Rode `uv sync`."
        )
    return CheckResult("faster-whisper", CheckStatus.OK, f"faster-whisper {fw} · ctranslate2 {ct2}")


def check_mlx_whisper(settings: Settings) -> CheckResult:
    from clipador.errors import ClipadorError
    from clipador.services import transcriber as tx

    name = "Whisper na GPU (MLX)"
    try:
        backend = tx.resolve_backend(settings)
    except ClipadorError as e:
        return CheckResult(name, CheckStatus.FAIL, e.message)
    if backend != "mlx":
        if not tx.is_apple_silicon():
            return CheckResult(
                name, CheckStatus.WARN, "Mac Intel: transcrição na CPU (faster-whisper, int8)."
            )
        if settings.whisper_backend == "faster-whisper":
            return CheckResult(
                name, CheckStatus.WARN,
                "CLIPADOR_WHISPER_BACKEND=faster-whisper: transcrição na CPU (2-3x mais lenta).",
                "Remova a variável (ou use `auto`) para usar a GPU do Mac.",
            )  # fmt: skip
        return CheckResult(name, CheckStatus.FAIL, "mlx-whisper não instalado.", "Rode `uv sync`.")
    from importlib.metadata import version

    return CheckResult(
        name, CheckStatus.OK,
        f"mlx-whisper {version('mlx-whisper')} · mlx {version('mlx')} · modelo "
        f"'{tx.resolve_model(settings, 'mlx')}' (meça com `clipador bench-transcribe`)",
    )  # fmt: skip


def check_whisper_accel(settings: Settings) -> CheckResult:
    return check_mlx_whisper(settings) if IS_MAC else check_cuda_whisper(settings)


def check_cuda_whisper(settings: Settings) -> CheckResult:
    from clipador.services.cuda import CUDA_LIBS_HINT, cuda_status

    name = "CUDA para Whisper"
    st = cuda_status()
    if st.error:
        return CheckResult(name, CheckStatus.FAIL, st.error, CUDA_LIBS_HINT)
    if st.missing_dlls:
        return CheckResult(
            name,
            CheckStatus.FAIL,
            "cuBLAS/cuDNN não encontrados: " + ", ".join(st.missing_dlls),
            CUDA_LIBS_HINT,
        )
    if st.device_count == 0:
        return CheckResult(
            name,
            CheckStatus.WARN,
            "o ctranslate2 não enxerga nenhuma GPU CUDA: a transcrição vai rodar na CPU (int8), "
            "muito mais lenta.",
            "Atualize o driver NVIDIA (rode nvidia-smi para conferir).",
        )
    return CheckResult(
        name,
        CheckStatus.OK,
        f"{st.device_count} GPU(s) visível(is) · cuBLAS/cuDNN ok · modelo "
        f"'{_fw_model(settings)}' ({settings.whisper_compute_type})",
    )


def _fw_model(settings: Settings) -> str:
    from clipador.services.transcriber import resolve_model

    return resolve_model(settings, "faster-whisper")


def check_vram_free() -> CheckResult:
    name = "VRAM livre"
    if not shutil.which("nvidia-smi"):
        return CheckResult(name, CheckStatus.WARN, "nvidia-smi não encontrado.")
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
        )  # fmt: skip
        free, total = (int(v.strip()) for v in res.stdout.strip().splitlines()[0].split(","))
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as e:
        return CheckResult(name, CheckStatus.WARN, f"não foi possível ler: {e}")
    detail = f"{free / 1024:.1f} GB livres de {total / 1024:.1f} GB"
    if free < 3000:
        return CheckResult(
            name,
            CheckStatus.WARN,
            detail,
            "O Whisper 'medium' em float16 usa ~2–3 GB. Feche jogos/OBS durante a análise ou use "
            "CLIPADOR_WHISPER_COMPUTE_TYPE=int8_float16.",
        )
    return CheckResult(name, CheckStatus.OK, detail)


def check_twitch_downloader(settings: Settings) -> CheckResult:
    name = "TwitchDownloaderCLI"
    configured = settings.twitch_downloader_path
    exe = shutil.which(configured) or (configured if Path(configured).is_file() else None)
    hint = (
        "Baixe em https://github.com/lay295/TwitchDownloader/releases (TwitchDownloaderCLI-…-"
        + (
            "MacOS-Arm64.zip; depois `xattr -d com.apple.quarantine` no binário"
            if IS_MAC
            else "Windows-x64.zip"
        )
        + "), extraia e defina CLIPADOR_TWITCH_DOWNLOADER_PATH no .env com o caminho "
        "do executável. Sem ele, VODs da Twitch são analisados sem o sinal de chat."
    )
    if not exe:
        return CheckResult(name, CheckStatus.WARN, f"não encontrado ('{configured}').", hint)
    try:
        res = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=20,
        )  # fmt: skip
        version = (res.stdout.strip() or res.stderr.strip()).splitlines()[:1]
    except (OSError, subprocess.TimeoutExpired):
        version = []
    return CheckResult(name, CheckStatus.OK, f"{exe} {version[0] if version else ''}".strip())


def check_models_dir(settings: Settings) -> CheckResult:
    d = settings.models_dir
    d.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(d).free / 1024**3
    size_gb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1024**3
    detail = f"{d} · {size_gb:.1f} GB em modelos · {free_gb:.1f} GB livres"
    if free_gb < 5:
        return CheckResult(
            "Modelos Whisper",
            CheckStatus.WARN,
            detail,
            "O modelo 'medium' ocupa ~1,5 GB e o 'large-v3' ~3 GB. Libere espaço ou defina "
            "CLIPADOR_WHISPER_MODEL_DIR em outro disco.",
        )
    return CheckResult("Modelos Whisper", CheckStatus.OK, detail)


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
        check_hw_encoder(settings),
        check_render_filters(settings),
        check_fonts(settings),
        check_clips_space(settings),
        check_faster_whisper(),
        check_whisper_accel(settings),
        *([] if IS_MAC else [check_vram_free()]),  # unified memory: no separate VRAM
        check_models_dir(settings),
        check_twitch_downloader(settings),
        check_disk(settings),
        *check_ollama(settings),
    ]
