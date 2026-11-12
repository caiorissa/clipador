"""CUDA runtime setup for ctranslate2 (faster-whisper) and GPU diagnostics.

On Windows the cuBLAS/cuDNN DLLs come from the `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels
(installed by `uv sync`), which live in site-packages/nvidia/*/bin. Python does not search those
folders by default, so we register them before ctranslate2 loads.
"""

from __future__ import annotations

import contextlib
import ctypes
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_registered = False

# DLLs ctranslate2 4.x needs on Windows (CUDA 12 + cuDNN 9).
WINDOWS_CUDA_DLLS = ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll", "cudnn_ops64_9.dll")

CUDA_LIBS_HINT = (
    "Rode `uv sync` (instala nvidia-cublas-cu12 e nvidia-cudnn-cu12 no ambiente do projeto) e "
    "atualize o driver NVIDIA. Se ainda falhar, instale o CUDA Toolkit 12 e o cuDNN 9 da NVIDIA "
    "e coloque as pastas bin no PATH."
)


def nvidia_lib_dirs() -> list[Path]:
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return []
    dirs: list[Path] = []
    for root in spec.submodule_search_locations:
        for sub in sorted(Path(root).iterdir()):
            for name in ("bin", "lib"):
                d = sub / name
                if d.is_dir():
                    dirs.append(d)
    return dirs


def register_cuda_dlls() -> None:
    """Make the pip-installed CUDA libraries loadable (idempotent)."""
    global _registered
    if _registered:
        return
    _registered = True
    # ctranslate2 ships no native kernels for the newest GPUs (e.g. RTX 50 / Blackwell), so CUDA
    # JIT-compiles them on first use (~15 s). The default JIT cache is too small to keep them;
    # a 4 GB cache makes later runs start in ~1 s.
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(4 * 1024**3))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    dirs = nvidia_lib_dirs()
    if sys.platform == "win32":
        for d in dirs:
            if d.name == "bin":
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
    else:
        # On Linux ctranslate2 dlopens by soname; LD_LIBRARY_PATH must be set before start,
        # so preload the libraries instead.
        for d in dirs:
            for lib in sorted(d.glob("lib*.so*")):
                with contextlib.suppress(OSError):
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


@dataclass
class CudaStatus:
    device_count: int
    missing_dlls: list[str]
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.device_count > 0 and not self.missing_dlls and self.error is None


def cuda_status() -> CudaStatus:
    register_cuda_dlls()
    missing: list[str] = []
    if sys.platform == "win32":
        for dll in WINDOWS_CUDA_DLLS:
            try:
                ctypes.WinDLL(dll)
            except OSError:
                missing.append(dll)
    try:
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
    except Exception as e:  # import error / broken CUDA install
        return CudaStatus(0, missing, f"{type(e).__name__}: {e}")
    return CudaStatus(count, missing)
