"""Application settings: environment (.env, CLIPADOR_*) + optional config.toml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

PROJECT_ROOT = Path.cwd()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLIPADOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        toml_file="config.toml",
        extra="ignore",
    )

    data_dir: Path = Path("data")

    web_host: str = "127.0.0.1"
    web_port: int = 8000

    worker_concurrency: int = 2
    worker_poll_seconds: float = 1.0
    # A running job whose heartbeat is older than this is stuck and goes back to the queue.
    job_stale_seconds: int = 90

    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    # Empty = run the yt-dlp installed in this environment (python -m yt_dlp).
    ytdlp_command: str = ""
    max_download_height: int = 1080
    cookies_from_browser: str | None = None
    cookies_file: Path | None = None

    ollama_url: str = "http://localhost:11434"

    # --- transcription (faster-whisper) ---
    whisper_model: str = "medium"  # also: large-v3-turbo, small, large-v3 (may not fit 8 GB)
    # float16 on GPU; use int8_float16 if VRAM is tight. CPU fallback always uses int8.
    whisper_compute_type: str = "float16"
    whisper_device: str = "auto"  # auto | cuda | cpu
    language: str = "pt"  # "auto" = detect
    whisper_beam_size: int = 1
    whisper_vad_filter: bool = True
    transcribe_chunk_seconds: int = 600
    transcribe_chunk_overlap_seconds: int = 4
    # Folder for downloaded Whisper models (default: <data_dir>/models).
    whisper_model_dir: Path | None = None

    # --- chat ---
    # TwitchDownloaderCLI executable (https://github.com/lay295/TwitchDownloader/releases).
    twitch_downloader_path: str = "TwitchDownloaderCLI"

    # --- local LLM (Ollama) ---
    llm_host: str = ""  # empty = ollama_url
    # ~7-8B instruct, quantized (fits 8 GB with room for context). Alternatives: llama3.1:8b,
    # qwen3:8b, gemma3:12b (tight). Compare with `clipador llm-bench`.
    llm_model: str = "qwen2.5:7b"
    llm_temperature: float = 0.2
    llm_num_ctx: int = 6144
    llm_timeout_seconds: float = 120.0
    llm_keep_alive: str = "2m"  # between calls; the model is unloaded (0) at the end of the stage
    target_clips: int = 5  # clips you want per source; enrich looks at K = 3x this
    llm_max_candidates: int = 30
    default_game: str = "GTA 6"
    prompts_dir: Path | None = None  # default: <project>/prompts
    enrich_prompt: str = "enrich_v1"  # prompts/<name>.md
    # final score = w_detect * detection + w_llm * llm (both 0-100), minus penalties
    rank_weight_detect: float = 0.5
    rank_weight_llm: float = 0.5
    rank_penalty_needs_context: float = 8.0
    rank_penalty_low_standalone: float = 6.0
    rank_low_standalone_threshold: int = 4

    # --- rendering ---
    render_encoder: str = "auto"  # auto (NVENC if it really works, else libx264) | nvenc | x264
    render_max_fps: int = 60  # output fps = source fps (VFR -> CFR), capped here
    render_style: str = "destaque_amarelo"
    zoom_amount: float = 0.12  # punch-in on the peak (0 = off)
    # False = renders wait for the single GPU lock like Whisper/Ollama (one GPU job at a time).
    render_parallel_with_gpu_jobs: bool = False
    caption_styles_path: Path | None = None  # default: <repo>/config/caption_styles.toml
    fonts_dir: Path | None = None  # default: <repo>/assets/fonts (embedded OFL font)
    # Safe zone (fractions of the 1920 px height): platform UI covers the bottom and the top.
    safe_bottom: float = 0.20
    safe_top: float = 0.10
    caption_lift: float = 0.04  # captions sit this much above the bottom safe line
    opening_text_seconds: float = 2.0
    loudness_target: float = -14.0
    loudness_true_peak: float = -1.5
    audio_fade_seconds: float = 0.08
    min_caption_event: float = 0.25
    preview_width: int = 540
    preview_height: int = 960
    platform_max_seconds: float = 60.0  # warn (don't block) above this

    # --- detection ---
    detect_preset: str = "equilibrado"  # equilibrado | foco_chat | foco_audio | foco_fala
    # Tolerance (s) when checking if a golden moment falls inside a candidate (clipador eval).
    eval_tolerance_seconds: float = 2.0

    @model_validator(mode="after")
    def _normalize(self) -> Settings:
        if not self.data_dir.is_absolute():
            self.data_dir = (PROJECT_ROOT / self.data_dir).resolve()
        if self.cookies_from_browser == "":
            self.cookies_from_browser = None
        if self.cookies_file is not None and str(self.cookies_file) == "":
            self.cookies_file = None
        self.worker_concurrency = max(1, self.worker_concurrency)
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority: explicit args > env vars > .env > config.toml
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
        )

    # --- derived paths -------------------------------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "clipador.db"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.db_path.as_posix()}"

    @property
    def vods_dir(self) -> Path:
        return self.data_dir / "vods"

    @property
    def clips_dir(self) -> Path:
        return self.data_dir / "clips"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def models_dir(self) -> Path:
        return self.whisper_model_dir or (self.data_dir / "models")

    @property
    def hype_words_path(self) -> Path:
        return self.data_dir / "hype_words.toml"

    @property
    def llm_base_url(self) -> str:
        return (self.llm_host or self.ollama_url).rstrip("/")

    @property
    def prompts_path(self) -> Path:
        if self.prompts_dir:
            return self.prompts_dir
        repo = Path(__file__).resolve().parents[2] / "prompts"  # src/clipador/config.py -> repo
        return repo if repo.is_dir() else PROJECT_ROOT / "prompts"

    @property
    def enrich_k(self) -> int:
        return max(1, min(self.llm_max_candidates, 3 * self.target_clips))

    def source_work_dir(self, source_id: int) -> Path:
        return self.work_dir / str(source_id)

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir, self.vods_dir, self.clips_dir, self.tmp_dir, self.logs_dir,
            self.work_dir,
        ):  # fmt: skip
            d.mkdir(parents=True, exist_ok=True)

    def abs_path(self, relative: str | Path) -> Path:
        """Resolve a path stored in the DB (relative to data_dir) to an absolute path."""
        return self.data_dir / Path(relative)

    def rel_path(self, absolute: Path) -> str:
        """Path relative to data_dir, always with forward slashes (portable across OSes)."""
        return absolute.resolve().relative_to(self.data_dir.resolve()).as_posix()


@lru_cache
def get_settings() -> Settings:
    return Settings()
