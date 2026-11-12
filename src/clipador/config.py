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
