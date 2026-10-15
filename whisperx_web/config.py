"""Environment-driven configuration for the WhisperX web API.

All settings are read from environment variables once at startup, so the same
image works unchanged for both the CPU and CUDA deployment targets.
"""

from __future__ import annotations

import os

HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _get_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, "").strip()
    return value or default


def _get_token() -> str | None:
    for name in HF_TOKEN_ENV_VARS:
        value = _get_str(name)
        if value:
            return value
    return None


class Settings:
    """Runtime configuration for the API server and the WhisperX models."""

    def __init__(self) -> None:
        self.host: str = _get_str("WHISPERX_HOST", "0.0.0.0") or "0.0.0.0"
        self.port: int = _get_int("WHISPERX_PORT", 8000)
        self.device: str = _get_str("WHISPERX_DEVICE") or _auto_device()
        self.device_index: int = _get_int("WHISPERX_DEVICE_INDEX", 0)
        self.compute_type: str = _get_str("WHISPERX_COMPUTE_TYPE") or "default"
        self.default_model: str = _get_str("WHISPERX_MODEL", "small") or "small"
        self.default_batch_size: int = _get_int("WHISPERX_BATCH_SIZE", 8)
        self.threads: int = _get_int("WHISPERX_THREADS", 4)
        self.download_root: str | None = _get_str("WHISPERX_DOWNLOAD_ROOT")
        self.local_files_only: bool = _get_bool("WHISPERX_LOCAL_FILES_ONLY", False)
        self.hf_token: str | None = _get_token()
        self.api_key: str | None = _get_str("WHISPERX_API_KEY")
        self.model_cache_size: int = _get_int("WHISPERX_MODEL_CACHE_SIZE", 1)
        self.preload_model: bool = _get_bool("WHISPERX_PRELOAD_MODEL", False)
        self.default_diarize_model: str = (
            _get_str("WHISPERX_DIARIZE_MODEL", "pyannote/speaker-diarization-community-1")
            or "pyannote/speaker-diarization-community-1"
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Settings(device={self.device!r}, compute_type={self.compute_type!r}, "
            f"default_model={self.default_model!r}, batch_size={self.default_batch_size})"
        )


def _auto_device() -> str:
    """Pick ``cuda`` when a CUDA device is visible, otherwise ``cpu``."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Build ``Settings`` from the environment once and cache the instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
