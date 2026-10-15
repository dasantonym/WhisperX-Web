"""Unit tests for ``whisperx_web.config`` (environment parsing)."""

from __future__ import annotations

import sys
import types

import pytest

from whisperx_web import config as cfg
from whisperx_web.config import (
    Settings,
    _auto_device,
    _get_bool,
    _get_int,
    _get_str,
    _get_token,
    get_settings,
)


def torch_stub(available: bool) -> types.ModuleType:
    module = types.ModuleType("torch")
    cuda = types.ModuleType("torch.cuda")
    cuda.is_available = lambda: available
    module.cuda = cuda
    return module


class TestGetBool:
    @pytest.mark.parametrize(
        "value", ["1", "true", "True", "TRUE", "yes", "on", " true "]
    )
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("PROBE_FLAG", value)
        assert _get_bool("PROBE_FLAG", False) is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "junk"])
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("PROBE_FLAG", value)
        assert _get_bool("PROBE_FLAG", True) is False

    def test_unset_uses_default(self, monkeypatch):
        monkeypatch.delenv("PROBE_FLAG", raising=False)
        assert _get_bool("PROBE_FLAG", True) is True
        assert _get_bool("PROBE_FLAG", False) is False


class TestGetInt:
    def test_parses_int(self, monkeypatch):
        monkeypatch.setenv("PROBE_INT", "12")
        assert _get_int("PROBE_INT", 1) == 12

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("PROBE_INT", "  7  ")
        assert _get_int("PROBE_INT", 1) == 7

    @pytest.mark.parametrize("value", ["", "   ", "not-a-number"])
    def test_invalid_falls_back_to_default(self, monkeypatch, value):
        monkeypatch.setenv("PROBE_INT", value)
        assert _get_int("PROBE_INT", 42) == 42

    def test_unset_uses_default(self, monkeypatch):
        monkeypatch.delenv("PROBE_INT", raising=False)
        assert _get_int("PROBE_INT", 42) == 42


class TestGetStr:
    def test_returns_stripped_value(self, monkeypatch):
        monkeypatch.setenv("PROBE_STR", "  value  ")
        assert _get_str("PROBE_STR", "fallback") == "value"

    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_falls_back_to_default(self, monkeypatch, value):
        monkeypatch.setenv("PROBE_STR", value)
        assert _get_str("PROBE_STR", "fallback") == "fallback"

    def test_unset_returns_default_or_none(self, monkeypatch):
        monkeypatch.delenv("PROBE_STR", raising=False)
        assert _get_str("PROBE_STR", "fallback") == "fallback"
        assert _get_str("PROBE_STR") is None


class TestGetToken:
    def test_prefers_hf_token(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "first")
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "second")
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "third")
        assert _get_token() == "first"

    @pytest.mark.parametrize(
        ("env_var", "token"),
        [
            ("HF_TOKEN", "a"),
            ("HUGGINGFACE_TOKEN", "b"),
            ("HUGGING_FACE_HUB_TOKEN", "c"),
        ],
    )
    def test_accepts_all_supported_names(self, monkeypatch, env_var, token):
        monkeypatch.setenv(env_var, token)
        assert _get_token() == token

    def test_none_set_returns_none(self, monkeypatch):
        for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            monkeypatch.delenv(name, raising=False)
        assert _get_token() is None


class TestSettings:
    def test_defaults(self):
        s = Settings()

        assert s.host == "0.0.0.0"
        assert s.port == 8000
        assert s.device == "cpu"  # clean_env sets WHISPERX_DEVICE=cpu
        assert s.device_index == 0
        assert s.compute_type == "default"
        assert s.default_model == "small"
        assert s.default_batch_size == 8
        assert s.threads == 4
        assert s.download_root is None
        assert s.local_files_only is False
        assert s.hf_token is None
        assert s.api_key is None
        assert s.model_cache_size == 1
        assert s.preload_model is False
        assert s.default_diarize_model == "pyannote/speaker-diarization-community-1"

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("WHISPERX_HOST", "10.0.0.1")
        monkeypatch.setenv("WHISPERX_PORT", "9999")
        monkeypatch.setenv("WHISPERX_MODEL", "large-v3")
        monkeypatch.setenv("WHISPERX_BATCH_SIZE", "16")
        monkeypatch.setenv("WHISPERX_THREADS", "2")
        monkeypatch.setenv("WHISPERX_DOWNLOAD_ROOT", " /tmp/models ")
        monkeypatch.setenv("WHISPERX_LOCAL_FILES_ONLY", "1")
        monkeypatch.setenv("WHISPERX_API_KEY", "sk-test")
        monkeypatch.setenv("WHISPERX_MODEL_CACHE_SIZE", "3")
        monkeypatch.setenv("WHISPERX_PRELOAD_MODEL", "yes")
        monkeypatch.setenv("WHISPERX_DIARIZE_MODEL", "pyannote/custom")
        monkeypatch.setenv("HF_TOKEN", "tok-123")

        s = Settings()

        assert s.host == "10.0.0.1"
        assert s.port == 9999
        assert s.default_model == "large-v3"
        assert s.default_batch_size == 16
        assert s.threads == 2
        assert s.download_root == "/tmp/models"
        assert s.local_files_only is True
        assert s.api_key == "sk-test"
        assert s.model_cache_size == 3
        assert s.preload_model is True
        assert s.default_diarize_model == "pyannote/custom"
        assert s.hf_token == "tok-123"

    def test_invalid_values_fall_back_to_defaults(self, monkeypatch):
        monkeypatch.setenv("WHISPERX_PORT", "not-a-number")
        monkeypatch.setenv("WHISPERX_HOST", "   ")
        monkeypatch.setenv("WHISPERX_MODEL", "")

        s = Settings()

        assert s.port == 8000
        assert s.host == "0.0.0.0"
        assert s.default_model == "small"

    def test_device_falls_back_to_auto_detection(self, monkeypatch):
        monkeypatch.delenv("WHISPERX_DEVICE", raising=False)
        monkeypatch.setitem(sys.modules, "torch", torch_stub(True))
        assert Settings().device == "cuda"

        monkeypatch.setitem(sys.modules, "torch", torch_stub(False))
        assert Settings().device == "cpu"

    def test_missing_torch_falls_back_to_cpu(self, monkeypatch):
        monkeypatch.delenv("WHISPERX_DEVICE", raising=False)
        monkeypatch.setitem(sys.modules, "torch", None)  # forces ImportError
        assert Settings().device == "cpu"
        assert _auto_device() == "cpu"


class TestGetSettings:
    def test_caches_first_instance(self, monkeypatch):
        monkeypatch.setattr(cfg, "_settings", None)
        first = get_settings()
        monkeypatch.setenv("WHISPERX_MODEL", "large-v3")  # must not affect cache
        assert get_settings() is first

    def test_repeated_calls_return_same_instance(self, monkeypatch):
        monkeypatch.setattr(cfg, "_settings", None)
        assert get_settings() is get_settings()
