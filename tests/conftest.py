"""Shared fixtures for the WhisperX-Web unit test suite.

The suite is hermetic: ``whisperx`` (and its heavyweight ``torch`` stack) is
replaced by a stub *before* any ``whisperx_web`` module is imported, so unit
tests never load models, touch CUDA or decode audio. Tests that exercise
model caching and the transcription pipeline monkeypatch the stub callables
with recording fakes (see ``tests/test_transcriber.py``).

Set ``WHISPERX_WEB_REAL_WHISPERX=1`` to skip the stub and run against the
real ``whisperx`` distribution instead. The CI ``compat`` job uses this mode
for ``tests/test_whisperx_compat.py`` (and the optional smoke test), which
verify that the WhisperX API surface ``whisperx_web`` depends on still
matches reality.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Settings probes torch for CUDA only when WHISPERX_DEVICE is unset; give it a
# deterministic answer so the real torch is never imported during tests.
os.environ.setdefault("WHISPERX_DEVICE", "cpu")


def _install_whisperx_stub() -> None:
    """Register a minimal ``whisperx`` package stub in ``sys.modules``."""
    whisperx = types.ModuleType("whisperx")
    whisperx.__whisperx_web_stub__ = True
    whisperx.__path__ = []  # mark as a package so submodule imports resolve

    audio = types.ModuleType("whisperx.audio")
    audio.SAMPLE_RATE = 16_000

    diarize = types.ModuleType("whisperx.diarize")

    class DiarizationPipeline:  # pragma: no cover - replaced by fakes in tests
        """Placeholder; tests monkeypatch it with a recording fake."""

        def __init__(self, model_name=None, token=None, device=None, cache_dir=None):
            self.model_name = model_name
            self.token = token
            self.device = device
            self.cache_dir = cache_dir

        def __call__(self, audio, **kwargs):
            raise AssertionError("DiarizationPipeline stub must be replaced in tests")

    diarize.DiarizationPipeline = DiarizationPipeline

    def _must_be_mocked(*args, **kwargs):  # pragma: no cover
        raise AssertionError("whisperx callables must be monkeypatched in tests")

    for name in (
        "load_model",
        "load_audio",
        "load_align_model",
        "align",
        "assign_word_speakers",
    ):
        setattr(whisperx, name, _must_be_mocked)

    whisperx.audio = audio
    whisperx.diarize = diarize
    sys.modules["whisperx"] = whisperx
    sys.modules["whisperx.audio"] = audio
    sys.modules["whisperx.diarize"] = diarize


# The stub must be installed before any whisperx_web import. Skip it in
# real-whisperx mode (used by the CI compat job); see the module docstring.
if os.environ.get("WHISPERX_WEB_REAL_WHISPERX", "") != "1":
    _install_whisperx_stub()

import pytest  # noqa: E402

from whisperx_web.config import HF_TOKEN_ENV_VARS  # noqa: E402
from whisperx_web.transcriber import TranscriptionOutput  # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove every WhisperX/HF environment variable for deterministic Settings."""
    for name in list(os.environ):
        if name.startswith("WHISPERX_") or name in HF_TOKEN_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WHISPERX_DEVICE", "cpu")
    yield


@pytest.fixture
def settings():
    """Deterministic ``Settings`` (attribute overrides rather than env vars)."""
    from whisperx_web.config import Settings

    s = Settings()
    s.host = "127.0.0.1"
    s.port = 8765
    s.device = "cpu"
    s.device_index = 0
    s.compute_type = "default"
    s.default_model = "small"
    s.default_batch_size = 8
    s.threads = 4
    s.download_root = None
    s.local_files_only = False
    s.hf_token = None
    s.api_key = None
    s.model_cache_size = 2
    s.preload_model = False
    s.default_diarize_model = "pyannote/speaker-diarization-community-1"
    return s


@pytest.fixture
def make_output():
    """Factory for a representative ``TranscriptionOutput`` (aligned + diarized)."""

    def _make(**overrides) -> TranscriptionOutput:
        result = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.5,
                    "text": "Hello world.",
                    "speaker": "SPEAKER_00",
                    "avg_logprob": -0.25,
                    "no_speech_prob": 0.02,
                    "words": [
                        {
                            "start": 0.0,
                            "end": 0.5,
                            "word": "Hello",
                            "probability": 0.98,
                            "speaker": "SPEAKER_00",
                        },
                        {
                            "start": 0.6,
                            "end": 1.5,
                            "word": "world.",
                            "probability": 0.9,
                            "speaker": "SPEAKER_00",
                        },
                    ],
                },
                {"start": 2.0, "end": 3.25, "text": " Goodbye."},
            ],
        }
        kwargs: dict = {
            "result": result,
            "duration": 3.25,
            "language": "en",
            "task": "transcribe",
            "warning": None,
        }
        kwargs.update(overrides)
        return TranscriptionOutput(**kwargs)

    return _make
