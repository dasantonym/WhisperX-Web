"""Compatibility checks against the real ``whisperx`` package.

The hermetic unit suite runs against a stub module (see ``conftest.py``),
so it verifies ``whisperx_web`` behaviour against an *assumed* WhisperX API
shape — it would pass even if upstream renamed everything. These tests run
against the real ``whisperx`` distribution and assert that every API surface
used by ``whisperx_web/transcriber.py`` still exists with the arguments we
pass and the result structure we parse:

- ``whisperx.load_model(...)``
- ``whisperx.load_align_model(...)``
- ``whisperx.load_audio(...)``
- ``whisperx.align(...)``
- ``whisperx.assign_word_speakers(...)``
- ``whisperx.audio.SAMPLE_RATE``
- ``whisperx.diarize.DiarizationPipeline(model_name=, token=, device=,
  cache_dir=)`` called as ``pipeline(audio, num_speakers=, min_speakers=,
  max_speakers=)``
- ``pipeline.transcribe(audio, batch_size=, chunk_size=, language=, task=)``
- the result dict shape (``segments`` with ``start/end/text/words/speaker``,
  plus a top-level ``language`` key) — covered by the smoke test

CI runs these in the ``compat`` job of ``.github/workflows/docker.yml``
against the *pinned* whisperx version, so a breaking upstream release fails
the build before any Docker image is published. The scheduled canary build
additionally exercises the *latest* release inside the Dockerfiles.

They are skipped automatically when only the stub is installed.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

import whisperx

pytestmark = pytest.mark.skipif(
    getattr(whisperx, "__whisperx_web_stub__", False),
    reason="the whisperx stub module is installed (hermetic unit-test mode)",
)

# WhisperX's top-level names are lazy (*args, **kwargs) wrappers that forward
# to submodule functions; signature inspection must target the implementation.
# Maps a top-level name to the submodule attribute implementing it.
_IMPLEMENTATIONS = {
    "load_model": ("asr", "load_model"),
    "load_align_model": ("alignment", "load_align_model"),
    "align": ("alignment", "align"),
    "load_audio": ("audio", "load_audio"),
    "assign_word_speakers": ("diarize", "assign_word_speakers"),
}


def _inspectable(name: str):
    """Return the function behind ``whisperx.<name>``.

    Asserts the public name still exists, then unwraps the lazy
    ``(*args, **kwargs)`` forwarding wrapper if present, so keyword
    arguments can actually be checked. If upstream stops wrapping, the
    direct signature is inspected instead.
    """
    fn = getattr(whisperx, name, None)
    assert callable(fn), f"whisperx.{name} is missing or not callable"

    named = [
        p
        for p in inspect.signature(fn).parameters.values()
        if p.kind is not inspect.Parameter.VAR_KEYWORD
        and p.kind is not inspect.Parameter.VAR_POSITIONAL
    ]
    if named:
        return fn

    module_name, attr = _IMPLEMENTATIONS[name]
    module = importlib.import_module(f"whisperx.{module_name}")
    impl = getattr(module, attr, None)
    assert callable(impl), (
        f"whisperx.{name} forwards to whisperx.{module_name}.{attr}, "
        "which is missing or not callable"
    )
    return impl


def _assert_accepts(fn, keyword_args: list[str], *, min_positional: int = 0) -> None:
    """Assert ``fn`` still accepts the keyword arguments we pass to it."""
    params = inspect.signature(fn).parameters
    has_var_keyword = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    missing = [name for name in keyword_args if name not in params]
    assert has_var_keyword or not missing, (
        f"{fn.__module__}.{fn.__qualname__} no longer accepts "
        f"keyword argument(s) {missing}"
    )
    positional = [
        p
        for p in params.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    assert len(positional) >= min_positional, (
        f"{fn.__module__}.{fn.__qualname__} accepts only {len(positional)} "
        f"positional arguments, expected at least {min_positional}"
    )


def test_load_model_accepts_our_arguments():
    # whisperx_web/transcriber.py calls load_model positionally with the
    # model name and everything else by keyword.
    _assert_accepts(
        _inspectable("load_model"),
        [
            "device",
            "device_index",
            "compute_type",
            "asr_options",
            "vad_method",
            "vad_options",
            "task",
            "language",
            "download_root",
            "local_files_only",
            "threads",
            "use_auth_token",
        ],
        min_positional=1,
    )


def test_load_align_model_accepts_our_arguments():
    _assert_accepts(
        _inspectable("load_align_model"),
        ["language_code", "device", "model_name"],
    )


def test_align_accepts_our_arguments():
    # Called positionally as (segments, model, metadata, audio) plus keywords.
    _assert_accepts(
        _inspectable("align"),
        ["device", "interpolate_method", "return_char_alignments"],
        min_positional=4,
    )


def test_assign_word_speakers_accepts_our_arguments():
    # Called positionally as (diarize_segments, result).
    _assert_accepts(_inspectable("assign_word_speakers"), [], min_positional=2)


def test_load_audio_and_sample_rate():
    assert callable(_inspectable("load_audio"))
    sample_rate = whisperx.audio.SAMPLE_RATE
    assert isinstance(sample_rate, int) and sample_rate > 0


def test_diarization_pipeline_constructor_accepts_our_arguments():
    from whisperx.diarize import DiarizationPipeline

    _assert_accepts(
        DiarizationPipeline.__init__,
        ["model_name", "token", "device", "cache_dir"],
        min_positional=1,  # self
    )


def test_diarization_pipeline_call_accepts_our_arguments():
    from whisperx.diarize import DiarizationPipeline

    _assert_accepts(
        DiarizationPipeline.__call__,
        ["num_speakers", "min_speakers", "max_speakers"],
        min_positional=2,  # self + audio
    )


def test_pipeline_transcribe_accepts_our_arguments():
    # transcribe() is invoked on whatever load_model() returns; in current
    # WhisperX that is FasterWhisperPipeline. If the import path moves, that
    # itself is a change we must review.
    try:
        from whisperx.asr import FasterWhisperPipeline
    except ImportError as exc:  # pragma: no cover
        pytest.fail(
            "whisperx.asr.FasterWhisperPipeline (the pipeline returned by "
            f"load_model) is gone or moved: {exc}"
        )
    _assert_accepts(
        FasterWhisperPipeline.transcribe,
        ["batch_size", "chunk_size", "language", "task"],
        min_positional=2,  # self + audio
    )
