"""End-to-end smoke test against the real WhisperX stack.

Runs the full path exercised by ``whisperx_web/transcriber.py`` —
``load_audio`` → ``load_model`` → ``pipeline.transcribe`` →
``load_align_model`` → ``align`` — on a short public-domain speech sample
and validates the result structure the response renderers depend on
(``segments`` with ``start/end/text/words``, top-level ``language``).

This downloads the whisper ``tiny`` model (~75 MB) and the sample audio, so
it is opt-in: it runs only when ``WHISPERX_WEB_SMOKE=1`` (and never against
the stub module). Diarization is not covered — it needs an HF token and a
gated pyannote model.

Run locally::

    WHISPERX_WEB_REAL_WHISPERX=1 WHISPERX_WEB_SMOKE=1 \
        pytest tests/test_whisperx_smoke.py

or dispatch the Docker workflow with the "run smoke test" input enabled.
"""

from __future__ import annotations

import os
import urllib.request

import pytest

import whisperx

pytestmark = [
    pytest.mark.skipif(
        getattr(whisperx, "__whisperx_web_stub__", False),
        reason="the whisperx stub module is installed (hermetic unit-test mode)",
    ),
    pytest.mark.skipif(
        os.environ.get("WHISPERX_WEB_SMOKE", "") != "1",
        reason="set WHISPERX_WEB_SMOKE=1 to run the end-to-end smoke test",
    ),
]

# "And so my fellow Americans ..." — 11 seconds of clear English speech,
# used by OpenAI Whisper's own test suite (public domain).
SAMPLE_URL = "https://github.com/openai/whisper/raw/main/tests/jfk.flac"


@pytest.fixture(scope="module")
def sample_audio_path(tmp_path_factory) -> str:
    destination = tmp_path_factory.mktemp("audio") / "jfk.flac"
    try:
        urllib.request.urlretrieve(SAMPLE_URL, destination)
    except OSError as exc:
        pytest.skip(f"could not download sample audio: {exc}")
    return str(destination)


def test_end_to_end_transcription_and_alignment(sample_audio_path: str):
    audio = whisperx.load_audio(sample_audio_path)
    assert len(audio) > 0

    pipeline = whisperx.load_model(
        "tiny",
        device="cpu",
        compute_type="int8",
    )
    result = pipeline.transcribe(audio, batch_size=1, chunk_size=30)

    # Top-level result structure consumed by whisperx_web.
    assert "segments" in result
    assert isinstance(result.get("language"), str) and result["language"]
    assert isinstance(result["segments"], list) and result["segments"]
    first = result["segments"][0]
    for key in ("start", "end", "text"):
        assert key in first

    # Alignment path and word-level structure consumed by formats.py.
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device="cpu",
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device="cpu",
        interpolate_method="nearest",
        return_char_alignments=False,
    )
    assert aligned["segments"]
    assert aligned["segments"][0]["words"], "alignment produced no word timestamps"
    for key in ("start", "end", "word"):
        assert key in aligned["segments"][0]["words"][0]
