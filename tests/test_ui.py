"""Tests for the Gradio browser UI and its mounting rules."""

from __future__ import annotations

import pytest

from whisperx_web.main import create_app


@pytest.fixture
def client_factory(settings):
    """TestClient factory over a freshly created app (settings overridable)."""
    from fastapi.testclient import TestClient

    def _make(**overrides):
        for key, value in overrides.items():
            setattr(settings, key, value)
        return TestClient(create_app(settings))

    return _make


def test_root_serves_ui_without_api_key(client_factory):
    client = client_factory(api_key=None)
    response = client.get("/")
    assert response.status_code == 200
    assert "gradio" in response.text.lower()
    # API routes stay reachable next to the mounted UI.
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 200
    assert client.get("/docs").status_code == 200


def test_root_disabled_when_api_key_set(client_factory):
    client = client_factory(api_key="secret")
    response = client.get("/")
    assert response.status_code == 200
    assert "disabled" in response.text.lower()
    assert "whisperx_api_key" in response.text.lower()
    # The API itself still requires the key.
    assert client.get("/v1/models").status_code == 401
    assert (
        client.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code
        == 200
    )


def test_format_result_renders_every_response_format(settings, make_output):
    from whisperx_web.ui import format_result

    output = make_output()
    for fmt in ("json", "text", "verbose_json", "srt", "vtt", "tsv"):
        rendered = format_result(output, fmt)
        assert rendered.strip(), f"empty rendering for {fmt}"
    assert '"text": "Hello world. Goodbye."' in format_result(
        output, "verbose_json"
    )
    assert format_result(output, "text") == "Hello world. Goodbye."
    assert format_result(output, "srt").startswith("1\n")


def test_transcribe_audio_builds_params(
    settings, make_output, monkeypatch, tmp_path
):
    """The UI handler forwards every option into TranscriptionParams."""
    from whisperx_web import ui as ui_module
    from whisperx_web.transcriber import TranscriptionParams

    captured: dict = {}

    def fake_transcribe(self, audio_path, params):
        captured["path"] = audio_path
        captured["params"] = params
        return make_output()

    monkeypatch.setattr(
        ui_module.WhisperXTranscriber, "transcribe", fake_transcribe
    )
    transcriber = ui_module.WhisperXTranscriber(settings)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake")

    result = ui_module.transcribe_audio(
        transcriber,
        settings,
        str(audio),
        "verbose_json",
        model="",
        task="translate",
        language="de",
        prompt="context",
        temperature=0.5,
        temperature_increment_on_fallback=None,
        batch_size=None,
        chunk_size=15,
        beam_size=3,
        best_of=2,
        patience=None,
        length_penalty=None,
        repetition_penalty=None,
        no_repeat_ngram_size=None,
        compression_ratio_threshold=None,
        logprob_threshold=None,
        no_speech_threshold=None,
        condition_on_previous_text=True,
        suppress_numerals=True,
        without_timestamps=False,
        hotwords="",
        vad_method="silero",
        vad_onset=0.4,
        vad_offset=0.3,
        align=True,
        align_model="",
        interpolate_method="previous",
        return_char_alignments=True,
        diarize=True,
        diarize_model="",
        num_speakers=2,
        min_speakers=None,
        max_speakers=None,
        hf_token="hf_x",
    )

    assert '"text": "Hello world. Goodbye."' in result
    params: TranscriptionParams = captured["params"]
    assert params.model == settings.default_model  # empty -> server default
    assert params.task == "translate"
    assert params.language == "de"
    assert params.prompt == "context"
    assert params.temperature == 0.5
    assert params.temperature_increment_on_fallback is None  # cleared disables
    assert params.batch_size == settings.default_batch_size  # empty -> default
    assert params.chunk_size == 15
    assert params.beam_size == 3
    assert params.best_of == 2
    # Plain-typed knobs fall back to their dataclass defaults when cleared.
    assert params.patience == 1.0
    assert params.length_penalty == 1.0
    assert params.repetition_penalty == 1.0
    assert params.no_repeat_ngram_size == 0
    # Thresholds: cleared means disabled (None is forwarded as a real value).
    assert params.compression_ratio_threshold is None
    assert params.logprob_threshold is None
    assert params.no_speech_threshold is None
    assert params.condition_on_previous_text is True
    assert params.suppress_numerals is True
    assert params.without_timestamps is False
    assert params.hotwords is None
    assert params.vad_method == "silero"
    assert params.vad_onset == 0.4
    assert params.vad_offset == 0.3
    assert params.align is True
    assert params.align_model is None
    assert params.interpolate_method == "previous"
    assert params.return_char_alignments is True
    assert params.diarize is True
    # None is correct: the transcriber resolves the server default later.
    assert params.diarize_model is None
    assert params.num_speakers == 2
    assert params.min_speakers is None
    assert params.max_speakers is None
    assert params.hf_token == "hf_x"
    assert captured["path"] == str(audio)


def test_transcribe_audio_requires_audio(settings):
    import gradio as gr

    from whisperx_web import ui as ui_module
    from whisperx_web.transcriber import WhisperXTranscriber

    with pytest.raises(gr.Error, match="audio"):
        ui_module.transcribe_audio(
            WhisperXTranscriber(settings), settings, None, "json"
        )
