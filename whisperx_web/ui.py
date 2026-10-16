"""Gradio browser interface, mounted at the API root path.

Gives anyone pointing a browser at the service a transcription playground:
an audio input plus every WhisperX configuration knob exposed by
``TranscriptionParams`` (model/task/language, decoding, VAD, alignment and
diarization options), rendered through the same response formats as the API.

The UI calls the local ``WhisperXTranscriber`` directly — it is only mounted
when no API key is configured (see ``whisperx_web.main.create_app``), so a
server-side page can never be used to bypass API-key authentication.
"""

from __future__ import annotations

import os

# Gradio phones home for usage analytics by default; a self-hosted
# transcription server should not.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402

from huggingface_hub.errors import (  # noqa: E402
    GatedRepoError,
    RepositoryNotFoundError,
)

from whisperx_web import formats  # noqa: E402
from whisperx_web.api_types import ResponseFormat, WHISPER_MODEL_NAMES  # noqa: E402
from whisperx_web.config import Settings  # noqa: E402
from whisperx_web.transcriber import (  # noqa: E402
    TranscriptionParams,
    WhisperXTranscriber,
)

_FORMAT_CHOICES = [fmt.value for fmt in ResponseFormat]
_VAD_METHODS = ["pyannote", "silero"]
_INTERPOLATE_METHODS = ["nearest", "previous", "zero"]


def build_params(settings: Settings, **fields: object) -> TranscriptionParams:
    """Turn raw UI values into a validated ``TranscriptionParams``.

    Mirrors the API's optional-field semantics: ``None``/empty means "use the
    server default" for plain-typed fields (``batch_size``, decoding knobs),
    while for the threshold fields and speaker counts ``None`` is itself a
    meaningful value ("disable"/"unspecified") and is forwarded as-is.
    """
    model = fields.get("model") or settings.default_model
    language = fields.get("language") or None
    prompt = fields.get("prompt") or None
    hotwords = fields.get("hotwords") or None

    # Plain-typed numbers: an empty field falls back to the parameter default
    # (never forward None where ctranslate2 requires an int/float).
    batch_size = fields.get("batch_size") or settings.default_batch_size
    optional_decoding = {}
    for name, default in (
        ("beam_size", 5),
        ("best_of", 5),
        ("patience", 1.0),
        ("length_penalty", 1.0),
        ("repetition_penalty", 1.0),
        ("no_repeat_ngram_size", 0),
    ):
        value = fields.get(name)
        optional_decoding[name] = default if value is None else value

    # Thresholds and speaker counts: None is a valid value (disable/unset).
    thresholds = {
        name: fields.get(name)
        for name in (
            "compression_ratio_threshold",
            "logprob_threshold",
            "no_speech_threshold",
            "num_speakers",
            "min_speakers",
            "max_speakers",
        )
    }

    return TranscriptionParams(
        model=model,
        task=fields.get("task") or "transcribe",
        language=language,
        prompt=prompt,
        temperature=fields.get("temperature", 0.0),
        temperature_increment_on_fallback=fields.get(
            "temperature_increment_on_fallback"
        ),
        batch_size=batch_size,
        chunk_size=fields.get("chunk_size", 30),
        align=bool(fields.get("align")),
        align_model=fields.get("align_model") or None,
        interpolate_method=fields.get("interpolate_method") or "nearest",
        return_char_alignments=bool(fields.get("return_char_alignments")),
        diarize=bool(fields.get("diarize")),
        diarize_model=fields.get("diarize_model") or None,
        hf_token=fields.get("hf_token") or None,
        vad_method=fields.get("vad_method") or "pyannote",
        vad_onset=fields.get("vad_onset", 0.5),
        vad_offset=fields.get("vad_offset", 0.363),
        condition_on_previous_text=bool(fields.get("condition_on_previous_text")),
        hotwords=hotwords,
        suppress_numerals=bool(fields.get("suppress_numerals")),
        without_timestamps=bool(fields.get("without_timestamps")),
        **optional_decoding,
        **thresholds,
    )


def format_result(output: formats.TranscriptionOutput, response_format: str) -> str:
    """Render a transcription result exactly like the API endpoint would."""
    response = formats.render_response(output, ResponseFormat(response_format))
    body = response.body.decode("utf-8")
    if response.media_type.startswith("application/json"):
        import json

        body = json.dumps(json.loads(body), indent=2)
    return body


def transcribe_audio(
    transcriber: WhisperXTranscriber,
    settings: Settings,
    audio_path: str | None,
    response_format: str,
    **fields: object,
) -> str:
    """Full UI handler: build params, transcribe, render."""
    if not audio_path:
        raise gr.Error("Please record or upload an audio file first.")
    params = build_params(settings, **fields)
    try:
        output = transcriber.transcribe(audio_path, params)
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        raise gr.Error(
            f"Could not load a HuggingFace model: {exc}. Check the model name; "
            "for gated models, accept the terms on huggingface.co and provide "
            "an authorized token below (or via the server-side HF_TOKEN)."
        ) from exc
    return format_result(output, response_format)


def create_ui(settings: Settings, transcriber: WhisperXTranscriber) -> gr.Blocks:
    """Build the Gradio Blocks app for the service root path."""
    model_choices = sorted(WHISPER_MODEL_NAMES)
    if settings.default_model not in WHISPER_MODEL_NAMES:
        model_choices.append(settings.default_model)
        model_choices.sort()

    with gr.Blocks(title="WhisperX Web") as demo:
        gr.Markdown("# WhisperX Web")
        gr.Markdown(
            f"OpenAI-compatible WhisperX transcription server "
            f"(device: **{settings.device}**). Configure everything below, or "
            "use the [API](/docs) directly — this page drives the same engine."
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(
                    type="filepath",
                    label="Audio",
                    sources=["upload", "microphone"],
                )
                response_format = gr.Dropdown(
                    choices=_FORMAT_CHOICES,
                    value=ResponseFormat.VERBOSE_JSON.value,
                    label="Response format",
                )
                button = gr.Button("Transcribe", variant="primary")
            with gr.Column(scale=2):
                output = gr.Textbox(label="Result", lines=26, show_copy_button=True)

        with gr.Accordion("Model & decoding", open=True):
            model = gr.Dropdown(
                choices=model_choices,
                value=settings.default_model,
                label="Model",
                allow_custom_value=True,
                info="A WhisperX model name or any HuggingFace Whisper path.",
            )
            task = gr.Dropdown(
                choices=["transcribe", "translate"],
                value="transcribe",
                label="Task",
                info="'translate' transcribes non-English audio into English.",
            )
            language = gr.Textbox(
                label="Language",
                placeholder="auto-detect",
                info="ISO 639-1/639-2 code (e.g. 'en'). Empty = auto-detect.",
            )
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="",
                info="Optional context to guide the model's style.",
            )
            temperature = gr.Slider(
                0.0, 1.0, value=0.0, step=0.05, label="Temperature"
            )
            temperature_increment_on_fallback = gr.Number(
                value=0.2,
                minimum=0.0,
                maximum=1.0,
                label="Temperature increment on fallback",
                info="Empty or 0 disables the fallback temperature schedule.",
            )
            batch_size = gr.Number(
                value=settings.default_batch_size,
                minimum=1,
                precision=0,
                label="Batch size",
            )
            chunk_size = gr.Number(
                value=30,
                minimum=1,
                precision=0,
                label="Chunk size (seconds)",
                info="Duration of audio chunks fed to VAD/ASR.",
            )
            with gr.Row():
                beam_size = gr.Number(
                    value=5, minimum=1, precision=0, label="Beam size"
                )
                best_of = gr.Number(
                    value=5, minimum=1, precision=0, label="Best of"
                )
                patience = gr.Number(value=1.0, label="Patience")
                length_penalty = gr.Number(value=1.0, label="Length penalty")
                repetition_penalty = gr.Number(
                    value=1.0, label="Repetition penalty"
                )
                no_repeat_ngram_size = gr.Number(
                    value=0, minimum=0, precision=0, label="No-repeat n-gram size"
                )
            with gr.Row():
                compression_ratio_threshold = gr.Number(
                    value=2.4,
                    label="Compression ratio threshold",
                    info="Clear to disable.",
                )
                logprob_threshold = gr.Number(
                    value=-1.0, label="Log-prob threshold", info="Clear to disable."
                )
                no_speech_threshold = gr.Number(
                    value=0.6, label="No-speech threshold", info="Clear to disable."
                )
            with gr.Row():
                condition_on_previous_text = gr.Checkbox(
                    value=False,
                    label="Condition on previous text",
                    info="Feed previous output as prompt for the next window.",
                )
                suppress_numerals = gr.Checkbox(
                    value=False, label="Suppress numerals"
                )
                without_timestamps = gr.Checkbox(
                    value=True, label="Without timestamps"
                )
            hotwords = gr.Textbox(
                label="Hotwords", info="Optional provided vocabulary hint."
            )

        with gr.Accordion("Voice activity detection (VAD)", open=False):
            vad_method = gr.Dropdown(
                choices=_VAD_METHODS,
                value="pyannote",
                label="VAD method",
            )
            vad_onset = gr.Slider(
                0.0, 1.0, value=0.5, step=0.01, label="VAD onset"
            )
            vad_offset = gr.Slider(
                0.0, 1.0, value=0.363, step=0.01, label="VAD offset"
            )

        with gr.Accordion("Alignment (word timestamps)", open=False):
            align = gr.Checkbox(
                value=False,
                label="Align word timestamps",
                info="Runs a wav2vec2 alignment pass for word-level timing.",
            )
            align_model = gr.Textbox(
                label="Alignment model",
                placeholder="default for the detected language",
                info="Custom wav2vec2 HuggingFace path.",
            )
            interpolate_method = gr.Dropdown(
                choices=_INTERPOLATE_METHODS,
                value="nearest",
                label="Missing-timestamp interpolation",
            )
            return_char_alignments = gr.Checkbox(
                value=False, label="Include character-level alignments"
            )

        with gr.Accordion("Diarization (speaker labels)", open=False):
            diarize = gr.Checkbox(
                value=False, label="Run diarization"
            )
            diarize_model = gr.Textbox(
                label="Diarization model",
                placeholder=settings.default_diarize_model,
                info="Pyannote diarization model name.",
            )
            with gr.Row():
                num_speakers = gr.Number(
                    label="Number of speakers",
                    precision=0,
                    minimum=1,
                    info="Exact count (overrides min/max). Clear if unknown.",
                )
                min_speakers = gr.Number(
                    label="Min speakers", precision=0, minimum=1
                )
                max_speakers = gr.Number(
                    label="Max speakers", precision=0, minimum=1
                )
            hf_token = gr.Textbox(
                label="HuggingFace token",
                placeholder="",
                type="password",
                info="Needed once to fetch gated pyannote models.",
            )

        inputs = [
            audio, response_format,
            model, task, language, prompt, temperature,
            temperature_increment_on_fallback, batch_size, chunk_size,
            beam_size, best_of, patience, length_penalty, repetition_penalty,
            no_repeat_ngram_size, compression_ratio_threshold, logprob_threshold,
            no_speech_threshold, condition_on_previous_text, suppress_numerals,
            without_timestamps, hotwords,
            vad_method, vad_onset, vad_offset,
            align, align_model, interpolate_method, return_char_alignments,
            diarize, diarize_model, num_speakers, min_speakers, max_speakers,
            hf_token,
        ]

        def handler(audio_path, fmt, *values):
            names = (
                "model", "task", "language", "prompt", "temperature",
                "temperature_increment_on_fallback", "batch_size", "chunk_size",
                "beam_size", "best_of", "patience", "length_penalty",
                "repetition_penalty", "no_repeat_ngram_size",
                "compression_ratio_threshold", "logprob_threshold",
                "no_speech_threshold", "condition_on_previous_text",
                "suppress_numerals", "without_timestamps", "hotwords",
                "vad_method", "vad_onset", "vad_offset",
                "align", "align_model", "interpolate_method",
                "return_char_alignments",
                "diarize", "diarize_model", "num_speakers", "min_speakers",
                "max_speakers", "hf_token",
            )
            fields = dict(zip(names, values))
            return transcribe_audio(
                transcriber, settings, audio_path, fmt, **fields
            )

        button.click(handler, inputs=inputs, outputs=output)

    return demo
