"""OpenAI-compatible transcription API built on WhisperX."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from pydantic import BaseModel

from whisperx_web import formats
from whisperx_web.api_types import (
    MEDIA_TYPES,
    HealthResponse,
    ResponseFormat,
    Task,
)
from whisperx_web.config import Settings, get_settings
from whisperx_web.transcriber import (
    TranscriptionOutput,
    TranscriptionParams,
    WhisperXTranscriber,
)

__all__ = ["app", "create_app", "cli"]

logger = logging.getLogger("whisperx_web")

_MODEL_NAMES = {
    "tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en",
    "medium", "large-v1", "large-v2", "large-v3", "large", "large-v3-turbo",
    "turbo", "distil-large-v2", "distil-large-v3", "distil-medium.en",
    "distil-small.en", "large-v2-melotto", "belarge", "large-v3-turbo-q8",
}


class TranscriptionForm(BaseModel):
    """All OpenAI transcription fields plus WhisperX-specific knobs."""

    # NOTE: the file must be declared *inside* the form model. FastAPI
    # (>=0.115) only flattens a Pydantic model into multipart form fields when
    # it is the sole multipart parameter; mixing an UploadFile parameter with
    # a form model silently degrades the model to a single embedded field.
    file: Annotated[
        UploadFile,
        File(description="Audio file to transcribe."),
    ]

    # -- OpenAI-compatible fields -------------------------------------------
    model: Annotated[
        str | None,
        Form(
            description=(
                "ID of the model to use: a WhisperX model name (one of "
                f"{', '.join(sorted(_MODEL_NAMES))}) or any huggingface "
                "whisper model path (e.g. openai/whisper-large-v3). "
                "Defaults to the server-configured model."
            ),
        ),
    ] = None
    language: Annotated[
        str | None,
        Form(
            max_length=7,
            description="ISO 639-1/639-2 language code; auto-detected when omitted.",
        ),
    ] = None
    prompt: Annotated[
        str | None,
        Form(description="Optional context to guide the model's style."),
    ] = None
    response_format: Annotated[
        ResponseFormat,
        Form(description="Output format of the transcription."),
    ] = ResponseFormat.JSON
    temperature: Annotated[
        float,
        Form(ge=0.0, le=1.0, description="Sampling temperature."),
    ] = 0.0
    task: Annotated[
        Task,
        Form(
            description=(
                "'transcribe' or 'translate' (non-English audio to English)."
            ),
        ),
    ] = Task.TRANSCRIBE

    # -- WhisperX decoding parameters ---------------------------------------
    batch_size: Annotated[
        int | None, Form(ge=1, description="Batch size for batched inference.")
    ] = None
    chunk_size: Annotated[
        int,
        Form(ge=1, description="Duration of audio chunks fed to VAD/ASR (seconds)."),
    ] = 30
    beam_size: Annotated[
        int | None, Form(ge=1, description="Beam size; enables beam search decoding.")
    ] = None
    patience: Annotated[
        float | None,
        Form(gt=0.0, description="Beam search patience (beam width multiplier)."),
    ] = None
    length_penalty: Annotated[
        float | None,
        Form(gt=0.0, description="Exponential length penalty for beam search."),
    ] = None
    compression_ratio_threshold: Annotated[
        float | None,
        Form(description="GZIP compression ratio above which decoding fails."),
    ] = None
    logprob_threshold: Annotated[
        float | None,
        Form(description="Average log-probability below which decoding fails."),
    ] = None
    no_speech_threshold: Annotated[
        float | None,
        Form(description="No-speech probability above which a segment is silent."),
    ] = None
    condition_on_previous_text: Annotated[
        bool | None,
        Form(description="Feed previous output as prompt for the next window."),
    ] = None
    initial_prompt: Annotated[
        str | None,
        Form(description="Optional prompt for the first window."),
    ] = None
    word_timestamps: Annotated[
        bool,
        Form(
            description=(
                "Extract word-level timestamps. Requires alignment; "
                "automatically enables it."
            ),
        ),
    ] = False

    # -- WhisperX VAD parameters --------------------------------------------
    vad_onset: Annotated[
        float, Form(ge=0.0, le=1.0, description="VAD onset threshold.")
    ] = 0.500
    vad_offset: Annotated[
        float, Form(ge=0.0, le=1.0, description="VAD offset threshold.")
    ] = 0.363
    vad_method: Annotated[
        str, Form(description="VAD method ('pyannote' or 'silero').")
    ] = "pyannote"

    # -- WhisperX alignment parameters --------------------------------------
    align: Annotated[
        bool | None,
        Form(
            description=(
                "Perform word-level alignment. Defaults to true when "
                "word_timestamps is requested."
            ),
        ),
    ] = None
    align_model: Annotated[
        str | None,
        Form(description="Custom wav2vec2 alignment model (huggingface path)."),
    ] = None
    interpolate_method: Annotated[
        str, Form(description="Method for missing word timestamps ('nearest').")
    ] = "nearest"
    return_char_alignments: Annotated[
        bool, Form(description="Include character-level alignments in output.")
    ] = False

    # -- WhisperX diarization parameters ------------------------------------
    diarize: Annotated[
        bool, Form(description="Run speaker diarization and label segments/words.")
    ] = False
    diarize_model: Annotated[
        str | None,
        Form(
            description=(
                "Pyannote diarization model name "
                "(default: pyannote/speaker-diarization-community-1)."
            ),
        ),
    ] = None
    min_speakers: Annotated[
        int | None, Form(ge=1, description="Minimum number of speakers in the audio.")
    ] = None
    max_speakers: Annotated[
        int | None, Form(ge=1, description="Maximum number of speakers in the audio.")
    ] = None
    num_speakers: Annotated[
        int | None, Form(ge=1, description="Exact number of speakers in the audio.")
    ] = None
    hf_token: Annotated[
        str | None,
        Form(
            description=(
                "HuggingFace token used to fetch models when none was "
                "configured server-side."
            ),
        ),
    ] = None


def _form_to_params(
    form: TranscriptionForm, default_model: str
) -> TranscriptionParams:
    """Convert validated form fields into a WhisperX parameter object."""
    align = form.align
    if align is None:
        align = form.word_timestamps  # word timestamps require alignment
    prompt = form.initial_prompt if form.initial_prompt is not None else form.prompt
    # Optional decoding knobs: a form value of None means "not provided", so
    # forward only the supplied ones and let the dataclass defaults stand
    # (forwarding None would override e.g. beam_size=5 with None and crash
    # ctranslate2's generate(), which requires ints/floats).
    optional_decoding = {
        "batch_size": form.batch_size,
        "beam_size": form.beam_size,
        "patience": form.patience,
        "length_penalty": form.length_penalty,
        "compression_ratio_threshold": form.compression_ratio_threshold,
        "logprob_threshold": form.logprob_threshold,
        "no_speech_threshold": form.no_speech_threshold,
        "condition_on_previous_text": form.condition_on_previous_text,
    }
    return TranscriptionParams(
        model=form.model or default_model,
        language=form.language,
        task=form.task,
        temperature=form.temperature,
        chunk_size=form.chunk_size,
        prompt=prompt,
        align=align,
        align_model=form.align_model,
        interpolate_method=form.interpolate_method,
        return_char_alignments=form.return_char_alignments,
        diarize=form.diarize,
        diarize_model=form.diarize_model,
        min_speakers=form.min_speakers,
        max_speakers=form.max_speakers,
        num_speakers=form.num_speakers,
        hf_token=form.hf_token,
        **{
            key: value
            for key, value in optional_decoding.items()
            if value is not None
        },
    )


class AppState:
    """Holds shared services attached to ``app.state``."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.transcriber = WhisperXTranscriber(settings)
        self.preload_done = False


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        services: AppState = app.state.services
        if settings.preload_model:
            logger.info(
                "Preloading default ASR model %r at startup", settings.default_model
            )
            await asyncio.to_thread(services.transcriber.preload)
            services.preload_done = True
        yield

    app = FastAPI(
        title="WhisperX Web",
        version="0.1.0",
        description=(
            "OpenAI-compatible transcription API backed by WhisperX "
            "(batched ASR, word timestamps, alignment, diarization)."
        ),
        lifespan=lifespan,
    )
    app.state.services = AppState(settings)

    if settings.api_key:

        @app.middleware("http")
        async def check_api_key(
            request: Request, call_next: Callable[..., Awaitable[Response]]
        ) -> Response:
            if request.url.path in ("/health", "/openapi.json", "/docs"):
                return await call_next(request)
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] != settings.api_key:
                # User middleware runs above the exception handlers, so a
                # raised HTTPException would surface as a 500, not a 401.
                return JSONResponse(
                    status_code=401,
                    content={
                        "detail": "Invalid or missing API key "
                        "(Authorization: Bearer ...)."
                    },
                )
            return await call_next(request)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        services: AppState = app.state.services
        return HealthResponse(
            status="ok",
            device=settings.device,
            default_model=settings.default_model,
            preloaded=services.preload_done,
        )

    @app.post(
        "/v1/audio/transcriptions",
        response_class=Response,
        # Document every media type the endpoint may return for a 200
        # response (keys must be HTTP status codes, not format enums).
        responses={
            200: {"content": {media_type: {}} for media_type in MEDIA_TYPES.values()}
        },
    )
    async def transcriptions(
        form: Annotated[TranscriptionForm, Form()],
    ) -> Response:
        services: AppState = app.state.services
        file = form.file
        suffix = Path(file.filename or "audio").suffix or ".bin"
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(await file.read())
        finally:
            await file.close()

        params = _form_to_params(form, default_model=services.settings.default_model)
        started = time.monotonic()
        try:
            output: TranscriptionOutput = await asyncio.to_thread(
                services.transcriber.transcribe, tmp_path, params
            )
        except (GatedRepoError, RepositoryNotFoundError) as exc:
            # Model downloads (ASR / alignment / diarization) that hit an
            # unknown or unauthorized HuggingFace repo are client-fixable:
            # wrong model name, missing token, or unaccepted model terms.
            logger.warning("HuggingFace model access failed: %s", exc)
            return JSONResponse(
                status_code=400,
                content={
                    "detail": (
                        f"Could not load a requested HuggingFace model: {exc} "
                        "Check the model name; for gated models, accept the "
                        "terms on huggingface.co and provide an authorized "
                        "token (server-side HF_TOKEN env var or per-request "
                        "hf_token form field)."
                    )
                },
            )
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove temporary file %s", tmp_path)
        logger.info(
            "Transcribed %s in %.2fs", file.filename, time.monotonic() - started
        )
        return formats.render_response(output, form.response_format)

    return app


app = create_app()


def cli() -> None:
    """Console-script entry point: run the uvicorn server."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="whisperx-web",
        description="OpenAI-compatible WhisperX transcription server.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes."
    )
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "whisperx_web.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    cli()



