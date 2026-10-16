"""Request/response types for the OpenAI-compatible transcription API."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ResponseFormat(str, Enum):
    """Output formats understood by ``response_format``."""

    JSON = "json"
    TEXT = "text"
    VERBOSE_JSON = "verbose_json"
    SRT = "srt"
    VTT = "vtt"
    TSV = "tsv"


class Task(str, Enum):
    """Whisper task selection."""

    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


MEDIA_TYPES: dict[ResponseFormat, str] = {
    ResponseFormat.JSON: "application/json",
    ResponseFormat.TEXT: "text/plain; charset=utf-8",
    ResponseFormat.VERBOSE_JSON: "application/json",
    ResponseFormat.SRT: "application/x-subrip; charset=utf-8",
    ResponseFormat.VTT: "text/vtt; charset=utf-8",
    ResponseFormat.TSV: "text/tab-separated-values; charset=utf-8",
}


class TranscriptionResponse(BaseModel):
    """Payload of the plain ``json`` response format (OpenAI-compatible)."""

    text: str


class Word(BaseModel):
    """A single word-level timestamp (present once alignment has run)."""

    model_config = ConfigDict(extra="allow")

    start: float
    end: float
    word: str
    probability: float | None = None
    speaker: str | None = None


class Segment(BaseModel):
    """A transcript segment; extra keys are passed through when available."""

    model_config = ConfigDict(extra="allow")

    id: int
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[Word] | None = None


class VerboseTranscriptionResponse(BaseModel):
    """Payload of the ``verbose_json`` response format (OpenAI-compatible)."""

    task: str
    language: str | None = None
    duration: float
    text: str
    segments: list[Segment] = []
    words: list[Word] = []
    warning: str | None = None


class HealthResponse(BaseModel):
    """Payload of the ``/health`` endpoint."""

    status: str
    device: str
    default_model: str
    preloaded: bool


# WhisperX model names selectable through the ``model`` field (and the browser
# UI dropdown). Lives here so both the API and the UI share one definition.
WHISPER_MODEL_NAMES = {
    "tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en",
    "medium", "large-v1", "large-v2", "large-v3", "large", "large-v3-turbo",
    "turbo", "distil-large-v2", "distil-large-v3", "distil-medium.en",
    "distil-small.en", "large-v2-melotto", "belarge", "large-v3-turbo-q8",
}


MODEL_OWNER = "whisperx-web"


class ModelObject(BaseModel):
    """A single selectable model (OpenAI-compatible ``/v1/models`` entry).

    ``default`` is a WhisperX-Web extension: it marks the model used when a
    transcription request omits the ``model`` field.
    """

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = MODEL_OWNER
    default: bool = False


class ModelListResponse(BaseModel):
    """Payload of the ``/v1/models`` endpoint (OpenAI-compatible)."""

    object: Literal["list"] = "list"
    data: list[ModelObject]
