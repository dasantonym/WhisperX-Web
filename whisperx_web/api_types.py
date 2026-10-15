"""Request/response types for the OpenAI-compatible transcription API."""

from __future__ import annotations

from enum import Enum

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
