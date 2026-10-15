"""Rendering of WhisperX results into the OpenAI-compatible response formats."""

from __future__ import annotations

import json
from typing import Any

from fastapi.responses import Response

from whisperx_web.api_types import (
    MEDIA_TYPES,
    ResponseFormat,
    TranscriptionResponse,
    VerboseTranscriptionResponse,
    Word,
)
from whisperx_web.transcriber import TranscriptionOutput

_EXTRA_SEGMENT_KEYS = (
    "speaker",
    "avg_logprob",
    "no_speech_prob",
    "temperature",
    "compression_ratio",
)


def full_text(result: dict[str, Any]) -> str:
    """Concatenate segment texts into the plain transcript."""
    return "".join(seg.get("text", "") for seg in result.get("segments", [])).strip()


def _format_timestamp(seconds: float, comma: bool) -> str:
    seconds = max(0.0, seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int(round((secs - int(secs)) * 1000))
    if millis == 1000:  # guard against rounding up to a full second
        millis = 0
        secs += 1
    if secs >= 60:
        secs = 59
    sep = "," if comma else "."
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}{sep}{millis:03d}"


def render_srt(result: dict[str, Any]) -> str:
    lines = []
    for idx, seg in enumerate(result.get("segments", []), start=1):
        speaker = seg.get("speaker")
        text = seg.get("text", "").strip()
        if speaker:
            text = f"{speaker}: {text}"
        start = _format_timestamp(seg["start"], comma=True)
        end = _format_timestamp(seg["end"], comma=True)
        lines.extend([str(idx), f"{start} --> {end}", text, ""])
    return "\n".join(lines)


def render_vtt(result: dict[str, Any]) -> str:
    lines = ["WEBVTT", ""]
    for seg in result.get("segments", []):
        speaker = seg.get("speaker")
        text = seg.get("text", "").strip()
        if speaker:
            text = f"{speaker}: {text}"
        start = _format_timestamp(seg["start"], comma=False)
        end = _format_timestamp(seg["end"], comma=False)
        lines.extend([f"{start} --> {end}", text, ""])
    return "\n".join(lines)


def render_tsv(result: dict[str, Any]) -> str:
    segments = result.get("segments", [])
    has_speakers = any(seg.get("speaker") for seg in segments)
    header = (
        ["start", "end", "speaker", "text"]
        if has_speakers
        else ["start", "end", "text"]
    )
    lines = ["\t".join(header)]
    for seg in segments:
        row = [f"{seg['start']:.3f}", f"{seg['end']:.3f}"]
        if has_speakers:
            row.append(seg.get("speaker") or "")
        row.append(seg.get("text", "").strip().replace("\t", " "))
        lines.append("\t".join(row))
    return "\n".join(lines)


def _word_dicts(seg_words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        Word(
            start=w["start"],
            end=w["end"],
            word=w.get("word", ""),
            probability=w.get("probability"),
            speaker=w.get("speaker"),
        ).model_dump(exclude_none=True)
        for w in seg_words
        if w.get("start") is not None and w.get("end") is not None
    ]


def verbose_json_payload(
    result: dict[str, Any],
    duration: float,
    language: str | None,
    task: str,
    warning: str | None,
) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    for idx, seg in enumerate(result.get("segments", [])):
        seg_data: dict[str, Any] = {
            "id": idx,
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg.get("text", ""),
        }
        for key in _EXTRA_SEGMENT_KEYS:
            if seg.get(key) is not None:
                seg_data[key] = seg[key]
        seg_words = _word_dicts(seg.get("words") or [])
        if seg_words:
            seg_data["words"] = seg_words
        segments.append(seg_data)
        words.extend(seg_words)
    return VerboseTranscriptionResponse(
        task=task,
        language=language,
        duration=duration,
        text=full_text(result),
        segments=segments,  # type: ignore[arg-type]
        words=words,  # type: ignore[arg-type]
        warning=warning,
    ).model_dump(exclude_none=True)


def render_response(
    output: TranscriptionOutput, response_format: ResponseFormat
) -> Response:
    """Build the HTTP response for the requested ``response_format``."""
    result = output.result
    if response_format is ResponseFormat.JSON:
        return Response(
            content=json.dumps(
                TranscriptionResponse(text=full_text(result)).model_dump()
            ),
            media_type=MEDIA_TYPES[response_format],
        )
    if response_format is ResponseFormat.TEXT:
        return Response(
            content=full_text(result), media_type=MEDIA_TYPES[response_format]
        )
    if response_format is ResponseFormat.VERBOSE_JSON:
        return Response(
            content=json.dumps(
                verbose_json_payload(
                    result,
                    duration=output.duration,
                    language=output.language,
                    task=output.task,
                    warning=output.warning,
                )
            ),
            media_type=MEDIA_TYPES[response_format],
        )
    if response_format is ResponseFormat.SRT:
        return Response(
            content=render_srt(result), media_type=MEDIA_TYPES[response_format]
        )
    if response_format is ResponseFormat.VTT:
        return Response(
            content=render_vtt(result), media_type=MEDIA_TYPES[response_format]
        )
    if response_format is ResponseFormat.TSV:
        return Response(
            content=render_tsv(result), media_type=MEDIA_TYPES[response_format]
        )
    raise ValueError(f"Unsupported response format: {response_format}")



