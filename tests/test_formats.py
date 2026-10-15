"""Unit tests for ``whisperx_web.formats`` (response rendering)."""

from __future__ import annotations

import json

import pytest

from whisperx_web.api_types import MEDIA_TYPES, ResponseFormat
from whisperx_web.formats import (
    _format_timestamp,
    full_text,
    render_response,
    render_srt,
    render_tsv,
    render_vtt,
    verbose_json_payload,
)

TEXT = "Hello world. Goodbye."


@pytest.fixture
def output(make_output):
    return make_output()


class TestFullText:
    def test_joins_segment_texts_and_strips(self, output):
        assert full_text(output.result) == TEXT

    def test_empty_results(self):
        assert full_text({}) == ""
        assert full_text({"segments": []}) == ""


class TestFormatTimestamp:
    @pytest.mark.parametrize(
        ("seconds", "comma", "expected"),
        [
            (0.0, True, "00:00:00,000"),
            (1.5, True, "00:00:01,500"),
            (61.25, False, "00:01:01.250"),
            (3600.0, True, "01:00:00,000"),
            (-3.0, True, "00:00:00,000"),  # negative values are clamped
            (0.99995, True, "00:00:01,000"),  # millis rounding carries a second
            (59.99995, True, "00:00:59,000"),  # ...and never spills into minute
        ],
    )
    def test_formatting(self, seconds, comma, expected):
        assert _format_timestamp(seconds, comma) == expected


class TestRenderSrt:
    def test_numbered_blocks_with_speaker_prefix(self, output):
        expected = (
            "1\n"
            "00:00:00,000 --> 00:00:01,500\n"
            "SPEAKER_00: Hello world.\n"
            "\n"
            "2\n"
            "00:00:02,000 --> 00:00:03,250\n"
            "Goodbye.\n"
        )
        assert render_srt(output.result) == expected


class TestRenderVtt:
    def test_header_and_dot_timestamp_separator(self, output):
        content = render_vtt(output.result)
        assert content.startswith("WEBVTT\n\n")
        assert "00:00:00.000 --> 00:00:01.500\nSPEAKER_00: Hello world." in content
        assert "00:00:02.000 --> 00:00:03.250\nGoodbye." in content


class TestRenderTsv:
    def test_without_speakers(self):
        result = {"segments": [{"start": 0.0, "end": 1.5, "text": "a\tb"}]}
        assert render_tsv(result) == "start\tend\ttext\n0.000\t1.500\ta b"

    def test_with_speakers(self, output):
        expected = (
            "start\tend\tspeaker\ttext\n"
            "0.000\t1.500\tSPEAKER_00\tHello world.\n"
            "2.000\t3.250\t\tGoodbye."
        )
        assert render_tsv(output.result) == expected


class TestVerboseJsonPayload:
    def test_payload_shape(self, output):
        payload = verbose_json_payload(
            output.result, duration=3.25, language="en", task="transcribe", warning=None
        )

        assert payload["text"] == TEXT
        assert payload["task"] == "transcribe"
        assert payload["language"] == "en"
        assert payload["duration"] == pytest.approx(3.25)
        assert "warning" not in payload  # exclude_none drops the empty warning

        first, second = payload["segments"]
        assert first["id"] == 0
        assert second["id"] == 1
        assert first["speaker"] == "SPEAKER_00"
        assert first["avg_logprob"] == pytest.approx(-0.25)
        assert first["no_speech_prob"] == pytest.approx(0.02)
        assert "speaker" not in second
        assert "words" not in second
        assert first["words"] == [
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
        ]
        assert payload["words"] == first["words"]

    def test_words_with_missing_timestamps_are_dropped(self):
        result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "x",
                    "words": [
                        {"start": None, "end": 1.0, "word": "bad"},
                        {"start": 0.0, "end": None, "word": "also-bad"},
                        {"start": 0.0, "end": 0.5, "word": "ok"},
                    ],
                }
            ]
        }
        payload = verbose_json_payload(
            result, duration=1.0, language="en", task="transcribe", warning="partial"
        )

        assert payload["segments"][0]["words"] == [{"start": 0.0, "end": 0.5, "word": "ok"}]
        assert payload["warning"] == "partial"


class TestRenderResponse:
    def test_json(self, output):
        response = render_response(output, ResponseFormat.JSON)
        assert response.headers["content-type"] == MEDIA_TYPES[ResponseFormat.JSON]
        assert json.loads(response.body) == {"text": TEXT}

    def test_text(self, output):
        response = render_response(output, ResponseFormat.TEXT)
        assert response.headers["content-type"] == MEDIA_TYPES[ResponseFormat.TEXT]
        assert response.body.decode() == TEXT

    def test_verbose_json(self, output):
        response = render_response(output, ResponseFormat.VERBOSE_JSON)
        assert response.headers["content-type"] == MEDIA_TYPES[ResponseFormat.VERBOSE_JSON]
        body = json.loads(response.body)
        assert body["text"] == TEXT
        assert body["duration"] == pytest.approx(3.25)
        assert body["segments"][0]["id"] == 0

    @pytest.mark.parametrize(
        ("response_format", "prefix"),
        [
            (ResponseFormat.SRT, "1\n"),
            (ResponseFormat.VTT, "WEBVTT\n"),
            (ResponseFormat.TSV, "start\tend\tspeaker\ttext\n"),
        ],
    )
    def test_subtitle_formats(self, output, response_format, prefix):
        response = render_response(output, response_format)
        assert response.headers["content-type"] == MEDIA_TYPES[response_format]
        assert response.body.decode().startswith(prefix)
