"""API-level unit tests for ``whisperx_web.main`` (form mapping + endpoints)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from starlette.datastructures import UploadFile

from whisperx_web.main import TranscriptionForm, _form_to_params, create_app

TEXT = "Hello world. Goodbye."
UPLOAD_PARTS = {"file": ("clip.wav", b"audio-bytes", "audio/wav")}


def make_form(**overrides) -> TranscriptionForm:
    upload = UploadFile(io.BytesIO(b"audio-bytes"), filename="clip.wav")
    return TranscriptionForm(file=upload, **overrides)


class TestFormToParams:
    def test_defaults(self):
        params = _form_to_params(make_form(), default_model="small")

        assert params.model == "small"
        assert params.task == "transcribe"
        assert params.temperature == 0.0
        assert params.align is False
        assert params.prompt is None
        assert params.hf_token is None
        # Optional decode knobs keep their dataclass defaults (never None):
        assert params.beam_size == 5
        assert params.patience == 1.0
        assert params.logprob_threshold == pytest.approx(-1.0)

    def test_model_override(self):
        params = _form_to_params(make_form(model="tiny"), default_model="small")
        assert params.model == "tiny"

    def test_word_timestamps_enable_alignment(self):
        params = _form_to_params(make_form(word_timestamps=True), default_model="small")
        assert params.align is True

    def test_prompt_falls_back_to_initial_prompt(self):
        assert _form_to_params(make_form(prompt="p"), default_model="small").prompt == "p"
        both = _form_to_params(
            make_form(prompt="p", initial_prompt="ctx"), default_model="small"
        )
        assert both.prompt == "ctx"  # initial_prompt wins when both are provided

    def test_optional_decoding_forwarded_only_when_provided(self):
        params = _form_to_params(
            make_form(beam_size=3, patience=2.0, logprob_threshold=-0.7),
            default_model="small",
        )
        assert params.beam_size == 3
        assert params.patience == 2.0
        assert params.logprob_threshold == pytest.approx(-0.7)
        assert params.length_penalty == 1.0  # untouched default

    def test_alignment_and_diarization_fields(self):
        form = make_form(
            align=True,
            align_model="custom/align",
            diarize=True,
            diarize_model="pyannote/x",
            min_speakers=2,
            max_speakers=5,
            num_speakers=3,
            hf_token="tok",
        )
        params = _form_to_params(form, default_model="small")

        assert params.align is True
        assert params.align_model == "custom/align"
        assert params.diarize is True
        assert params.diarize_model == "pyannote/x"
        assert (params.min_speakers, params.max_speakers, params.num_speakers) == (2, 5, 3)
        assert params.hf_token == "tok"


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def captured(app, monkeypatch, make_output):
    """Stub the transcriber; captures the (path, params) of each call."""
    state: dict = {}

    def fake_transcribe(audio_path, params):
        state["path"] = audio_path
        state["params"] = params
        return make_output(task=params.task)

    monkeypatch.setattr(app.state.services.transcriber, "transcribe", fake_transcribe)
    return state


class TestHealthEndpoint:
    def test_reports_settings(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "device": "cpu",
            "default_model": "small",
            "preloaded": False,
        }


class TestModelsEndpoint:
    def test_lists_known_models_with_default_flag(self, client):
        response = client.get("/v1/models")

        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "list"
        models = body["data"]
        ids = [model["id"] for model in models]
        assert ids == sorted(ids)
        assert {"tiny", "large-v3", "turbo"} <= set(ids)
        assert [model["id"] for model in models if model["default"]] == ["small"]
        for model in models:
            assert model["object"] == "model"
            assert model["owned_by"] == "whisperx-web"
            assert isinstance(model["created"], int)

    def test_custom_default_model_is_listed_and_flagged(self, settings):
        settings.default_model = "openai/whisper-large-v3"
        app = create_app(settings)
        with TestClient(app) as client:
            body = client.get("/v1/models").json()

        ids = [model["id"] for model in body["data"]]
        assert "openai/whisper-large-v3" in ids
        assert ids == sorted(ids)
        assert [model["id"] for model in body["data"] if model["default"]] == [
            "openai/whisper-large-v3"
        ]

    def test_requires_api_key_when_configured(self, settings):
        settings.api_key = "secret"
        app = create_app(settings)
        with TestClient(app) as client:
            unauthorized = client.get("/v1/models")
            authorized = client.get(
                "/v1/models", headers={"Authorization": "Bearer secret"}
            )

        assert unauthorized.status_code == 401
        assert authorized.status_code == 200


class TestTranscriptionsEndpoint:
    @staticmethod
    def post(client, **data):
        return client.post("/v1/audio/transcriptions", files=UPLOAD_PARTS, data=data)

    def test_default_json_response(self, client, captured):
        response = self.post(client)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert response.json() == {"text": TEXT}
        assert captured["params"].model == "small"
        assert captured["params"].task == "transcribe"
        # The upload is written to a .wav temp file and cleaned up afterwards:
        assert Path(captured["path"]).suffix == ".wav"
        assert not Path(captured["path"]).exists()

    def test_model_override(self, client, captured):
        self.post(client, model="tiny")
        assert captured["params"].model == "tiny"

    def test_request_fields_forwarded(self, client, captured):
        self.post(
            client,
            language="de",
            task="translate",
            temperature="0.5",
            batch_size="4",
            chunk_size="15",
            prompt="ctx",
            hf_token="tok",
            diarize="true",
        )
        params = captured["params"]

        assert params.language == "de"
        assert params.task == "translate"
        assert params.temperature == pytest.approx(0.5)
        assert params.batch_size == 4
        assert params.chunk_size == 15
        assert params.prompt == "ctx"
        assert params.hf_token == "tok"
        assert params.diarize is True

    def test_word_timestamps_force_alignment(self, client, captured):
        self.post(client, word_timestamps="true")
        assert captured["params"].align is True

    def test_text_format(self, client, captured):
        response = self.post(client, response_format="text")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert response.text == TEXT

    def test_verbose_json_format(self, client, captured):
        response = self.post(client, response_format="verbose_json")
        body = response.json()

        assert body["text"] == TEXT
        assert body["duration"] == pytest.approx(3.25)
        assert body["segments"][0]["speaker"] == "SPEAKER_00"

    def test_srt_format(self, client, captured):
        response = self.post(client, response_format="srt")

        assert response.headers["content-type"] == "application/x-subrip; charset=utf-8"
        assert response.text.startswith("1\n00:00:00,000 --> 00:00:01,500\n")


class TestHuggingFaceErrorMapping:
    @staticmethod
    def post(client):
        return client.post("/v1/audio/transcriptions", files=UPLOAD_PARTS)

    @pytest.mark.parametrize(
        "error",
        [GatedRepoError("gated repo"), RepositoryNotFoundError("missing repo")],
    )
    def test_returns_400_with_actionable_detail(self, app, monkeypatch, error):
        def failing(audio_path, params):
            raise error

        monkeypatch.setattr(app.state.services.transcriber, "transcribe", failing)
        with TestClient(app) as client:
            response = self.post(client)

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "Could not load a requested HuggingFace model" in detail
        assert "huggingface.co" in detail
        assert "hf_token" in detail

    def test_unexpected_error_returns_500(self, app, monkeypatch):
        def failing(audio_path, params):
            raise RuntimeError("boom")

        monkeypatch.setattr(app.state.services.transcriber, "transcribe", failing)
        with TestClient(app, raise_server_exceptions=False) as client:
            response = self.post(client)

        assert response.status_code == 500


class TestRequestValidation:
    @pytest.mark.parametrize(
        "data",
        [
            {"temperature": "1.5"},  # outside [0, 1]
            {"task": "summarize"},  # not a Task enum value
            {"response_format": "xml"},  # not a ResponseFormat value
            {"batch_size": "0"},  # below ge=1
            {"vad_onset": "2"},  # outside [0, 1]
            {"language": "12345678"},  # longer than max_length 7
        ],
    )
    def test_invalid_fields_return_422(self, client, data):
        response = client.post(
            "/v1/audio/transcriptions", files=UPLOAD_PARTS, data=data
        )
        assert response.status_code == 422

    def test_missing_file_returns_422(self, client):
        response = client.post("/v1/audio/transcriptions")
        assert response.status_code == 422


class TestApiKeyMiddleware:
    @pytest.fixture
    def auth_app(self, settings):
        settings.api_key = "secret"
        return create_app(settings)

    @pytest.fixture
    def auth_client(self, auth_app):
        with TestClient(auth_app) as client:
            yield client

    @staticmethod
    def post(client, headers=None):
        return client.post(
            "/v1/audio/transcriptions", files=UPLOAD_PARTS, headers=headers
        )

    def test_health_and_openapi_are_exempt(self, auth_client):
        assert auth_client.get("/health").status_code == 200
        assert auth_client.get("/openapi.json").status_code == 200

    def test_missing_key_returns_401(self, auth_client):
        response = self.post(auth_client)

        assert response.status_code == 401
        assert "Invalid or missing API key" in response.json()["detail"]

    def test_wrong_key_returns_401(self, auth_client):
        response = self.post(auth_client, headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401

    def test_correct_key_passes(self, auth_app, monkeypatch, make_output):
        monkeypatch.setattr(
            auth_app.state.services.transcriber,
            "transcribe",
            lambda path, params: make_output(task=params.task),
        )
        with TestClient(auth_app) as client:
            response = self.post(client, headers={"Authorization": "Bearer secret"})

        assert response.status_code == 200
        assert response.json() == {"text": TEXT}
