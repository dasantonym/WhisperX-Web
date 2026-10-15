"""Unit tests for ``whisperx_web.transcriber``.

``whisperx`` is stubbed out in ``tests/conftest.py``; these tests monkeypatch
the stub callables with recording fakes to cover parameter handling, the
pipeline caches and the transcription orchestration without loading models.
"""

from __future__ import annotations

import threading

import pytest

import whisperx_web.transcriber as transcriber_module
from whisperx_web.transcriber import (
    TranscriptionOutput,
    TranscriptionParams,
    WhisperXTranscriber,
)


class FakeASRPipeline:
    """Stands in for a loaded faster-whisper pipeline."""

    def __init__(self, model: str):
        self.model = model
        self.calls: list[dict] = []
        self.result = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
        }

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return dict(self.result)


@pytest.fixture
def loaded_models(monkeypatch):
    """Replace ``whisperx.load_model`` with a recording fake."""
    calls: list[tuple[str, dict]] = []

    def fake_load_model(model, **kwargs):
        calls.append((model, kwargs))
        return FakeASRPipeline(model)

    monkeypatch.setattr(transcriber_module.whisperx, "load_model", fake_load_model)
    return calls


@pytest.fixture
def audio_stub(monkeypatch):
    """Two seconds of fake 16 kHz audio."""
    monkeypatch.setattr(
        transcriber_module.whisperx, "load_audio", lambda path: list(range(32000))
    )


class TestTemperatures:
    def test_default_schedule_mirrors_whisperx_cli(self):
        params = TranscriptionParams(model="tiny", temperature=0.0)
        assert params.temperatures() == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def test_disabled_increment_returns_single_temperature(self):
        params = TranscriptionParams(
            model="tiny", temperature=0.5, temperature_increment_on_fallback=None
        )
        assert params.temperatures() == [0.5]

    def test_zero_increment_returns_single_temperature(self):
        params = TranscriptionParams(
            model="tiny", temperature=0.5, temperature_increment_on_fallback=0.0
        )
        assert params.temperatures() == [0.5]

    def test_custom_increment(self):
        params = TranscriptionParams(
            model="tiny", temperature=0.5, temperature_increment_on_fallback=0.25
        )
        assert params.temperatures() == [0.5, 0.75, 1.0]


class TestAsrOptions:
    def test_renames_logprob_threshold_for_faster_whisper(self):
        options = TranscriptionParams(model="tiny", logprob_threshold=-0.5).asr_options()
        assert options["log_prob_threshold"] == -0.5
        assert "logprob_threshold" not in options

    def test_prompt_fields_forwarded(self):
        params = TranscriptionParams(model="tiny", prompt="style", hotwords="topic")
        options = params.asr_options()
        assert options["initial_prompt"] == "style"
        assert options["hotwords"] == "topic"
        assert options["temperatures"] == params.temperatures()

    def test_sensible_defaults(self):
        options = TranscriptionParams(model="tiny").asr_options()
        assert options["beam_size"] == 5
        assert options["best_of"] == 5
        assert options["patience"] == 1.0
        assert options["without_timestamps"] is True
        assert options["condition_on_previous_text"] is False


class TestPipelineKey:
    def test_decode_time_params_do_not_change_key(self):
        base = TranscriptionParams(model="tiny").pipeline_key()
        other = TranscriptionParams(
            model="tiny", language="de", batch_size=3, align=False, diarize=True
        ).pipeline_key()
        assert base == other

    def test_model_and_vad_options_change_key(self):
        base = TranscriptionParams(model="tiny").pipeline_key()
        assert TranscriptionParams(model="base").pipeline_key() != base
        assert TranscriptionParams(model="tiny", vad_method="silero").pipeline_key() != base
        assert TranscriptionParams(model="tiny", vad_onset=0.6).pipeline_key() != base

    def test_thresholds_are_rounded_and_key_is_hashable(self):
        key = TranscriptionParams(
            model="tiny", vad_onset=0.50001, vad_offset=0.36301
        ).pipeline_key()
        assert hash(key) == hash(
            TranscriptionParams(
                model="tiny", vad_onset=0.5, vad_offset=0.363
            ).pipeline_key()
        )


class TestAsrPipelineCache:
    def test_same_params_share_one_pipeline(self, settings, loaded_models):
        transcriber = WhisperXTranscriber(settings)
        entry1 = transcriber.get_asr_pipeline(TranscriptionParams(model="tiny"))
        entry2 = transcriber.get_asr_pipeline(
            TranscriptionParams(model="tiny", language="de", batch_size=3)
        )

        assert entry1 is entry2
        assert [model for model, _ in loaded_models] == ["tiny"]
        assert isinstance(entry1.lock, type(threading.Lock()))

    def test_different_models_load_separate_pipelines(self, settings, loaded_models):
        transcriber = WhisperXTranscriber(settings)
        transcriber.get_asr_pipeline(TranscriptionParams(model="tiny"))
        transcriber.get_asr_pipeline(TranscriptionParams(model="base"))

        assert [model for model, _ in loaded_models] == ["tiny", "base"]
        assert len(transcriber._asr_pipelines) == 2

    def test_lru_eviction_reloads_oldest_pipeline(self, settings, loaded_models):
        settings.model_cache_size = 1
        transcriber = WhisperXTranscriber(settings)
        tiny = TranscriptionParams(model="tiny")
        base = TranscriptionParams(model="base")

        transcriber.get_asr_pipeline(tiny)
        transcriber.get_asr_pipeline(base)
        transcriber.get_asr_pipeline(tiny)

        assert [model for model, _ in loaded_models] == ["tiny", "base", "tiny"]
        assert len(transcriber._asr_pipelines) == 1
        assert tiny.pipeline_key() in transcriber._asr_pipelines
        assert base.pipeline_key() not in transcriber._pipeline_locks

    def test_cache_within_capacity_is_kept(self, settings, loaded_models):
        settings.model_cache_size = 2
        transcriber = WhisperXTranscriber(settings)
        for model in ("tiny", "base", "tiny"):
            transcriber.get_asr_pipeline(TranscriptionParams(model=model))

        assert [model for model, _ in loaded_models] == ["tiny", "base"]
        assert len(transcriber._asr_pipelines) == 2

    def test_load_model_receives_settings(self, settings, loaded_models):
        settings.compute_type = "int8"
        settings.threads = 7
        settings.hf_token = "server-token"
        settings.download_root = "/models"
        settings.local_files_only = True
        transcriber = WhisperXTranscriber(settings)

        transcriber.get_asr_pipeline(
            TranscriptionParams(model="tiny", vad_onset=0.5, vad_offset=0.4)
        )

        model, kwargs = loaded_models[0]
        assert model == "tiny"
        assert kwargs["device"] == "cpu"
        assert kwargs["compute_type"] == "int8"
        assert kwargs["threads"] == 7
        assert kwargs["use_auth_token"] == "server-token"
        assert kwargs["download_root"] == "/models"
        assert kwargs["local_files_only"] is True
        assert kwargs["vad_options"] == {"vad_onset": 0.5, "vad_offset": 0.4}

    def test_preload_uses_default_model(self, settings, loaded_models):
        transcriber = WhisperXTranscriber(settings)
        transcriber.preload()
        assert [model for model, _ in loaded_models] == [settings.default_model]


class TestTranscribe:
    def test_asr_only(self, settings, loaded_models, audio_stub):
        transcriber = WhisperXTranscriber(settings)
        params = TranscriptionParams(
            model="tiny",
            align=False,
            diarize=False,
            language="de",
            task="translate",
            batch_size=4,
            chunk_size=15,
        )

        output = transcriber.transcribe("clip.wav", params)

        assert isinstance(output, TranscriptionOutput)
        assert output.duration == pytest.approx(2.0)  # 32000 samples / 16 kHz
        assert output.language == "en"
        assert output.task == "translate"
        assert output.warning is None
        entry = transcriber._asr_pipelines[params.pipeline_key()]
        assert entry.pipeline.calls == [
            {"batch_size": 4, "chunk_size": 15, "language": "de", "task": "translate"}
        ]

    def test_language_falls_back_to_params(self, settings, loaded_models, audio_stub):
        transcriber = WhisperXTranscriber(settings)
        params = TranscriptionParams(model="tiny", align=False, language="de")
        entry = transcriber.get_asr_pipeline(params)  # prime the cache
        entry.pipeline.result = {"segments": []}

        output = transcriber.transcribe("clip.wav", params)

        assert output.language == "de"


class TestTranscribeAlignment:
    def test_align_merges_word_timestamps(
        self, settings, monkeypatch, loaded_models, audio_stub
    ):
        monkeypatch.setattr(
            transcriber_module.whisperx,
            "load_align_model",
            lambda language_code, device, model_name=None: (object(), {"ctc": True}),
        )
        align_calls: list[dict] = []

        def fake_align(segments, model, metadata, audio, **kwargs):
            align_calls.append(kwargs)
            return {
                "language": "en",
                "segments": [
                    dict(segments[0], words=[{"start": 0.0, "end": 1.0, "word": "hi"}])
                ],
            }

        monkeypatch.setattr(transcriber_module.whisperx, "align", fake_align)
        transcriber = WhisperXTranscriber(settings)
        params = TranscriptionParams(
            model="tiny",
            align=True,
            interpolate_method="linear",
            return_char_alignments=True,
        )

        output = transcriber.transcribe("clip.wav", params)

        assert output.warning is None
        assert output.result["segments"][0]["words"] == [
            {"start": 0.0, "end": 1.0, "word": "hi"}
        ]
        assert align_calls == [
            {
                "device": "cpu",
                "interpolate_method": "linear",
                "return_char_alignments": True,
            }
        ]

    def test_align_failure_sets_warning_and_keeps_segments(
        self, settings, monkeypatch, loaded_models, audio_stub
    ):
        monkeypatch.setattr(
            transcriber_module.whisperx,
            "load_align_model",
            lambda *args, **kwargs: (object(), {}),
        )

        def broken_align(*args, **kwargs):
            raise ValueError("no default alignment model for lang xx")

        monkeypatch.setattr(transcriber_module.whisperx, "align", broken_align)
        transcriber = WhisperXTranscriber(settings)

        output = transcriber.transcribe(
            "clip.wav", TranscriptionParams(model="tiny", align=True)
        )

        assert output.warning == (
            "Word-level alignment skipped: no default alignment model for lang xx"
        )
        assert output.result["segments"][0]["text"] == "hi"


class TestTranscribeDiarization:
    def test_diarize_assigns_speakers(
        self, settings, monkeypatch, loaded_models, audio_stub
    ):
        created: list[dict] = []

        class FakeDiarizationPipeline:
            def __init__(self, model_name=None, token=None, device=None, cache_dir=None):
                created.append(
                    {
                        "model_name": model_name,
                        "token": token,
                        "device": device,
                        "cache_dir": cache_dir,
                    }
                )

            def __call__(
                self, audio, num_speakers=None, min_speakers=None, max_speakers=None
            ):
                return {
                    "segments": [
                        {"start": 0.0, "end": 1.0, "text": "hi", "speaker": "A"}
                    ]
                }

        monkeypatch.setattr(
            transcriber_module, "DiarizationPipeline", FakeDiarizationPipeline
        )
        assigned: list[tuple] = []

        def fake_assign(diarized, result):
            assigned.append((diarized, result))
            return {
                "language": "en",
                "segments": [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": "A"}],
            }

        monkeypatch.setattr(
            transcriber_module.whisperx, "assign_word_speakers", fake_assign
        )
        transcriber = WhisperXTranscriber(settings)
        params = TranscriptionParams(
            model="tiny",
            align=False,
            diarize=True,
            diarize_model="pyannote/custom",
            hf_token="tok",
            min_speakers=2,
        )

        output = transcriber.transcribe("clip.wav", params)

        assert created == [
            {
                "model_name": "pyannote/custom",
                "token": "tok",
                "device": "cpu",
                "cache_dir": None,
            }
        ]
        assert output.result["segments"][0]["speaker"] == "A"
        assert assigned


class TestAlignModelCache:
    def test_caches_per_language_and_model(self, settings, monkeypatch):
        calls: list[tuple] = []

        def fake_load_align_model(language_code, device, model_name=None):
            calls.append((language_code, device, model_name))
            return object(), {}

        monkeypatch.setattr(
            transcriber_module.whisperx, "load_align_model", fake_load_align_model
        )
        transcriber = WhisperXTranscriber(settings)

        first = transcriber.get_align_model("en")
        assert transcriber.get_align_model("en") is first
        transcriber.get_align_model("fr")
        assert transcriber.get_align_model("en", "custom/model") is not first

        assert calls == [
            ("en", "cpu", None),
            ("fr", "cpu", None),
            ("en", "cpu", "custom/model"),
        ]


class TestDiarizePipelineCache:
    def test_caches_per_model_and_token(self, settings, monkeypatch):
        created: list[tuple] = []

        class FakeDiarizationPipeline:
            def __init__(self, model_name=None, token=None, device=None, cache_dir=None):
                created.append((model_name, token))

        monkeypatch.setattr(
            transcriber_module, "DiarizationPipeline", FakeDiarizationPipeline
        )
        transcriber = WhisperXTranscriber(settings)

        first = transcriber.get_diarize_pipeline("pyannote/a", "tok")
        assert transcriber.get_diarize_pipeline("pyannote/a", "tok") is first
        transcriber.get_diarize_pipeline("pyannote/a", "other")

        assert created == [("pyannote/a", "tok"), ("pyannote/a", "other")]

    def test_defaults_from_settings(self, settings, monkeypatch):
        settings.hf_token = "server-token"
        created: list[tuple] = []

        class FakeDiarizationPipeline:
            def __init__(self, model_name=None, token=None, device=None, cache_dir=None):
                created.append((model_name, token))

        monkeypatch.setattr(
            transcriber_module, "DiarizationPipeline", FakeDiarizationPipeline
        )
        transcriber = WhisperXTranscriber(settings)

        transcriber.get_diarize_pipeline(None, None)

        assert created == [(settings.default_diarize_model, "server-token")]
