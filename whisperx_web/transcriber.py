"""WhisperX model lifecycle management and the transcription pipeline.

ASR pipelines are cached in an LRU keyed by every load-time argument, so
requests that differ only in decode-time parameters (language, batch size,
alignment, ...) share one loaded model. Each pipeline has its own lock:
inference on one pipeline is serialized, different pipelines may run in
parallel worker threads (``asyncio.to_thread`` from the endpoint).
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import whisperx
from whisperx.audio import SAMPLE_RATE
from whisperx.diarize import DiarizationPipeline

from whisperx_web.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionParams:
    """All request parameters that influence transcription behaviour."""

    # Model selection and decoding
    model: str
    task: str = "transcribe"
    language: str | None = None
    prompt: str | None = None
    temperature: float = 0.0
    temperature_increment_on_fallback: float | None = 0.2
    batch_size: int = 8
    chunk_size: int = 30

    # Alignment
    align: bool = True
    align_model: str | None = None
    interpolate_method: str = "nearest"
    return_char_alignments: bool = False

    # Diarization
    diarize: bool = False
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    diarize_model: str | None = None
    hf_token: str | None = None

    # VAD
    vad_method: str = "pyannote"
    vad_onset: float = 0.5
    vad_offset: float = 0.363

    # Load-time ASR options (faster-whisper generation config)
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    compression_ratio_threshold: float | None = 2.4
    logprob_threshold: float | None = -1.0
    no_speech_threshold: float | None = 0.6
    condition_on_previous_text: bool = False
    hotwords: str | None = None
    suppress_numerals: bool = False
    without_timestamps: bool = True

    def temperatures(self) -> list[float]:
        """Fallback temperature schedule, mirroring the WhisperX CLI."""
        increment = self.temperature_increment_on_fallback
        if increment is None or increment <= 0:
            return [self.temperature]
        steps = int(round((1.0 - self.temperature) / increment))
        return [round(self.temperature + i * increment, 4) for i in range(steps + 1)]

    def asr_options(self) -> dict[str, Any]:
        """Options passed to ``load_model``; identical calls share a pipeline."""
        return {
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "patience": self.patience,
            "length_penalty": self.length_penalty,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "temperatures": self.temperatures(),
            "compression_ratio_threshold": self.compression_ratio_threshold,
            # faster-whisper's TranscriptionOptions spells this `log_prob_threshold`
            "log_prob_threshold": self.logprob_threshold,
            "no_speech_threshold": self.no_speech_threshold,
            "condition_on_previous_text": self.condition_on_previous_text,
            "initial_prompt": self.prompt,
            "hotwords": self.hotwords,
            "without_timestamps": self.without_timestamps,
            "suppress_numerals": self.suppress_numerals,
        }

    def pipeline_key(self) -> tuple:
        """Hashable identity of a loadable ASR pipeline."""
        return (
            self.model,
            json.dumps(self.asr_options(), sort_keys=True, default=str),
            self.vad_method,
            round(self.vad_onset, 4),
            round(self.vad_offset, 4),
        )


@dataclass
class TranscriptionOutput:
    """The raw WhisperX result plus metadata for response rendering."""

    result: dict[str, Any]
    duration: float
    language: str | None
    task: str
    warning: str | None = None


@dataclass
class _PipelineEntry:
    pipeline: Any
    lock: threading.Lock


class WhisperXTranscriber:
    """Loads and caches WhisperX models and runs the full transcription stack."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._asr_pipelines: "OrderedDict[tuple, _PipelineEntry]" = OrderedDict()
        self._pipeline_locks: dict[tuple, threading.Lock] = {}
        self._cache_lock = threading.Lock()
        self._load_lock = threading.Lock()  # serialize model *loading* only
        self._align_models: dict[tuple, tuple[Any, dict]] = {}
        self._align_locks: dict[tuple, threading.Lock] = {}
        self._diarize_pipelines: dict[tuple, DiarizationPipeline] = {}
        self._diarize_locks: dict[tuple, threading.Lock] = {}

    # ------------------------------------------------------------------ ASR

    def preload(self, params: TranscriptionParams | None = None) -> None:
        """Load the default ASR pipeline ahead of the first request."""
        params = params or TranscriptionParams(
            model=self.settings.default_model,
            batch_size=self.settings.default_batch_size,
        )
        self.get_asr_pipeline(params)

    def get_asr_pipeline(self, params: TranscriptionParams) -> _PipelineEntry:
        key = params.pipeline_key()
        with self._cache_lock:
            if key in self._asr_pipelines:
                self._asr_pipelines.move_to_end(key)
                return self._asr_pipelines[key]
            lock = self._pipeline_locks.setdefault(key, threading.Lock())
        with self._load_lock:
            with self._cache_lock:
                if key in self._asr_pipelines:
                    self._asr_pipelines.move_to_end(key)
                    return self._asr_pipelines[key]
            logger.info(
                "Loading WhisperX ASR model %r (device=%s, compute_type=%s)",
                params.model, self.settings.device, self.settings.compute_type,
            )
            pipeline = whisperx.load_model(
                params.model,
                device=self.settings.device,
                device_index=self.settings.device_index,
                compute_type=self.settings.compute_type,
                asr_options=params.asr_options(),
                vad_method=params.vad_method,
                vad_options={
                    "vad_onset": params.vad_onset,
                    "vad_offset": params.vad_offset,
                },
                task="transcribe",  # per-request task is passed to transcribe()
                language=None,  # auto tokenizer; language chosen per request
                download_root=self.settings.download_root,
                local_files_only=self.settings.local_files_only,
                threads=self.settings.threads,
                use_auth_token=self.settings.hf_token,
            )
            with self._cache_lock:
                self._asr_pipelines[key] = _PipelineEntry(pipeline, lock)
                while len(self._asr_pipelines) > self.settings.model_cache_size:
                    evicted, _ = self._asr_pipelines.popitem(last=False)
                    self._pipeline_locks.pop(evicted, None)
                    logger.info("Evicted cached ASR pipeline %s", evicted[0])
        return self._asr_pipelines[key]

    # -------------------------------------------------------------- alignment

    def get_align_model(self, language_code: str, model_name: str | None = None):
        key = (language_code, model_name)
        with self._cache_lock:
            if key in self._align_models:
                return self._align_models[key]
            lock = self._align_locks.setdefault(key, threading.Lock())
        with lock:
            with self._cache_lock:
                if key in self._align_models:
                    return self._align_models[key]
            logger.info(
                "Loading alignment model for language %r (model=%s)",
                language_code, model_name or "default",
            )
            model, metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self.settings.device,
                model_name=model_name,
            )
            with self._cache_lock:
                self._align_models[key] = (model, metadata)
        return self._align_models[key]

    # ------------------------------------------------------------ diarization

    def get_diarize_pipeline(
        self, model_name: str | None, token: str | None
    ) -> DiarizationPipeline:
        model_name = model_name or self.settings.default_diarize_model
        token = token or self.settings.hf_token
        key = (model_name, token or "")
        with self._cache_lock:
            if key in self._diarize_pipelines:
                return self._diarize_pipelines[key]
            lock = self._diarize_locks.setdefault(key, threading.Lock())
        with lock:
            with self._cache_lock:
                if key in self._diarize_pipelines:
                    return self._diarize_pipelines[key]
            logger.info("Loading diarization pipeline %r", model_name)
            pipeline = DiarizationPipeline(
                model_name=model_name,
                token=token,
                device=self.settings.device,
                cache_dir=self.settings.download_root,
            )
            with self._cache_lock:
                self._diarize_pipelines[key] = pipeline
        return self._diarize_pipelines[key]

    # ------------------------------------------------------------ transcription

    def transcribe(
        self, audio_path: str, params: TranscriptionParams
    ) -> TranscriptionOutput:
        """Run ASR (+ optional alignment and diarization) on an audio file."""
        audio = whisperx.load_audio(audio_path)
        duration = len(audio) / SAMPLE_RATE

        entry = self.get_asr_pipeline(params)
        with entry.lock:
            logger.info(
                "Transcribing %s (%.1fs audio, model=%s, batch_size=%s)",
                audio_path, duration, params.model, params.batch_size,
            )
            result = entry.pipeline.transcribe(
                audio,
                batch_size=params.batch_size,
                chunk_size=params.chunk_size,
                language=params.language,
                task=params.task,
            )

        language = result.get("language") or params.language
        warning: str | None = None

        if params.align:
            align_key = (language, params.align_model)
            try:
                align_model, align_metadata = self.get_align_model(*align_key)
                with self._align_locks[align_key]:
                    result = whisperx.align(
                        result["segments"],
                        align_model,
                        align_metadata,
                        audio,
                        device=self.settings.device,
                        interpolate_method=params.interpolate_method,
                        return_char_alignments=params.return_char_alignments,
                    )
            except ValueError as exc:
                # e.g. no bundled default alignment model for this language
                logger.warning("Word-level alignment skipped: %s", exc)
                warning = f"Word-level alignment skipped: {exc}"

        if params.diarize:
            diarize_key = (
                params.diarize_model or self.settings.default_diarize_model,
                params.hf_token or self.settings.hf_token or "",
            )
            diarize_pipeline = self.get_diarize_pipeline(*diarize_key)
            with self._diarize_locks[diarize_key]:
                diarize_segments = diarize_pipeline(
                    audio,
                    num_speakers=params.num_speakers,
                    min_speakers=params.min_speakers,
                    max_speakers=params.max_speakers,
                )
                result = whisperx.assign_word_speakers(diarize_segments, result)

        return TranscriptionOutput(
            result=result,
            duration=duration,
            language=language,
            task=params.task,
            warning=warning,
        )




