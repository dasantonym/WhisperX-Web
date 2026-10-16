# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0/).

## [Unreleased]

### Fixed

- The CUDA Docker image started with `exec: whisperx-web: not found` because
  its Ubuntu 22.04 base ships a Debian-patched pip 22.0.2 that checks
  build-system requirements only by distribution name and ignores version
  constraints; the project's legacy fallback therefore built with the system
  setuptools (< 61), which predates PEP 621 support, and silently produced an
  empty `UNKNOWN-0.0.0` package without the `whisperx-web` console script (or
  the `whisperx_web` package at all). `Dockerfile.cuda` now upgrades pip
  before installing, and `pyproject.toml` declares the build system
  explicitly (`setuptools>=64` + `wheel`) so every pip version installs a
  PEP 621-capable backend.
- The CUDA image logged a `torchcodec is not installed correctly` warning at
  startup from pyannote.audio (WhisperX's diarization backend). torchcodec
  ships one native build per FFmpeg major version; against Ubuntu 22.04's
  FFmpeg 4.4 only the FFmpeg 4 variant can load, and that variant needs
  `libpython3.10.so.1.0`, which the base image does not ship. `Dockerfile.cuda`
  now installs the `libpython3.10` package so torchcodec imports cleanly. The
  warning was cosmetic — WhisperX always passes pyannote in-memory waveform
  tensors, so torchcodec's file decoding was never exercised — but it is gone.

### Added

- OpenAI-compatible `GET /v1/models` endpoint for model discovery. It lists
  every WhisperX shorthand model name accepted by the `model` field plus the
  server-configured default (flagged `"default": true`, appended even when it
  is a custom HuggingFace path); any HuggingFace Whisper path remains accepted
  per request even when not listed. The endpoint requires the API key when
  `WHISPERX_API_KEY` is set.
- GitHub Actions workflow that builds the CPU and CUDA Docker images and
  publishes them to the GitHub Container Registry on pushes to `main`,
  `v*` release tags, and manual dispatch (images are tagged `cpu`, `cuda`,
  `<version>-cpu`/`-cuda`, and `latest-cpu`/`latest-cuda`); pull requests
  build both images without pushing.

## [0.1.0] - 2026-10-16

Initial release. While the project is in the 0.x initial-development phase
allowed by Semantic Versioning, incompatible API changes may occur in minor
releases; from 1.0.0 onward, breaking changes require a major version bump.

### Added

- OpenAI-compatible transcription API: `POST /v1/audio/transcriptions` accepts
  the OpenAI Audio API multipart fields (`model`, `language`, `prompt`,
  `response_format`, `temperature`, `task`) plus WhisperX tuning fields (batch
  size, VAD/chunking, beam search, length penalty, alignment, and diarization
  options), so OpenAI SDK clients work as drop-in replacements.
- All six response formats: `json`, `text`, `verbose_json`, `srt`, `vtt`, and
  `tsv`, with `SPEAKER_XX:` labels in subtitle and TSV output when diarization
  runs.
- WhisperX capabilities: batched inference, word-level timestamps via
  alignment, and speaker diarization (pyannote) with per-request model
  selection.
- LRU caches for ASR pipelines, per-language alignment models, and diarization
  models (`WHISPERX_MODEL_CACHE_SIZE`), with serialized model loading so
  concurrent requests share loaded models safely.
- Optional bearer-token authentication via `WHISPERX_API_KEY`; `/health`,
  `/docs`, and `/openapi.json` remain public.
- Automatic device selection (CUDA when available, otherwise CPU) and
  configuration through `WHISPERX_*` environment variables, including the
  default model, compute type, download root, offline mode, and startup model
  preload.
- HuggingFace integration: any Whisper checkpoint or HuggingFace model path
  can be requested, with token handling for gated models; unknown or
  unauthorized models fail with HTTP 400 carrying the upstream HuggingFace
  error.
- `whisperx-web` console script and CPU/CUDA Dockerfiles.
- README covering usage, request fields, configuration, and Docker images.
- Hermetic pytest suite (109 tests) covering configuration, response
  formatting, transcription orchestration, and the HTTP API, runnable without
  torch, models, or network access; `dev` extra with `pytest` and `httpx2`.

### Fixed

- `GET /openapi.json` returned a 500 because documented response media types
  were keyed by `ResponseFormat` enum members instead of HTTP status codes;
  the schema now documents all media types under status 200.

[Unreleased]: https://github.com/dasantonym/WhisperX-Web/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dasantonym/WhisperX-Web/releases/tag/v0.1.0
