# WhisperX Web

An OpenAI-compatible transcription web API built on
[WhisperX](https://github.com/m-bain/whisperX). It exposes a single
`POST /v1/audio/transcriptions` endpoint that accepts the same multipart form
fields as the OpenAI Audio API, plus every WhisperX knob (batching, VAD,
alignment, diarization) as additional form fields.

## Features

- OpenAI-compatible request/response contract: drop-in replacement for
  `client.audio.transcriptions.create(...)`.
- All six response formats: `json`, `text`, `verbose_json`, `srt`, `vtt`, `tsv`.
- WhisperX batched inference, word-level alignment, and speaker diarization.
- Per-request model selection with an LRU pipeline cache
  (`WHISPERX_MODEL_CACHE_SIZE`).
- Optional bearer-token authentication.
- CPU and CUDA Docker images.

## Quick start

Requires **ffmpeg** on the host (`whisperx.load_audio` shells out to it) and
Python 3.10+.

```bash
pip install -e .
whisperx-web --host 0.0.0.0 --port 8000
```

or run the app directly:

```bash
uvicorn whisperx_web.main:app --host 0.0.0.0 --port 8000
```

Transcribe a file:

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@speech.mp3 \
  -F model=small \
  -F response_format=verbose_json
```

OpenAI client example:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="ignored")
transcript = client.audio.transcriptions.create(
    model="small", file=open("speech.mp3", "rb")
)
print(transcript.text)
```

## Endpoints

| Method | Path                        | Description                     |
| ------ | --------------------------- | ------------------------------- |
| `POST` | `/v1/audio/transcriptions`  | Transcribe an uploaded audio file |
| `GET`  | `/health`                   | Server/device/model status      |
| `GET`  | `/docs`, `/openapi.json`    | Interactive API documentation   |

## Request fields (`POST /v1/audio/transcriptions`)

`file` (multipart) is required; every other field is optional.

### OpenAI-compatible fields

| Field             | Type    | Default        | Description |
| ----------------- | ------- | -------------- | ----------- |
| `model`           | string  | server default | A WhisperX model name (`tiny` … `large-v3`, `turbo`, `distil-*`, …) or any HuggingFace Whisper path (e.g. `openai/whisper-large-v3`). |
| `language`        | string  | auto-detect    | ISO 639-1/639-2 language code. |
| `prompt`          | string  | –              | Context to guide the model's style. |
| `response_format` | string  | `json`         | `json`, `text`, `verbose_json`, `srt`, `vtt`, or `tsv`. |
| `temperature`     | float   | `0.0`          | Sampling temperature (0–1). |
| `task`            | string  | `transcribe`   | `transcribe` or `translate` (non-English audio → English). |

### WhisperX decoding fields

| Field                          | Type      | Default | Description |
| ------------------------------ | --------- | ------- | ----------- |
| `batch_size`                   | int       | server default (`8`) | Batch size for batched inference. |
| `chunk_size`                   | int       | `30`    | Seconds of audio per VAD/ASR chunk. |
| `beam_size`                    | int       | –       | Beam width; enables beam search. |
| `patience`                     | float     | –       | Beam search patience (width multiplier). |
| `length_penalty`               | float     | –       | Exponential length penalty for beam search. |
| `compression_ratio_threshold`  | float     | –       | GZIP ratio above which decoding fails. |
| `logprob_threshold`            | float     | –       | Mean log-probability below which decoding fails. |
| `no_speech_threshold`          | float     | –       | No-speech probability above which a segment is skipped. |
| `condition_on_previous_text`   | bool      | –       | Feed previous output as prompt for the next window. |
| `initial_prompt`               | string    | –       | Prompt for the first window (takes precedence over `prompt`). |
| `word_timestamps`              | bool      | `false` | Extract word timestamps; implicitly enables alignment. |

### VAD fields

| Field        | Type   | Default    | Description |
| ------------ | ------ | ---------- | ----------- |
| `vad_onset`  | float  | `0.500`    | VAD onset threshold. |
| `vad_offset` | float  | `0.363`    | VAD offset threshold. |
| `vad_method` | string | `pyannote` | `pyannote` or `silero`. |

### Alignment fields

| Field                    | Type    | Default    | Description |
| ------------------------ | ------- | ---------- | ----------- |
| `align`                  | bool    | auto       | Word-level alignment; defaults to `word_timestamps`. |
| `align_model`            | string  | per-language | Custom wav2vec2 alignment model (HuggingFace path). |
| `interpolate_method`     | string  | `nearest`  | Fill method for missing word timestamps. |
| `return_char_alignments` | bool    | `false`    | Include character-level alignments. |

### Diarization fields

| Field            | Type   | Default                          | Description |
| ---------------- | ------ | -------------------------------- | ----------- |
| `diarize`        | bool   | `false`                          | Label segments/words with speakers. |
| `diarize_model`  | string | `pyannote/speaker-diarization-community-1` | Pyannote model name. |
| `min_speakers`   | int    | –                                | Minimum speaker count. |
| `max_speakers`   | int    | –                                | Maximum speaker count. |
| `num_speakers`   | int    | –                                | Exact speaker count. |
| `hf_token`       | string | server token                     | HuggingFace token for gated models. |

Diarization models are gated: accept the model terms at
[huggingface.co](https://huggingface.co/pyannote/speaker-diarization-community-1)
and provide a token (server-side via `HF_TOKEN` or per-request via `hf_token`).
Requests that reference an unknown or unauthorized HuggingFace model (including
gated models without a valid token) fail with HTTP 400 carrying the underlying
HuggingFace error.

## Response formats

- `json` — `{"text": "…"}`
- `text` — plain transcript (`text/plain`)
- `verbose_json` — task, language, duration, `text`, `segments` (with optional
  `speaker`, `words`), `words`, and any `warning`.
- `srt` / `vtt` — subtitle cues; cues are prefixed `SPEAKER_00:` when
  diarization ran.
- `tsv` — `start`, `end`, (`speaker`,) `text` columns.

## Configuration (environment variables)

| Variable                   | Default                                        | Description |
| -------------------------- | ---------------------------------------------- | ----------- |
| `WHISPERX_HOST` / `WHISPERX_PORT` | `0.0.0.0` / `8000` | Bind address (used by `whisperx-web`). |
| `WHISPERX_DEVICE`          | `cuda` if available else `cpu`                 | `cpu`, `cuda`, `mps`, … |
| `WHISPERX_DEVICE_INDEX`    | `0`                                            | CUDA device index. |
| `WHISPERX_COMPUTE_TYPE`    | `default`                                      | `int8`, `float16`, … (CTranslate2 compute type). |
| `WHISPERX_MODEL`           | `small`                                        | Default ASR model. |
| `WHISPERX_BATCH_SIZE`      | `8`                                            | Default batch size. |
| `WHISPERX_THREADS`         | `4`                                            | CPU threads for onnxruntime. |
| `WHISPERX_DOWNLOAD_ROOT`   | HF cache default                               | Model download directory (mount a volume in Docker). |
| `WHISPERX_LOCAL_FILES_ONLY`| `false`                                        | Never reach the network for models. |
| `HF_TOKEN` (or `HUGGINGFACE_TOKEN`, `HUGGING_FACE_HUB_TOKEN`) | – | HuggingFace token for gated models. |
| `WHISPERX_API_KEY`         | –                                              | When set, requests must send `Authorization: Bearer <key>`. `/health`, `/docs`, `/openapi.json` stay open. |
| `WHISPERX_MODEL_CACHE_SIZE`| `1`                                            | LRU size for loaded ASR pipelines. |
| `WHISPERX_PRELOAD_MODEL`   | `false`                                        | Load the default model at startup. |
| `WHISPERX_DIARIZE_MODEL`   | `pyannote/speaker-diarization-community-1`     | Default diarization model. |

## Docker

### Prebuilt images

CI publishes both images to the GitHub Container Registry on pushes to
`main`, `v*` release tags, and manual dispatch:

```bash
docker pull ghcr.io/dasantonym/whisperx-web:cpu
docker pull ghcr.io/dasantonym/whisperx-web:cuda
```

`cpu` and `cuda` track `main`; release tags (`0.1.0-cpu`, `0.1.0-cuda`, and
so on) and `latest-cpu` / `latest-cuda` are added when a `v*` tag is pushed.
New packages are private by default — make the package public in the GitHub
package settings if you want anonymous pulls.

### Build locally

CPU image:

```bash
docker build -f Dockerfile.cpu -t whisperx-web:cpu .
docker run --rm -p 8000:8000 -v whisperx-models:/models \
  -e WHISPERX_DOWNLOAD_ROOT=/models -e WHISPERX_PRELOAD_MODEL=1 \
  whisperx-web:cpu
```

CUDA image:

```bash
docker build -f Dockerfile.cuda -t whisperx-web:cuda .
docker run --rm --gpus all -p 8000:8000 -v whisperx-models:/models \
  -e WHISPERX_DOWNLOAD_ROOT=/models \
  -e WHISPERX_COMPUTE_TYPE=float16 \
  whisperx-web:cuda
```

Mounting `WHISPERX_DOWNLOAD_ROOT` keeps downloaded models across restarts.

## Notes

- Requests are transcribed in a worker thread; model loading is serialized per
  pipeline key, so concurrent requests share a loaded model safely.
- Alignment models are cached per language and downloaded on first use.
- Temporary uploads are deleted immediately after transcription.
- A one-time `torchcodec is not installed correctly` warning at startup
  (e.g. on CPU or Apple Silicon builds where the PyPI torchcodec wheel's CUDA
  dependencies are absent) is harmless: WhisperX hands pyannote
  already-decoded waveform tensors, and pyannote only uses torchcodec to read
  audio from file paths, which never happens here.
