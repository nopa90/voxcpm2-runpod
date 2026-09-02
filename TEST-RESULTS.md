# VoxCPM2 RunPod — Test Results & Learnings (2026-09-02)

Status: **Two serverless endpoints verified end-to-end.** 7 TTS jobs + 1
whisper transcription job completed. Six distinct Russian voices generated via
`voice_design`, transcribed back by whisper with word-for-word accuracy.

## Whisper round-trip verification (2026-09-02)

Deployed `yairlifshitz/whisper-runpod-serverless:latest` (weights-baked image,
9.64 GB compressed — same weights-in-image pattern) to verify our synthesized
Russian audio transcribes cleanly.

| Endpoint | Image | Template | GPU | Workers |
|----------|-------|----------|-----|---------|
| `d6wx79l48172qr` — voxcpm2-tts | `nopa90/voxcpm2-runpod:weights` | `cr67vewzar` | ADA_24 | 0-2 |
| `w9ic3c9cmvx2i2` — whisper-ivrit | `yairlifshitz/whisper-runpod-serverless:latest` | `dg21plqv81` | ADA_24 | 0-2 |

Test: transcribed `tests/audio/voxcpm-seg-01-commander.wav` (12.5 s Russian,
auto language detect) with `faster-whisper` / `large-v3-turbo`.

- **Result: word-for-word match** to the source script, 3 segments:
  1. "Флот, говорит командир, сегодня ночью мы держим рубеж."
  2. "Враг прорвался на северном хребте, но мы не отдадим ни метра земли."
  3. "Приготовиться к обороне, огонь открывать только по моему сигналу."
- Word confidences 0.98-1.0, `avg_logprob -0.089`, `no_speech_prob 0`
- Job COMPLETED in 76.7 s wall (cold start incl. 9.6 GB image pull + model load)
- Output = aggregate stream of `progress` + `segments` events (word timestamps,
  `speakers` field present for diarization), not a flat transcript
- Punctuation normalized by whisper (commas vs periods at boundaries) — expect
  to re-punctuate transcripts if exact text matters

Request shape (engine, model, transcribe_args with `blob` or `url`):

```json
{ "input": { "engine": "faster-whisper", "model": "large-v3-turbo",
             "transcribe_args": { "blob": "<base64>", "language": "ru" } } }
```

### Whisper image anatomy (registry inspection, no full pull)

Layers: pytorch:2.7.1-cuda12.8 conda base (4.2 GB) + ffmpeg + pip
(`ivrit[all]`, torch, hf-hub, runpod) + one RUN-layer per baked model:
2× ivrit ct2 whisper models + stock `large-v3-turbo` + pyannote diarization +
speechbrain ECAPA embeddings; final layer `ADD infer.py .` (3 KB handler).

Handler ideas worth borrowing:
- **`local_files_only=True`** on model load — baked weights never silently hit network
- **Per-model RUN layers** — changing one model only re-downloads its layer on rebuild
- **Generator handler + `return_aggregate_stream: True`** — streams progress/segments
- Pin everything (torch 2.7.1, hf-hub 0.36.0, ivrit 0.2.6)
- Do NOT copy: startup `sys.exit(1)` on no-CUDA at import (our lesson: never crash
  at import → stuck queues), `api_key` passed inside job input
- Their tag history (21 GB → 9.6 GB) shows aggressive size iteration

## Proven working

| Capability | Result |
|------------|--------|
| Serverless endpoint (RunPod, ADA_24 pool / L4-class, `:weights` image) | ✅ |
| `mode: tts` — plain Russian synthesis | ✅ 4.0 s, clean 48 kHz mono PCM |
| `mode: voice_design` — `"(description)text"` format | ✅ 6/6 distinct voices |
| English natural-language voice descriptions + Russian speech text | ✅ works |
| 48 kHz, 16-bit mono output | ✅ header-verified |
| Stitching segments into one track | ✅ 73 s demo |

## Throughput (NVIDIA L4, warm worker, weights baked in)

- **Cold start** (image pull + model load on first job): ~2-4 min
- **Model load** once per worker, cached across jobs
- **Generation ≈ realtime or faster**: ~10-13 s of speech rendered in ~10 s wall
  on a warm worker (10 inference timesteps, default cfg)
- Workers scale to zero after idle timeout (60 s set on endpoint)

## Voices generated (all distinct, user-approved)

| Label | Description used (English) | Speech dur |
|-------|----------------------------|-----------|
| commander | Deep authoritative male commander, aged 50, low pitch, gravelly texture, calm commanding, slow deliberate | 12.5 s |
| young-woman | Bright young female, aged 20, clear, energetic, light cheerful, fast lively | 9.8 s |
| elder-narrator | Elderly male narrator, aged 70, warm weathered, slow deliberate, gentle wise | 12.5 s |
| soldier | Young male soldier, aged 25, breathless intense, mid pitch, urgent stressed, short clipped | 11.8 s |
| anchor | Calm professional female news anchor, aged 35, smooth polished, neutral standard Russian, measured | 10.9 s |
| villain | Harsh menacing male, low pitch, slow, cold cruel, whispery edge | 13.4 s |

Full texts live in the kernel `SEGMENTS` list (runpod kernel session) and the
`(description)` prefix is REQUIRED — without parentheses the model silently
does plain TTS (no error).

## Reproduction

1. Images on Docker Hub: `nopa90/voxcpm2-runpod:weights` (commit `b92ce81`).
2. Endpoint used: `d6wx79l48172qr` (template `cr67vewzar` → `:weights` tag).
   Template references the tag, so image rebuilds roll out to existing endpoints.
3. Request shape (one job per utterance):

```json
{ "input": { "mode": "voice_design",
             "text": "(Deep authoritative male commander, aged 50, ...)Флот, говорит командир..." } }
```

4. Submit to `/run`, poll `/status/{id}` until COMPLETED; output has
   `wav_b64`, `sample_rate: 48000`, `duration_seconds`, `file_size_bytes`.

## Audio artifacts (local only, not in git)

```
tests/audio/voxcpm-seg-0N-<voice>.wav   # per-voice segments
tests/audio/voxcpm-long-test.wav        # 73 s stitched demo
```

## Gotchas worth remembering

- Model loads **lazily inside `handler()`** — never at import (import crash =
  silent worker crash-loop + jobs stuck IN_QUEUE forever).
- **Entire handler body is inside one try/except** returning
  `{"error": ..., "type": ...}` — a thrown exception can mark jobs FAILED and
  churn the queue; a returned dict always completes cleanly.
- `transformers==4.51.3` must stay pinned — voxcpm has no upper bound and the
  latest transformers breaks `LlamaTokenizerFast` (TypeError on add_tokens).
- Worker logs are **console-only** (no API) — Serverless → endpoint → Workers → Logs.
- Endpoint `gpuIds` are **GPU pool IDs** (`ADA_24`), not model names.
- Templates must have **unique names** — reuse an existing one instead of
  re-creating, or suffix the name.
- Wave stitching: read each segment's PCM via the stdlib `wave` module,
  concat with zero-padding silences, write mono/16-bit/48 kHz. (`wave` errors
  like `# channels not specified` vanish when params are set explicitly.)
