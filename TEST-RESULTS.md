# VoxCPM2 RunPod — Test Results & Learnings (2026-09-02)

Status: **POC endpoint verified end-to-end.** 7 jobs completed, six distinct
Russian voices generated via `voice_design`, one 73 s stitched demo.

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
