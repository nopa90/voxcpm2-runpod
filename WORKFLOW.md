# VoxCPM2 RunPod Serverless Endpoint — Implementation Workflow

## Purpose

Deploy [VoxCPM2](https://github.com/OpenBMB/VoxCPM) (OpenBMB, 2B params, 30 languages, 48kHz) as a RunPod Serverless Endpoint with persistent storage via a 10 GB Network Volume.

## Prerequisites

- [ ] RunPod account with billing enabled
- [ ] `RUNPOD_API_KEY` set in environment (`export RUNPOD_API_KEY="rpa_..."`)
  - Find at: RunPod Console → Settings → API Keys
- [ ] Docker installed locally (for building + pushing the worker image)
- [ ] RunPod CLI (`pip install runpod`)
- [ ] Access to a Docker registry (Docker Hub or RunPod's registry)

## Architecture

```
┌────────────────────────────────────────────────────┐
│              RunPod Serverless                      │
│                                                     │
│  ┌──────────────┐    /workspace/models/VoxCPM2/     │
│  │  GPU Worker   │◄──── Network Volume (10 GB) ─────│
│  │  (RTX 4090)   │                                  │
│  │               │    Persistent NFS. Survives       │
│  │  handler.py   │    scale-to-zero.                 │
│  └──────┬───▲────┘                                  │
│         │   │                                       │
└─────────┼───┼───────────────────────────────────────┘
          │   │
    request  response
    (JSON)   (base64 WAV)
          │   │
     ┌────▼───┴────┐
     │  Your code  │
     │  curl/app   │
     └─────────────┘
```

## Model Weights

| File                            | Size     |
|---------------------------------|----------|
| `model.safetensors` (2B LM)    | 4.37 GB  |
| `audiovae.pth` (AudioVAE V2)  | 360 MB   |
| `tokenizer.json`               | 3.5 MB   |
| **Total**                       | **~4.8 GB** |

Two delivery options, both built by the GitHub Actions workflows in the repo:

| Image tag | Workflow | Weights | Use when |
|-----------|----------|---------|----------|
| `:latest` | `build.yml` | downloaded at worker cold start (~4.6 GB from HF, cached on worker) | default POC |
| `:weights` | `build-weights.yml` | **baked into the image** (SHA-256-verified download at build time) | fast cold starts; +5 GB image size |

Both use the same `src/handler.py`; the handler skips its runtime download
when `VOXCPM_MODEL_DIR` already contains `model.safetensors`.

Weights verification (from the proven volume download): `model.safetensors`
= 4,580,080,592 B, sha256 `f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d`;
`audiovae.pth` = 376,951,122 B.

---

## Implementation Status (POC)

- [x] Step 1: Network Volume — created, used for model download, **dropped** (2026-09-01)
  - History: `d1zjhd73vh` (10 GB US-CA-2) → deleted for stock issues; `834lt42bp3`
    (20 GB US-MO-2) → held verified model → deleted per user decision
- [x] Step 2: Model download procedure proven (curl retry loop, verified SHA-256)
- [x] Step 3: Build Docker image via **GitHub Actions** → Docker Hub (`nopa90/voxcpm2-runpod`)
  - repo: https://github.com/nopa90/voxcpm2-runpod (secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`)
  - workflow: `.github/workflows/build.yml` → tag `:latest` (code only, cold-start weights)
  - workflow: `.github/workflows/build-weights.yml` → tag `:weights` (weights baked in)
  - Local build abandoned: ~13 GB over home internet; runpod CLI has no remote builder
- [x] Step 4: Serverless Endpoint lifecycle exercised — then **deleted** (2026-09-02)
  - History: `jo9m97h074ypaw` + `mk9h0hu65h36bi` deleted; fresh `s923mat522tbiy` created, tested,
    and deleted when workers hung server-side. Account is at **zero endpoints**.
  - Endpoint creation uses the SDK (`runpod.create_template` + `runpod.create_endpoint`)
  - Gotcha: endpoint `gpuIds` takes **GPU pool IDs** (`ADA_24`, `AMPERE_80`, …), not model names like "NVIDIA L4"
- [ ] Step 5: Test the endpoint (`scripts/04-test-endpoint.py` — basic TTS, voice_design, clone)
  - Recreate endpoint from template when ready; test one job before scaling

**Bugfixes shipped in the image (commit `77df5b1`):**
1. voxcpm + latest transformers broke (`LlamaTokenizerFast` TypeError) → pinned `transformers==4.51.3`
2. model loaded at import crashed the worker silently → lazy-load inside `handler()`
3. leftover `SAMPLE_RATE = model.tts_model...` at import dereferenced None → set after load

The removed `runpod.toml`/`hub.json`/obsolete scripts documented the abandoned
venv-on-volume and volume-based paths; only `scripts/04-test-endpoint.py` remains.

**Strategy change:** no Network Volume. Model weights either download from
HuggingFace on worker cold start (image `:latest`) or are baked into the image
(`:weights`). No volume cost, no DC-pinning problem. Docker Hub public image
storage is free.

**Step 4 image build — corrected plan:** the runpod CLI has no remote builder
(`runpod builder` does not exist in v1.x; `project deploy` is venv-on-volume,
not docker). The image will be built by **GitHub Actions** in GitHub's cloud
and pushed to Docker Hub (`nopa90/voxcpm2-runpod`). Nothing downloads to the
local Mac; Docker Hub public image storage is free.

Detailed API lessons live in: **`reference/runpod-api-lessons.md`**

---

## Step-by-Step Implementation

### Step 1: Set environment variables

The API key lives in `.env` as `RUNPOD_KEY` (not `RUNPOD_API_KEY`). Scripts read it from the environment:

```bash
export RUNPOD_API_KEY=$(grep '^RUNPOD_KEY=' ~/Projects/AI-Studio/.env | cut -d= -f2)
```

> Note: `RUNPOD_S3_USER`/`RUNPOD_S3_SECRET` in `~/.zshrc` are S3 storage credentials, not the management API key.

### Step 2: Create Network Volume (10 GB)  — DONE

```bash
python3 scripts/01-create-volume.py --region US-CA-2
```

**Gotchas hit:**
- Must pass `dataCenterId` (not `region`) — schema requires it.
- `US-OR-1` doesn't support network volumes; use `US-CA-2` (or see the list in the reference doc).
- A `500 Internal Server Error` from this mutation can mean the volume **was created** — always re-query `myself { networkVolumes }` before retrying, or you get silent duplicates (we did).
- Raw `urllib` needs a custom `User-Agent` or Cloudflare 403s the request.

Volume created: `d1zjhd73vh`

### Step 3: Download model weights onto the volume

Spin up a **temporary GPU Pod** with the volume mounted, run the download script, then terminate the Pod.

```bash
export RUNPOD_VOLUME_ID=d1zjhd73vh
python3 scripts/02-download-model-on-pod.py --volume-id $RUNPOD_VOLUME_ID
```

This does:
1. Creates a temporary Pod (GPU, with volume at `/workspace`)
2. SSHs in and runs `pip install voxcpm && python -c "from voxcpm import VoxCPM; VoxCPM.from_pretrained('openbmb/VoxCPM2')"`
3. Waits for model to appear at `/workspace/models/VoxCPM2/`
4. Terminates the temporary Pod
5. Volume persists with model weights

Volume: dropped (was `834lt42bp3`). See Strategy change note in the status
section — POC proceeds without a volume.

### Step 3: Download model weights onto the volume  — DONE

```bash
python3 scripts/02-download-model-on-pod.py --volume-id 834lt42bp3
```

**What happened / gotchas hit:**

1. **Datacenter must have GPU stock.** `podFindAndDeployOnDemand` with a
   `dataCenterId` fails with "no instances available" if that DC has no stock
   of that GPU. US-CA-2 had none for any common GPU. We probed DC×GPU
   combinations with throwaway pods (failed deploys are free) and found
   **US-MO-2 has NVIDIA L4**. Volume was deleted and recreated there
   (`834lt42bp3`) — cheap while the volume is empty.
2. **A pod can only mount a volume in its own datacenter.** Plan accordingly.
3. **Pod base image lacks `huggingface_hub`** — `pip install` it in the remote script.
4. **bash-tool timeouts kill the SSH session but NOT the pod** — the pod keeps
   billing. Either run the remote download under `setsid nohup ... &` (survives
   disconnect) and poll, or make sure the local runner can block long enough.
5. **Xet downloads stall on RunPod pods** (`CAS Client Error`, `httpx.ReadTimeout`).
   Fix: `HF_HUB_DISABLE_XET=1` helps but was still flaky; the reliable path was
   a **persistent curl retry loop with fresh redirects** for the two big files
   (see `dl-big.sh` pattern in reference/runpod-api-lessons.md). Connection
   resets repeatedly, then completed at 43 MB/s on a good attempt.
6. **`pgrep -f` self-match trap**: an SSH command line containing the pattern
   matches itself — always grep for the full command (`bash /workspace/x.sh`).
7. **Verify downloads**: sizes (4,580,080,592 B safetensors / 376,951,122 B
   audiovae) and SHA-256 (`f7f964cf…` matches HF's expected hash).

### Step 4: Build and push the Docker image

```bash
# Log in to Docker Hub (or use RunPod registry)
docker login

# Build the worker image
docker build -t YOUR_DOCKER_USER/voxcpm2-runpod:latest .

# Push
docker push YOUR_DOCKER_USER/voxcpm2-runpod:latest
```

### Step 5: Create Serverless Endpoint

```bash
python3 scripts/03-create-endpoint.py \
    --volume-id <vol_abc123> \
    --image YOUR_DOCKER_USER/voxcpm2-runpod:latest
```

This creates the endpoint and prints the **endpoint ID**.

### Step 6: Test the endpoint

```bash
# Basic TTS
python3 scripts/04-test-endpoint.py \
    --endpoint-id <endpoint_id> \
    --text "Hello, this is VoxCPM2 running on RunPod!"

# Voice design (Russian)
python3 scripts/04-test-endpoint.py \
    --endpoint-id <endpoint_id> \
    --text "(голос молодой женщины, уверенный)Привет, это VoxCPM2!"

# Voice cloning
python3 scripts/04-test-endpoint.py \
    --endpoint-id <endpoint_id> \
    --text "Cloned voice test." \
    --reference-audio path/to/speaker.wav
```

### Step 7: Publish to RunPod Hub (optional)

```bash
# Follow RunPod Hub publishing docs
# Builds on hub.json in this directory
```

---

## Files in This Directory

| File                                   | Purpose                                      |
|----------------------------------------|----------------------------------------------|
| `WORKFLOW.md`                          | This file — step-by-step guide               |
| `handler.py`                           | RunPod serverless worker (model + inference) |
| `Dockerfile`                           | Container image definition                   |
| `requirements.txt`                     | Python dependencies                          |
| `hub.json`                             | RunPod Hub listing metadata                  |
| `scripts/01-create-volume.py`          | Create 10 GB Network Volume                  |
| `scripts/02-download-model-on-pod.py`  | Populate volume with model weights           |
| `scripts/03-create-endpoint.py`        | Create serverless endpoint                   |
| `scripts/04-test-endpoint.py`          | Test the live endpoint                       |

---

## Cost Estimate (POC)

| Item                             | Cost         |
|----------------------------------|--------------|
| Network Volume 10 GB             | ~$0.70/mo    |
| Compute (RTX 4090, pay-per-use)  | ~$0.44/hr    |
| Cold start per activation        | ~20-40s      |
| Per 10s audio clip               | ~$0.0005     |
| **Monthly POC cost** (light use) | **~$2-5**    |

---

## Rollback / Teardown

```bash
python3 scripts/05-teardown.py --volume-id <vol_abc123> --endpoint-id <endpoint_id>
```

This deletes the endpoint and the network volume.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|-------------|
| Endpoint timeout on first request | Cold start + model loading from volume (~30-40s). Increase `executionTimeout` in endpoint config. |
| OOM on GPU | Ensure GPU selected has ≥24GB VRAM (RTX 4090/3090/A5000) |
| Volume mount fails | Verify volume region matches worker region |
| Model not found on volume | Step 2 incomplete. Re-run download script. |

---

## Completion Criteria

- [ ] Network Volume created, model downloaded onto it
- [ ] Docker image builds and pushes successfully
- [ ] Endpoint created and connected to volume
- [ ] `scripts/04-test-endpoint.py` returns a valid WAV file
- [ ] WAV file plays correct speech at 48kHz