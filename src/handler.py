"""
RunPod Serverless Worker for VoxCPM2.

Modes:
  tts              — basic text-to-speech
  voice_design     — create voice from natural-language description (prefix text with description)
  clone            — clone voice from reference audio
  ultimate_clone   — highest-fidelity clone (reference audio + transcript)

Input JSON:
  {
    "mode": "tts",                     // optional, default "tts"
    "text": "Hello world",             // required
    "reference_wav_b64": "...",        // base64 WAV, required for clone/ultimate_clone
    "prompt_text": "transcript of...", // required for ultimate_clone
    "cfg_value": 2.0,                  // optional
    "inference_timesteps": 10,         // optional
    "seed": 42,                        // optional
    "format": "wav"                    // optional, "wav" (default) or "mp3"
  }

Output JSON:
  {
    "wav_b64": "<base64 encoded audio>",
    "sample_rate": 48000,
    "duration_seconds": 3.5
  }
"""

import runpod
import os
import sys
import time
import tempfile
import base64
import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Model loading — runs once at cold start
# ---------------------------------------------------------------------------

MODEL_DIR = os.environ.get("VOXCPM_MODEL_DIR", "/app/models/VoxCPM2")
HF_REPO = "openbmb/VoxCPM2"


def ensure_model_downloaded() -> str:
    """Volume-free POC: download the model into container disk on first cold start.
    Files persist on the worker across requests until the worker scales down."""
    marker = os.path.join(MODEL_DIR, "model.safetensors")
    if os.path.exists(marker):
        print(f"[voxcpm2] Model already present at {MODEL_DIR}", flush=True)
        return MODEL_DIR

    print(f"[voxcpm2] Cold start: downloading {HF_REPO} into {MODEL_DIR} ...", flush=True)
    t0 = time.time()
    from huggingface_hub import snapshot_download
    snapshot_download(HF_REPO, local_dir=MODEL_DIR)
    print(f"[voxcpm2] Model downloaded in {time.time() - t0:.0f}s", flush=True)
    return MODEL_DIR


def load_model():
    """Load VoxCPM2 (downloading first if needed)."""
    global model, SAMPLE_RATE
    if model is not None:
        return model
    model_dir = ensure_model_downloaded()
    print(f"[voxcpm2] Loading model from {model_dir} ...", flush=True)
    t0 = time.time()

    from voxcpm import VoxCPM

    model = VoxCPM.from_pretrained(
        model_dir,
        load_denoiser=False,
    )
    SAMPLE_RATE = model.tts_model.sample_rate  # typically 48000

    elapsed = time.time() - t0
    print(f"[voxcpm2] Model loaded in {elapsed:.1f}s", flush=True)
    return model


# Lazy-loaded on first request. NEVER load at import time: an import failure
# kills the worker before the serverless runtime reports the error, and the
# job hangs IN_QUEUE forever while the worker crash-loops.
model = None
SAMPLE_RATE = 48000  # default until the model reports its real rate on load


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def decode_b64_wav(b64_str: str) -> str:
    """Decode a base64 WAV string into a temporary file and return its path."""
    raw = base64.b64decode(b64_str)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(raw)
    tmp.close()
    return tmp.name


def encode_wav_b64(wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    """Encode a numpy array to a base64 WAV string."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name

    sf.write(out_path, wav, sample_rate)

    with open(out_path, "rb") as f:
        audio_bytes = f.read()

    os.unlink(out_path)

    duration = len(wav) / sample_rate
    return {
        "wav_b64": base64.b64encode(audio_bytes).decode("utf-8"),
        "sample_rate": sample_rate,
        "duration_seconds": round(duration, 2),
        "file_size_bytes": len(audio_bytes),
    }


def cleanup_file(path: str):
    """Safely remove a temporary file."""
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def handler(job):
    load_model()  # lazy: loads on first request, cached for the worker lifetime
    data = job["input"]

    text = data.get("text", "").strip()
    if not text:
        return {"error": "No 'text' provided in input."}

    mode = data.get("mode", "tts")

    # Common generation params
    gen_kwargs = {}
    if "cfg_value" in data:
        gen_kwargs["cfg_value"] = float(data["cfg_value"])
    if "inference_timesteps" in data:
        gen_kwargs["inference_timesteps"] = int(data["inference_timesteps"])
    if "seed" in data:
        gen_kwargs["seed"] = int(data["seed"])

    temp_files = []

    try:
        # ---- Mode routing ----

        if mode == "voice_design":
            # Voice design: text should already be formatted as
            # "(voice description)Text to speak."
            wav = model.generate(text=text, **gen_kwargs)

        elif mode == "clone":
            ref_b64 = data.get("reference_wav_b64")
            if not ref_b64:
                return {"error": "Clone mode requires 'reference_wav_b64'."}
            ref_path = decode_b64_wav(ref_b64)
            temp_files.append(ref_path)
            wav = model.generate(
                text=text,
                reference_wav_path=ref_path,
                **gen_kwargs,
            )

        elif mode == "ultimate_clone":
            prompt_b64 = data.get("prompt_wav_b64")
            prompt_text = data.get("prompt_text")
            if not prompt_b64:
                return {
                    "error": "Ultimate clone mode requires 'prompt_wav_b64'."
                }
            if not prompt_text:
                return {
                    "error": "Ultimate clone mode requires 'prompt_text'."
                }
            prompt_path = decode_b64_wav(prompt_b64)
            temp_files.append(prompt_path)

            # Optional: use a separate reference audio for better similarity
            ref_path = None
            ref_b64 = data.get("reference_wav_b64")
            if ref_b64:
                ref_path = decode_b64_wav(ref_b64)
                temp_files.append(ref_path)
            else:
                ref_path = prompt_path

            wav = model.generate(
                text=text,
                prompt_wav_path=prompt_path,
                prompt_text=prompt_text,
                reference_wav_path=ref_path,
                **gen_kwargs,
            )

        else:
            # Basic TTS
            wav = model.generate(text=text, **gen_kwargs)

        # Encode result
        result = encode_wav_b64(wav, SAMPLE_RATE)
        return result

    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}

    finally:
        for f in temp_files:
            cleanup_file(f)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})