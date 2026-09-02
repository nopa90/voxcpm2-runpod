#!/usr/bin/env python3
"""
04-test-endpoint.py
Test a VoxCPM2 RunPod Serverless Endpoint.

Usage:
    python3 scripts/04-test-endpoint.py --endpoint-id <id> --text "Hello world"
    python3 scripts/04-test-endpoint.py --endpoint-id <id> --text "(young woman, gentle)Hello" --mode voice_design
    python3 scripts/04-test-endpoint.py --endpoint-id <id> --text "Cloned." --reference-audio speaker.wav --mode clone

Requires: RUNPOD_API_KEY in environment.
"""

import os
import sys
import json
import base64
import argparse
import urllib.request

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    print("ERROR: RUNPOD_API_KEY not set.")
    sys.exit(1)


def call_endpoint(endpoint_id: str, payload: dict, sync: bool = True) -> dict:
    """Call the RunPod endpoint (sync or async)."""
    mode = "runsync" if sync else "run"
    url = f"https://api.runpod.ai/v2/{endpoint_id}/{mode}"
    headers = {
        "Content-Type": "application/json",
        # RunPod/Cloudflare rejects default python-urllib User-Agent with 403
        "User-Agent": "VoxCPM2-RunPod-Workflow/1.0",
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
    }
    data = json.dumps({"input": payload}).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def main():
    parser = argparse.ArgumentParser(description="Test VoxCPM2 endpoint")
    parser.add_argument("--endpoint-id", required=True, help="Endpoint ID")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--mode", default="tts", choices=["tts", "voice_design", "clone", "ultimate_clone"],
                        help="Generation mode (default: tts)")
    parser.add_argument("--reference-audio", help="Path to reference WAV file (for clone mode)")
    parser.add_argument("--prompt-text", help="Transcript of reference audio (for ultimate_clone mode)")
    parser.add_argument("--cfg-value", type=float, default=2.0, help="CFG guidance value (default: 2.0)")
    parser.add_argument("--inference-timesteps", type=int, default=10, help="Diffusion steps (default: 10)")
    parser.add_argument("--seed", type=int, help="Random seed (optional)")
    parser.add_argument("--output", default="output.wav", help="Output WAV path (default: output.wav)")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Use async (non-blocking) endpoint")
    args = parser.parse_args()

    # Build payload
    payload = {
        "mode": args.mode,
        "text": args.text,
        "cfg_value": args.cfg_value,
        "inference_timesteps": args.inference_timesteps,
    }
    if args.seed is not None:
        payload["seed"] = args.seed

    # Handle reference audio
    if args.mode in ("clone", "ultimate_clone"):
        if not args.reference_audio:
            print(f"ERROR: --reference-audio required for {args.mode} mode")
            sys.exit(1)
        with open(args.reference_audio, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        payload["reference_wav_b64"] = audio_b64
        if args.mode == "clone":
            payload["prompt_wav_b64"] = audio_b64
        
    if args.mode == "ultimate_clone":
        if not args.prompt_text:
            print("ERROR: --prompt-text required for ultimate_clone mode")
            sys.exit(1)
        payload["prompt_text"] = args.prompt_text

    # Call endpoint
    sync = not args.use_async
    print(f"Calling endpoint {args.endpoint_id} ({'sync' if sync else 'async'}) ...")
    print(f"  Mode: {args.mode}")
    print(f"  Text: {args.text[:80]}{'...' if len(args.text) > 80 else ''}")

    result = call_endpoint(args.endpoint_id, payload, sync=sync)

    # Handle response
    status = result.get("status")
    if status == "COMPLETED":
        output = result.get("output", {})
        if "error" in output:
            print(f"ERROR from worker: {output['error']}")
            sys.exit(1)

        wav_b64 = output.get("wav_b64")
        if not wav_b64:
            print(f"ERROR: No WAV data in response. Full output:")
            print(json.dumps(output, indent=2))
            sys.exit(1)

        # Save WAV file
        wav_bytes = base64.b64decode(wav_b64)
        with open(args.output, "wb") as f:
            f.write(wav_bytes)

        duration = output.get("duration_seconds", "unknown")
        sample_rate = output.get("sample_rate", "unknown")
        file_size = len(wav_bytes)

        print()
        print("=" * 60)
        print(f"  SUCCESS!")
        print(f"  Output file  : {args.output}")
        print(f"  File size    : {file_size / 1024:.1f} KB")
        print(f"  Duration     : {duration}s")
        print(f"  Sample rate  : {sample_rate} Hz")
        print(f"  Cost         : ${result.get('executionTime', 0) * 0.0001:.6f}")
        print("=" * 60)

    elif status == "IN_QUEUE" or status == "IN_PROGRESS":
        request_id = result.get("id", "unknown")
        print(f"  Async request submitted: {request_id}")
        print(f"  Status: {status}")
        print(f"  Check with: curl https://api.runpod.ai/v2/{args.endpoint_id}/status/{request_id}")
    else:
        print(f"Unexpected response:")
        print(json.dumps(result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()