#!/usr/bin/env bash
# Download VoxCPM2 weights with SHA-256 verification (used at Docker build time).
# Retry loop with fresh redirects — HF/Xet connections drop mid-download.
set -euo pipefail

DEST="${1:?usage: download-weights.sh <dest-dir>}"
REPO="openbmb/VoxCPM2"
BASE="https://huggingface.co/${REPO}/resolve/main"

mkdir -p "$DEST"

EXPECT_model_safetensors_SHA="f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d"
EXPECT_model_safetensors_SIZE="4580080592"
EXPECT_audiovae_SIZE="376951122"

download() {
    local file="$1"
    local attempt=0
    while (( attempt < 50 )); do
        attempt=$((attempt + 1))
        if curl --fail --location --http1.1 --continue-at - \
                --retry 10 --retry-all-errors --retry-delay 10 \
                --connect-timeout 30 --speed-time 180 --speed-limit 1024 \
                --keepalive-time 30 \
                --output "$DEST/$file" "$BASE/$file"; then
            return 0
        fi
        echo "curl failed (attempt $attempt), retrying with fresh redirect..."
        sleep 10
    done
    echo "ERROR: $file failed after $attempt attempts" >&2
    return 1
}

echo "[weights] downloading small files..."
for f in tokenizer.json config.json special_tokens_map.json; do
    [ -s "$DEST/$f" ] || download "$f" || true
done

echo "[weights] downloading model.safetensors (4.37 GB)..."
if [ ! -s "$DEST/model.safetensors" ]; then download model.safetensors; fi

echo "[weights] downloading audiovae.pth (360 MB)..."
if [ ! -s "$DEST/audiovae.pth" ]; then download audiovae.pth; fi

echo "[weights] verifying..."
SIZE=$(stat -c%s "$DEST/model.safetensors" 2>/dev/null || stat -f%z "$DEST/model.safetensors")
[ "$SIZE" = "$EXPECT_model_safetensors_SIZE" ] || { echo "ERROR: model.safetensors size $SIZE" >&2; exit 1; }
SHA=$(sha256sum "$DEST/model.safetensors" | cut -d' ' -f1)
[ "$SHA" = "$EXPECT_model_safetensors_SHA" ] || { echo "ERROR: model.safetensors sha mismatch: $SHA" >&2; exit 1; }
SIZE=$(stat -c%s "$DEST/audiovae.pth" 2>/dev/null || stat -f%z "$DEST/audiovae.pth")
[ "$SIZE" = "$EXPECT_audiovae_SIZE" ] || { echo "ERROR: audiovae.pth size $SIZE" >&2; exit 1; }

echo "[weights] all files verified OK"
