# RunPod PyTorch base image with CUDA 12.4 + Python 3.11
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies (VoxCPM needs ffmpeg, libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY builder/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy worker code
COPY src/handler.py /app/handler.py

# Environment
ENV HF_HOME=/app/cache/huggingface
ENV TORCH_HOME=/app/cache/torch
ENV TOKENIZERS_PARALLELISM=false
# Model is downloaded into container disk on first cold start (see handler.py)
ENV VOXCPM_MODEL_DIR=/app/models/VoxCPM2

# Start worker
CMD ["python3", "-u", "/app/handler.py"]