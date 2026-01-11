# YuE Singing Voice Synthesis - RunPod Serverless
#
# This Dockerfile builds a serverless handler for YuE model inference.
#
# Build: docker build -t yue-serverless .
# Test locally: docker run --gpus all -p 8000:8000 yue-serverless
#
# For RunPod deployment:
# 1. Push to Docker Hub: docker push yourusername/yue-serverless
# 2. Create serverless endpoint on RunPod with this image

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/runpod-volume/cache
ENV HF_HOME=/runpod-volume/cache
ENV TORCH_HOME=/runpod-volume/cache

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements-runpod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-runpod.txt

# Install flash-attention (optional - skip if build fails)
# flash-attn requires CUDA compilation, may fail in some build environments
# The model will still work without it, just use more VRAM
RUN pip install flash-attn --no-build-isolation || echo "flash-attn install failed, continuing without it"

# Copy the entire YuE repository
COPY . .

# Create cache directory
RUN mkdir -p /runpod-volume/cache

# Pre-download models during build (optional - speeds up cold starts)
# Uncomment to bake models into image (increases image size significantly)
# RUN python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('m-a-p/YuE-s1-7B-anneal-en-cot')"
# RUN python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('m-a-p/YuE-s2-1B-general')"

# Set the handler as entry point
CMD ["python", "-u", "handler.py"]
