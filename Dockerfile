FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Ensure CUDA is accessible for ONNX Runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PIP_CONSTRAINT=/app/constraints.txt
ENV PIP_NO_BUILD_ISOLATION=1
# Set CUDA library path - include /opt/conda/lib where PyTorch's CUDA libraries are located
# This helps ONNX Runtime find CUDA libraries
ENV LD_LIBRARY_PATH=/opt/conda/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

WORKDIR /app

# Create constraints file - allow numpy 1.x, flexible huggingface-hub
RUN echo "numpy<2.0" > /app/constraints.txt

# Install system packages AND gcc-11 for newer libstdc++
# Note: CUDA libraries are already in the PyTorch base image at /usr/local/cuda/lib64/
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    sox \
    git \
    gcc-11 \
    g++-11 \
    && rm -rf /var/lib/apt/lists/*

# Symlink the newer libstdc++ from gcc-11
RUN ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6

RUN pip install --upgrade pip wheel packaging

# Install Cython and build dependencies
RUN pip install --no-cache-dir "Cython"

# Install youtokentome from working fork FIRST
RUN pip install --no-cache-dir "git+https://github.com/LahiLuk/YouTokenToMe.git"

# Install megatron-core EXPLICITLY (required for NeMo 2.x)
RUN pip install --no-cache-dir "megatron-core"

# Install NeMo 2.4+ with ALL its dependencies
RUN pip install --no-cache-dir "nemo_toolkit[asr]>=2.4.0,<3.0"

# Install pyannote and runpod
RUN pip install --no-cache-dir "pyannote.audio" "runpod"

# Verify CUDA libraries exist before installing ONNX Runtime
RUN echo "Checking for CUDA libraries..." && \
    ls -la /usr/local/cuda/lib64/libcublas.so* 2>/dev/null || echo "⚠️ libcublas.so not found in /usr/local/cuda/lib64/" && \
    ls -la /usr/local/cuda/lib64/libcublasLt.so* 2>/dev/null || echo "⚠️ libcublasLt.so not found in /usr/local/cuda/lib64/" && \
    echo "✅ CUDA library check complete"

# Install ONNX Runtime GPU (required for pyannote 3.1)
# Using GPU version for CUDA 12.1 - onnxruntime-gpu 1.19.2 supports CUDA 12.1
# This provides GPU acceleration for speaker embeddings, improving diarization speed
RUN pip install --no-cache-dir "onnxruntime-gpu==1.19.2"

# Test ONNX GPU provider is available (verifies CUDA libraries are found)
# This will fail the build if CUDA provider is not available
RUN python3 -c "\
import onnxruntime as ort; \
import os; \
print('LD_LIBRARY_PATH:', os.environ.get('LD_LIBRARY_PATH', 'not set')); \
providers = ort.get_available_providers(); \
print('ONNX Providers:', providers); \
assert 'CUDAExecutionProvider' in providers, 'CUDA provider not found! Check that CUDA libraries are in LD_LIBRARY_PATH.'; \
print('✅ ONNX GPU is working!')"

# Create models directory for baked-in models
RUN mkdir -p /app/models

# Set HuggingFace cache directory (for build-time downloads)
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_DATASETS_CACHE=/app/models

# Accept HuggingFace token as build argument (OPTIONAL - for pre-downloading models)
# SECURITY: NEVER commit HF_TOKEN to GitHub! Use secrets/environment variables.
# 
# Option 1: Build locally with token (for faster startup):
#   docker build --build-arg HF_TOKEN=your_token_here .
#
# Option 2: GitHub Actions (use repository secrets):
#   In .github/workflows/build.yml: --build-arg HF_TOKEN=${{ secrets.HF_TOKEN }}
#
# Option 3: RunPod build (use RunPod secrets):
#   Set HF_TOKEN in RunPod build settings as a secret
#
# Option 4: Skip build-time download (default):
#   Models will download at runtime (works fine, just slower first request)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Download pyannote speaker diarization model during build (OPTIONAL)
# If HF_TOKEN is not provided, models will download at runtime instead
# Note: User must have accepted model terms at:
#   https://hf.co/pyannote/segmentation-3.1
#   https://hf.co/pyannote/speaker-diarization-3.1
RUN if [ -n "$HF_TOKEN" ] && [ "$HF_TOKEN" != "" ]; then \
        echo "📥 Downloading pyannote models during build (HF_TOKEN provided)..."; \
        python3 -c "\
from pyannote.audio import Pipeline; \
import os; \
os.makedirs('/app/models/pyannote-speaker-diarization-3.1', exist_ok=True); \
print('Downloading pyannote speaker-diarization-3.1 model...'); \
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token='$HF_TOKEN', cache_dir='/app/models'); \
print('✅ Pyannote model baked into image'); \
"; \
    else \
        echo "ℹ️  HF_TOKEN not provided during build (this is OK!)"; \
        echo "ℹ️  Models will download at runtime when needed (first request may be slower)"; \
        echo "ℹ️  To pre-download: docker build --build-arg HF_TOKEN=your_token ."; \
    fi

# Download Parakeet model during build (NeMo caches to default location)
# This pre-downloads the model so it's available immediately
RUN python3 -c "\
import nemo.collections.asr as nemo_asr; \
print('Downloading Parakeet TDT 0.6B v3 model...'); \
model = nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3', map_location='cpu'); \
print('✅ Parakeet model baked into image'); \
"

COPY handler.py .

ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "handler.py"]