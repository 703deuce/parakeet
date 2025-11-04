FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PIP_CONSTRAINT=/app/constraints.txt
ENV PIP_NO_BUILD_ISOLATION=1
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

WORKDIR /app

# Create constraints file - allow numpy 1.x, flexible huggingface-hub
RUN echo "numpy<2.0" > /app/constraints.txt

# Install system packages AND gcc-11 for newer libstdc++
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

# Install onnxruntime-gpu for pyannote 3.0 (required for ONNX-based embeddings)
RUN pip install --no-cache-dir "onnxruntime-gpu==1.16.3"

# Create models directory for baked-in models
RUN mkdir -p /app/models

# Set HuggingFace cache directory (for build-time downloads)
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_DATASETS_CACHE=/app/models

# Accept HuggingFace token as build argument (set in RunPod build settings)
# This allows downloading pyannote models during build
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Download pyannote speaker diarization model during build
# Note: This requires HF_TOKEN build arg and user must have accepted model terms
# at https://hf.co/pyannote/segmentation-3.0 and https://hf.co/pyannote/speaker-diarization-3.0
RUN if [ -n "$HF_TOKEN" ]; then \
        python3 -c "\
from pyannote.audio import Pipeline; \
import os; \
os.makedirs('/app/models/pyannote-speaker-diarization-3.0', exist_ok=True); \
print('Downloading pyannote speaker-diarization-3.0 model...'); \
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.0', use_auth_token='$HF_TOKEN', cache_dir='/app/models'); \
print('✅ Pyannote model baked into image'); \
"; \
    else \
        echo "⚠️ HF_TOKEN not provided - skipping pyannote model download (will download at runtime)"; \
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