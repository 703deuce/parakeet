# Use public PyTorch base image with CUDA support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies including C++ compiler for NeMo builds
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install core dependencies (keep base image PyTorch 2.2.0)
RUN pip install "cuda-python>=12.3"
RUN pip install numpy soundfile librosa
RUN pip install hydra-core omegaconf pyyaml
RUN pip install lhotse
RUN pip install tqdm requests transformers
RUN pip install sentencepiece scikit-learn pandas joblib
RUN pip install matplotlib soxr resampy jiwer
RUN pip install pooch numba llvmlite platformdirs future lazy_loader

# Install NeMo toolkit (this might upgrade torchvision)
RUN pip install "nemo_toolkit[asr]"

# CRITICAL FIX: Pin to pyannote 3.x (what you had on Sept 12)
RUN pip install "pyannote.audio>=3.0,<4.0"

# Install other packages
RUN pip install pydub runpod>=1.5.0

# CRITICAL FIX: Force reinstall compatible torchvision AFTER everything else
RUN pip install --force-reinstall --no-deps torchvision==0.17.0

# Copy the handler script
COPY handler.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose port (though not needed for serverless)
EXPOSE 8000

# Command to run the handler
CMD ["python", "handler.py"]
