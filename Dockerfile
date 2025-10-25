# Use public PyTorch base image with CUDA support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    sox \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# CRITICAL FIXES: Pin numpy and huggingface_hub to working versions
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "huggingface-hub==0.24.7"

# Install all other dependencies
RUN pip install --no-cache-dir \
    "cuda-python==12.6.0" \
    "soundfile==0.12.1" \
    "librosa==0.10.2.post1" \
    "hydra-core==1.3.2" \
    "omegaconf==2.3.0" \
    "pyyaml==6.0.2" \
    "lhotse==1.24.2" \
    "tqdm==4.66.5" \
    "requests==2.32.3" \
    "transformers==4.44.2" \
    "sentencepiece==0.2.0" \
    "scikit-learn==1.5.2" \
    "pandas==2.2.3" \
    "joblib==1.4.2" \
    "matplotlib==3.9.2" \
    "soxr==0.4.0" \
    "resampy==0.4.3" \
    "jiwer==3.0.4" \
    "pooch==1.8.2" \
    "numba==0.60.0" \
    "llvmlite==0.43.0" \
    "platformdirs==4.3.6" \
    "future==1.0.0" \
    "lazy_loader==0.4" \
    "pydub==0.25.1"

# Install pytorch-lightning and dependencies for NeMo
RUN pip install --no-cache-dir \
    "pytorch-lightning==2.4.0" \
    "torchmetrics==1.4.2"

# Install NeMo WITHOUT youtokentome dependency
RUN pip install --no-cache-dir --no-deps "nemo_toolkit==1.23.0"

# Install NeMo's required dependencies manually
RUN pip install --no-cache-dir \
    webdataset \
    braceexpand \
    editdistance \
    einops \
    frozendict \
    kaldi-python-io \
    kaldiio \
    marshmallow \
    packaging \
    pyannote.core \
    pyannote.metrics \
    rouge-score \
    wrapt

# Install pyannote.audio
RUN pip install --no-cache-dir "pyannote.audio==3.3.2"

# Install RunPod
RUN pip install --no-cache-dir "runpod==1.5.0"

# Copy the handler script
COPY handler.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose port (though not needed for serverless)
EXPOSE 8000

# Command to run the handler
CMD ["python", "handler.py"]
