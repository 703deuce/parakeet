FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    sox \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# WORKAROUND: Install wheel first (fixes youtokentome build)
RUN pip install wheel

# Install Cython for youtokentome
RUN pip install "Cython==0.29.37"

# Pin critical packages
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    huggingface-hub==0.23.5

# Install nemo and pyannote
RUN pip install --no-cache-dir \
    nemo_toolkit[asr]==1.23.0 \
    pyannote.audio==3.3.2 \
    runpod==1.5.0

COPY handler.py .

ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "handler.py"]