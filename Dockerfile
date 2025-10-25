# Use the PyTorch version from September 2024
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

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

# Just install everything like you did on Sept 12
RUN pip install --no-cache-dir \
    nemo_toolkit[asr]==1.23.0 \
    pyannote.audio==3.3.2 \
    runpod==1.5.0

COPY handler.py .

ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "handler.py"]