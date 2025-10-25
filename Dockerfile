FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PIP_CONSTRAINT=/app/constraints.txt
ENV PIP_NO_BUILD_ISOLATION=1

WORKDIR /app

# Create constraints file
RUN echo "numpy==1.26.4" > /app/constraints.txt && \
    echo "huggingface-hub==0.23.5" >> /app/constraints.txt

RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    sox \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel

# Install Cython and youtokentome directly
RUN pip install --no-cache-dir "Cython==0.29.37"
RUN pip install --no-cache-dir "youtokentome"

# Install numpy and huggingface-hub first
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "huggingface-hub==0.23.5"

# Install nemo and pyannote normally
RUN pip install --no-cache-dir \
    "nemo_toolkit[asr]==1.23.0" \
    "pyannote.audio==3.3.2" \
    "runpod==1.5.0"

COPY handler.py .

ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "handler.py"]