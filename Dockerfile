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

# Install Cython GLOBALLY so it's available in build environments
RUN pip install --no-cache-dir "Cython==0.29.37"

# Install numpy and huggingface-hub first
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "huggingface-hub==0.23.5"

# Install nemo WITHOUT youtokentome dependency check, then pyannote and runpod
RUN pip install --no-cache-dir --no-deps "nemo_toolkit[asr]==1.23.0" && \
    pip install --no-cache-dir "pyannote.audio==3.3.2" "runpod==1.5.0"

# Install all nemo dependencies EXCEPT youtokentome
RUN pip install --no-cache-dir \
    hydra-core omegaconf pytorch-lightning torchmetrics \
    transformers webdataset editdistance jiwer \
    kaldi-python-io librosa marshmallow matplotlib \
    numba onnx pandas sacremoses sentencepiece \
    scipy tensorboard text-unidecode wget wrapt

COPY handler.py .

ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "handler.py"]