FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PIP_CONSTRAINT=/app/constraints.txt
ENV PIP_NO_BUILD_ISOLATION=1
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

WORKDIR /app

# Create constraints file
RUN echo "numpy==1.26.4" > /app/constraints.txt && \
    echo "huggingface-hub==0.23.5" >> /app/constraints.txt

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

RUN pip install --upgrade pip wheel

# Install Cython GLOBALLY so it's available in build environments
RUN pip install --no-cache-dir "Cython==0.29.37"

# Install youtokentome from working fork
RUN pip install --no-cache-dir "git+https://github.com/LahiLuk/YouTokenToMe.git"

# Install numpy and huggingface-hub first
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "huggingface-hub==0.23.5"

# NOW install nemo WITH all dependencies (youtokentome already installed, so it won't try to build it)
RUN pip install --no-cache-dir "nemo_toolkit[asr]==1.23.0"

# Install pyannote and runpod
RUN pip install --no-cache-dir "pyannote.audio==3.3.2" "runpod==1.5.0"

COPY handler.py .

ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "handler.py"]