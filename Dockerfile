FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

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

RUN pip install --upgrade pip wheel

# Install Cython for youtokentome
RUN pip install --no-cache-dir "Cython"

# Install youtokentome from working fork
RUN pip install --no-cache-dir "git+https://github.com/LahiLuk/YouTokenToMe.git"

# Install NeMo 2.4 or latest 2.x (for Parakeet v3)
RUN pip install --no-cache-dir "nemo_toolkit[asr]>=2.4.0,<3.0"

# Install pyannote and runpod
RUN pip install --no-cache-dir "pyannote.audio" "runpod"

COPY handler.py .

ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "handler.py"]