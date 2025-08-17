# Use public PyTorch base image with CUDA support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install core dependencies
RUN pip install -U torch torchaudio torchvision
RUN pip install nemo_toolkit[asr]
RUN pip install pyannote.audio
RUN pip install pydub
RUN pip install runpod>=1.5.0

# Copy the handler script
COPY handler.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose port (though not needed for serverless)
EXPOSE 8000

# Command to run the handler
CMD ["python", "handler.py"]
