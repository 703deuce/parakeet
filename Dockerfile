# Use NVIDIA PyTorch base image with CUDA support
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    sox \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install dependencies with better conflict resolution
RUN pip install --no-cache-dir --force-reinstall -r requirements.txt

# Verify critical packages are working
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python -c "import torchaudio; print(f'TorchAudio version: {torchaudio.__version__}')" && \
    python -c "import nemo; print('NeMo imported successfully')"

# Copy the handler script
COPY handler.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose port (though not needed for serverless)
EXPOSE 8000

# Command to run the handler
CMD ["python", "handler.py"]
