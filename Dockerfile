# Use NVIDIA PyTorch base image with CUDA support
FROM nvcr.io/nvidia/pytorch:24.01-py3

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

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install dependencies following official documentation
# First install latest PyTorch, then other ML dependencies
RUN pip install -U torch torchaudio torchvision
RUN pip install -U nemo_toolkit[asr]
RUN pip install -r requirements.txt

# Copy the handler script
COPY handler.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose port (though not needed for serverless)
EXPOSE 8000

# Command to run the handler
CMD ["python", "handler.py"]
