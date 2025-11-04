# RunPod Serverless Deployment Guide

## Quick Deploy from GitHub

1. **Push to GitHub**: Upload all files to https://github.com/703deuce/parakeet.git

2. **Create RunPod Serverless Endpoint**:
   - Go to [RunPod Serverless](https://runpod.io/serverless)
   - Click "New Endpoint"
   - Choose "Container Image" deployment method
   - Use GitHub integration or build from Dockerfile

## Baking Models into Docker Image (Recommended)

**Benefits:**
- ✅ Cold start: 30-60 seconds (vs 5-10 minutes downloading models)
- ✅ Works in ALL regions (no network volume needed)
- ✅ No model download on startup
- ✅ Always available

**How to Enable:**

### Step 1: Get HuggingFace Token
1. Go to https://hf.co/settings/tokens
2. Create a token with **read** access
3. Accept model terms at:
   - https://hf.co/pyannote/segmentation-3.0
   - https://hf.co/pyannote/speaker-diarization-3.0

### Step 2: Set Build Argument in RunPod

When creating/editing your RunPod endpoint:

1. **If using GitHub integration:**
   - Go to your endpoint settings
   - Find "Build Arguments" or "Docker Build Args"
   - Add: `HF_TOKEN=your_huggingface_token_here`
   - Save and rebuild

2. **If using Docker Hub:**
   ```bash
   docker build --build-arg HF_TOKEN=your_huggingface_token_here -t yourusername/parakeet:v1 .
   docker push yourusername/parakeet:v1
   ```

3. **If using RunPod's build system:**
   - In RunPod dashboard → Endpoints → Edit
   - Under "Build Settings" → "Build Arguments"
   - Add: `HF_TOKEN` = `your_huggingface_token_here`
   - Trigger rebuild

### Step 3: Verify Models Are Baked

After rebuild, check logs on first startup:
- ✅ Should see: `📦 Loading baked-in pyannote model from Docker image`
- ✅ Should see: `✅ Baked-in pyannote model loaded successfully (no download needed)`
- ❌ Should NOT see: `🔄 Downloading pyannote speaker diarization model`

**Note:** If `HF_TOKEN` is not provided, models will download at runtime (slower cold start but still works).

## Container Configuration

### Recommended Settings:
- **Container Disk**: 
  - **With baked models**: 30GB minimum (models are ~5-10GB)
  - **Without baked models**: 20GB minimum (models download at runtime)
- **Memory**: 16GB minimum  
- **GPU**: RTX 4090, A100, or H100 (16GB+ VRAM)
- **Max Workers**: 1-5 (based on your needs)
- **Idle Timeout**: 300 seconds
- **Max Execution Time**: 600 seconds (10 minutes)

### Environment Variables:
Set these in your RunPod endpoint configuration for better security and flexibility:

| Variable | Description | Example |
|----------|-------------|---------|
| `RUNPOD_API_KEY` | Your RunPod API key (optional) | `your-api-key-here` |
| `RUNPOD_ENDPOINT_ID` | Your endpoint ID for logging | `abc123def456` |
| `RUNPOD_ENDPOINT_URL` | Full endpoint URL for clients | `https://api.runpod.ai/v2/abc123def456/runsync` |

**How to set in RunPod:**
1. Go to your serverless endpoint
2. Click "Settings" → "Environment Variables"
3. Add the variables above
4. Save and redeploy

## API Usage

### Endpoint URL Format:
```
https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync
```

### Request Format:
```json
{
  "input": {
    "audio_data": "base64_encoded_audio_data",
    "audio_format": "wav",
    "include_timestamps": false,
    "chunk_duration": 1200
  }
}
```

### Response Format:
```json
{
  "text": "Transcribed text with punctuation and capitalization.",
  "word_timestamps": [],
  "segment_timestamps": [],
  "char_timestamps": [],
  "audio_duration_seconds": 125.6,
  "chunks_processed": 1,
  "model_used": "nvidia/parakeet-tdt-0.6b-v2"
}
```

## Testing Your Deployment

Use the included test script:
```bash
python test_api.py your_audio.wav --api-endpoint https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync
```

With timestamps:
```bash
python test_api.py your_audio.wav --api-endpoint https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync --timestamps
```

## Cost Optimization

- **Cold Start (with baked models)**: ~30-60 seconds for first request
- **Cold Start (without baked models)**: ~5-10 minutes (downloading models)
- **Warm Requests**: ~0.1-2 seconds depending on audio length
- **Auto-scaling**: Scales to 0 when idle (no cost)
- **Processing Speed**: ~3380x real-time (very fast)

**💡 Tip:** Bake models into Docker image to reduce cold start from 5-10 minutes to 30-60 seconds!

## File Structure for GitHub

```
parakeet/
├── handler.py          # Main serverless handler
├── requirements.txt    # Python dependencies
├── Dockerfile         # Container configuration
├── README.md          # Documentation
├── test_api.py        # Testing script
├── example_usage.py   # Usage examples
└── DEPLOYMENT.md      # This file
```

## Troubleshooting

### Common Issues:

#### 1. **Dependency Conflicts During Build**
If you see PyTorch version conflicts:
```
ERROR: torch 2.8.0 is incompatible with torchtext 0.17.0a0
```

**Solutions:**
- Use the pinned versions in `requirements.txt` (torch==2.3.1)
- Or try `requirements-minimal.txt` for maximum compatibility
- Update Dockerfile to use: `COPY requirements-minimal.txt requirements.txt`

#### 2. **Model Loading Timeout**
- Increase max execution time to 600+ seconds
- First request takes 30-60 seconds to download model

#### 3. **Out of Memory Errors**
- Use GPU with 16GB+ VRAM (RTX 4090, A100, H100)
- Reduce chunk_duration parameter for very long audio

#### 4. **Audio Format Errors**
- Ensure proper base64 encoding
- Supported formats: WAV, MP3, FLAC, M4A, OGG
- Audio automatically resampled to 16kHz mono

#### 5. **NeMo Installation Issues**
If NeMo fails to install, try:
```dockerfile
# In Dockerfile, replace the pip install line with:
RUN pip install --no-cache-dir torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1
RUN pip install --no-cache-dir nemo_toolkit[asr]==2.2.0
```

### Alternative Docker Build:
If main requirements fail, use minimal version:
```bash
# Copy minimal requirements
cp requirements-minimal.txt requirements.txt
docker build -t your-username/parakeet-transcription .
```

### Logs:
Check RunPod dashboard for detailed error logs and performance metrics.
