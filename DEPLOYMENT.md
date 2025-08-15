# RunPod Serverless Deployment Guide

## Quick Deploy from GitHub

1. **Push to GitHub**: Upload all files to https://github.com/703deuce/parakeet.git

2. **Create RunPod Serverless Endpoint**:
   - Go to [RunPod Serverless](https://runpod.io/serverless)
   - Click "New Endpoint"
   - Choose "Container Image" deployment method
   - Use GitHub integration or build from Dockerfile

## Container Configuration

### Recommended Settings:
- **Container Disk**: 20GB minimum
- **Memory**: 16GB minimum  
- **GPU**: RTX 4090, A100, or H100 (16GB+ VRAM)
- **Max Workers**: 1-5 (based on your needs)
- **Idle Timeout**: 300 seconds
- **Max Execution Time**: 600 seconds (10 minutes)

### Environment Variables:
None required - model downloads automatically

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

- **Cold Start**: ~30-60 seconds for first request
- **Warm Requests**: ~0.1-2 seconds depending on audio length
- **Auto-scaling**: Scales to 0 when idle (no cost)
- **Processing Speed**: ~3380x real-time (very fast)

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
1. **Model loading timeout**: Increase max execution time
2. **Out of memory**: Use GPU with more VRAM
3. **Audio format errors**: Ensure proper base64 encoding
4. **Long processing**: Audio automatically chunked at 20 minutes

### Logs:
Check RunPod dashboard for detailed error logs and performance metrics.
