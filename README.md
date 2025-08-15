# NVIDIA Parakeet Audio Transcription API

A serverless API endpoint for audio transcription using the NVIDIA Parakeet TDT 0.6B v2 model. This service automatically handles long audio files by splitting them into chunks and processing them sequentially.

## Features

- **High-Quality Transcription**: Uses NVIDIA's Parakeet TDT 0.6B v2 model for accurate English speech recognition
- **Long Audio Support**: Automatically splits audio files longer than 20 minutes into manageable chunks
- **Timestamp Support**: Optional word-level, segment-level, and character-level timestamps
- **Multiple Audio Formats**: Supports WAV, MP3, FLAC, and other common audio formats
- **Serverless Deployment**: Designed for RunPod serverless infrastructure

## Model Specifications

- **Model**: nvidia/parakeet-tdt-0.6b-v2
- **Parameters**: 600 million
- **Max Audio Length**: 24 minutes per chunk (default: 20 minutes for safety)
- **Supported Languages**: English
- **Features**: Punctuation, capitalization, timestamps

## API Usage

### Input Format

Send a POST request with the following JSON structure:

```json
{
  "input": {
    "audio_data": "base64_encoded_audio_data",
    "audio_format": "wav",
    "include_timestamps": true,
    "chunk_duration": 1200
  }
}
```

### Parameters

- `audio_data` (required): Base64 encoded audio file
- `audio_format` (optional): Audio format (wav, mp3, flac, etc.). Default: "wav"
- `include_timestamps` (optional): Include word/segment timestamps. Default: false
- `chunk_duration` (optional): Chunk duration in seconds. Default: 1200 (20 minutes)

### Response Format

```json
{
  "text": "Transcribed text with punctuation and capitalization.",
  "word_timestamps": [
    {
      "word": "Hello",
      "start": 0.5,
      "end": 0.8
    }
  ],
  "segment_timestamps": [
    {
      "segment": "Hello world.",
      "start": 0.5,
      "end": 1.2
    }
  ],
  "char_timestamps": [
    {
      "char": "H",
      "start": 0.5,
      "end": 0.52
    }
  ],
  "audio_duration_seconds": 125.6,
  "chunks_processed": 1,
  "model_used": "nvidia/parakeet-tdt-0.6b-v2"
}
```

## Example Usage

### Python Client Example

```python
import base64
import requests
import json

def transcribe_audio(audio_file_path, api_endpoint, include_timestamps=False):
    # Read and encode audio file
    with open(audio_file_path, 'rb') as audio_file:
        audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
    
    # Prepare request
    payload = {
        "input": {
            "audio_data": audio_data,
            "audio_format": "wav",
            "include_timestamps": include_timestamps
        }
    }
    
    # Send request
    response = requests.post(api_endpoint, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Usage
result = transcribe_audio("audio.wav", "https://your-runpod-endpoint.com", include_timestamps=True)
if result:
    print("Transcription:", result['text'])
    if result['word_timestamps']:
        print("Word timestamps available")
```

### JavaScript/Node.js Example

```javascript
const fs = require('fs');
const axios = require('axios');

async function transcribeAudio(audioFilePath, apiEndpoint, includeTimestamps = false) {
    try {
        // Read and encode audio file
        const audioBuffer = fs.readFileSync(audioFilePath);
        const audioData = audioBuffer.toString('base64');
        
        // Prepare request
        const payload = {
            input: {
                audio_data: audioData,
                audio_format: 'wav',
                include_timestamps: includeTimestamps
            }
        };
        
        // Send request
        const response = await axios.post(apiEndpoint, payload);
        
        return response.data;
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
        return null;
    }
}

// Usage
transcribeAudio('audio.wav', 'https://your-runpod-endpoint.com', true)
    .then(result => {
        if (result) {
            console.log('Transcription:', result.text);
            if (result.word_timestamps) {
                console.log('Word timestamps available');
            }
        }
    });
```

## Deployment on RunPod

### 1. Build and Push Docker Image

```bash
# Build the Docker image
docker build -t your-username/parakeet-transcription .

# Push to Docker Hub
docker push your-username/parakeet-transcription
```

### 2. Deploy on RunPod

1. Go to [RunPod](https://runpod.io) and create a new serverless endpoint
2. Use your Docker image: `your-username/parakeet-transcription`
3. Set the following configuration:
   - **Container Disk**: 20GB minimum
   - **Memory**: 16GB minimum
   - **GPU**: RTX 4090, A100, or similar (minimum 16GB VRAM)
   - **Max Workers**: Based on your needs
   - **Idle Timeout**: 300 seconds (5 minutes)

### 3. Environment Variables

Set these environment variables in your RunPod serverless endpoint configuration:

| Variable | Description | Required |
|----------|-------------|----------|
| `RUNPOD_API_KEY` | Your RunPod API key | Optional |
| `RUNPOD_ENDPOINT_ID` | Your endpoint ID for logging | Optional |
| `RUNPOD_ENDPOINT_URL` | Full endpoint URL for client usage | Optional |

**To set environment variables in RunPod:**
1. Go to your serverless endpoint settings
2. Click "Environment Variables" 
3. Add the variables above
4. Deploy/redeploy your endpoint

## Performance Considerations

- **Model Loading**: The model takes ~30-60 seconds to load on first request
- **Processing Speed**: ~3380x real-time factor (RTFx) with batch size 128
- **Memory Requirements**: ~2GB RAM minimum for model loading
- **GPU Requirements**: NVIDIA GPU with 16GB+ VRAM recommended

## Audio Requirements

- **Sample Rate**: 16kHz (automatically resampled if different)
- **Channels**: Mono (automatically converted if stereo)
- **Supported Formats**: WAV, MP3, FLAC, M4A, OGG
- **Max Duration**: No limit (automatically chunked)

## Error Handling

The API returns detailed error messages for common issues:

- Invalid base64 audio data
- Unsupported audio format
- Model loading failures
- Processing timeouts

## Limitations

- English language only
- Accuracy varies with audio quality, accents, and domain
- Not recommended for incomplete sentences or single words
- Processing time scales with audio duration

## License

This project uses the NVIDIA Parakeet model under the CC-BY-4.0 license. See the [model page](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) for full license terms.

## Support

For issues and questions:
1. Check the logs in your RunPod dashboard
2. Verify your audio format and encoding
3. Ensure sufficient GPU memory is available
4. Review the model documentation on Hugging Face
