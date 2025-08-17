# 🔥 Firebase URL Workflow - No Base64, No Size Limits!

## 🎯 What This Solves

**Problem**: RunPod API has a 10MiB limit on request body size, preventing large audio file processing.

**Solution**: Send ONLY a Firebase Storage URL to RunPod, which then downloads and processes the file locally.

## 🚀 Complete Workflow

```
Client → Firebase Storage (upload) → Get URL → RunPod API (URL only) → Download → Process → Results
```

### Step-by-Step:

1. **Upload to Firebase Storage** (from your client)
2. **Get Firebase download URL**
3. **Send URL to RunPod API** (tiny payload, no base64)
4. **RunPod downloads file** from Firebase URL
5. **Process locally** with Parakeet + Pyannote
6. **Return results**

## 📋 API Call Format

### With Diarization:
```json
{
  "input": {
    "audio_url": "https://firebasestorage.googleapis.com/v0/b/your-bucket/o/audio.wav?alt=media",
    "audio_format": "wav",
    "include_timestamps": true,
    "use_diarization": true,
    "num_speakers": 2,
    "hf_token": "hf_your_token_here"
  }
}
```

### Transcription Only:
```json
{
  "input": {
    "audio_url": "https://firebasestorage.googleapis.com/v0/b/your-bucket/o/audio.wav?alt=media",
    "audio_format": "wav",
    "include_timestamps": true,
    "use_diarization": false
  }
}
```

## ✅ Advantages

- **No 10MiB limit** - process files of any size
- **No base64 encoding/decoding** - faster, more efficient
- **Tiny payload size** - just the URL string
- **Direct Firebase integration** - seamless workflow
- **Better performance** - no data transfer through API
- **Scalable** - works with any file size

## 🔧 Implementation Details

### Handler Changes:
- Added `audio_url` parameter support
- Downloads file from Firebase URL using existing `download_from_firebase()` function
- Processes locally with Parakeet + Pyannote
- No base64 handling needed

### New Functions:
- `process_downloaded_audio()` - Main processing for URL workflow
- `process_downloaded_audio_transcription_only()` - Transcription-only mode

## 📝 Testing

### 1. Update Configuration:
Ensure `test_config.py` has your actual API keys:
```python
API_KEY = "rpa_your_actual_key"
HF_TOKEN = "hf_your_actual_token"
```

### 2. Upload Test File:
Upload an audio file to Firebase Storage and get the download URL.

### 3. Update Test Script:
Replace the placeholder URL in `test_firebase_url_workflow.py`:
```python
firebase_url = "https://firebasestorage.googleapis.com/v0/b/your-bucket/o/your-file.wav?alt=media"
```

### 4. Run Test:
```bash
python test_firebase_url_workflow.py
```

## 🎯 Use Cases

### Perfect For:
- **Large audio files** (>10MB)
- **Batch processing** - upload multiple files, process URLs
- **SaaS applications** - client uploads to Firebase, sends URL to your API
- **Long recordings** - podcasts, meetings, interviews
- **High-quality audio** - lossless formats, high bitrates

### Not Needed For:
- **Small files** (<10MB) - can still use base64 if preferred
- **Real-time streaming** - this is for file processing

## 🔒 Security Considerations

- **Firebase Storage rules** - ensure proper access control
- **URL validation** - verify URLs are from your Firebase project
- **Authentication** - RunPod API key still required
- **File cleanup** - temporary files are automatically cleaned up

## 🚀 Deployment

### 1. Deploy Updated Handler:
The updated `handler.py` supports both workflows:
- **Legacy**: `audio_data` (base64, limited to 10MiB)
- **New**: `audio_url` (Firebase URL, no size limits)

### 2. Client Integration:
```javascript
// Example: Upload to Firebase, then send URL to RunPod
const uploadToFirebase = async (audioFile) => {
  // Upload to Firebase Storage
  const firebaseUrl = await uploadAudioToFirebase(audioFile);
  
  // Send URL to RunPod
  const response = await fetch('https://api.runpod.ai/v2/your-endpoint/run', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      input: {
        audio_url: firebaseUrl,
        audio_format: 'wav',
        use_diarization: true,
        hf_token: hfToken
      }
    })
  });
  
  return response.json();
};
```

## 🎉 Result

**You can now process audio files of ANY size without hitting the 10MiB API limit!**

The workflow is:
1. **Client → Firebase Storage** (direct upload)
2. **Client → RunPod API** (just the URL)
3. **RunPod → Firebase Storage** (download)
4. **RunPod → Process locally** (Parakeet + Pyannote)
5. **Return results**

No more base64 encoding, no more size limits, no more chunking needed! 🚀
