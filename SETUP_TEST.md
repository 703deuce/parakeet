# Firebase Pipeline Test Setup Guide
====================================

## 🚀 Quick Setup

### 1. Update Configuration
Edit `test_config.py` with your actual values:

```python
# RunPod Configuration
RUNPOD_ENDPOINT = "https://your-actual-endpoint.proxy.runpod.net"
API_KEY = "your-actual-api-key"

# HuggingFace Configuration  
HF_TOKEN = "hf_your-actual-token"
```

### 2. Verify Test Files
Ensure these files exist in your directory:
- `test_clean.wav` - Small file for basic testing
- `test2.wav` - Large file (>10MB) for Firebase auto-testing

### 3. Run the Test
```bash
python test_full_firebase_pipeline.py
```

## 🔧 What the Test Does

### Test 1: Small File + Firebase Forced
- Tests small file with `firebase_upload: true`
- Verifies Firebase workflow works for any file size
- Expected: Firebase upload → download → process → return results

### Test 2: Large File + Firebase Auto
- Tests large file (>10MB) without explicit Firebase flag
- Verifies automatic Firebase detection and workflow
- Expected: Auto Firebase upload → download → process → return results

### Test 3: Firebase + Streaming Mode
- Tests Firebase workflow with Parakeet v3 streaming enabled
- Verifies streaming configuration works with Firebase
- Expected: Firebase + streaming mode → process → return results

## 🎯 Expected Results

If everything works correctly, you should see:
```
✅ SUCCESS! Firebase pipeline completed
🎯 Workflow: direct_firebase_no_chunking
🔥 Firebase used: True
📝 Text length: [number] characters
🎤 Diarization: [number] segments
```

## ❌ Common Issues

### 1. Configuration Errors
- **RUNPOD_ENDPOINT**: Must be your actual RunPod serverless endpoint URL
- **API_KEY**: Must be your actual RunPod API key from the dashboard
- **HF_TOKEN**: Must be your actual HuggingFace token (hf_xxx format)

### 2. File Issues
- **Missing test files**: Ensure `test_clean.wav` and `test2.wav` exist
- **File permissions**: Ensure files are readable

### 3. Network Issues
- **Timeout errors**: Increase timeout values in config if needed
- **Connection errors**: Verify RunPod endpoint is running and accessible

### 4. Firebase Issues
- **Firebase config**: Ensure Firebase configuration is correct in handler.py
- **Storage permissions**: Verify Firebase Storage bucket permissions

## 🔍 Debugging

### Check RunPod Logs
Look at your RunPod endpoint logs for detailed error information.

### Test Firebase Separately
Use `test_firebase_storage.py` to verify Firebase Storage is working.

### Verify Endpoint
Test basic connectivity to your RunPod endpoint first.

## 📚 Next Steps

After successful testing:
1. Deploy the working handler.py to RunPod
2. Integrate with your Next.js SaaS application
3. Monitor performance and adjust timeouts as needed
4. Scale based on usage patterns

## 🆘 Need Help?

If tests fail:
1. Check RunPod endpoint logs
2. Verify all configuration values
3. Test Firebase Storage separately
4. Ensure test files are valid audio files
