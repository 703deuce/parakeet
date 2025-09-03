# 🧪 Test Setup Guide for Complete Workflow

This guide will help you test the complete Firebase URL → RunPod → Chunked Diarization → Speaker Consistency workflow.

## 📋 Prerequisites

1. **test.mp3** file in the current directory
2. **RunPod API endpoint** with the updated handler
3. **RunPod API key**
4. **HuggingFace token** for pyannote models

## 🔧 Setup Steps

### Step 1: Update Configuration

Edit `config.py` with your actual credentials:

```python
# RunPod Configuration
RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/abc123def/run"  # Your actual endpoint
API_KEY = "abc123def456ghi789"  # Your actual API key

# HuggingFace Configuration  
HF_TOKEN = "hf_abc123def456ghi789"  # Your actual HF token
```

### Step 2: Verify test.mp3

Make sure `test.mp3` exists in the current directory:

```bash
ls -la test.mp3
```

### Step 3: Install Dependencies

```bash
pip install requests pathlib
```

## 🚀 Running the Test

### Basic Test

```bash
python test_complete_workflow.py
```

### Test with Custom Audio File

```bash
# Edit config.py to change AUDIO_FILE
AUDIO_FILE = "your_audio.mp3"
```

## 📊 What the Test Does

1. **Upload/URL**: Gets audio URL (local file path for testing)
2. **RunPod**: Sends URL to RunPod with diarization enabled
3. **Chunking**: Automatically chunks audio > 15 minutes
4. **Diarization**: Runs pyannote on each chunk individually
5. **Speaker Matching**: Uses voice embeddings to match speakers across chunks
6. **Merging**: Combines all results with consistent speaker IDs
7. **Output**: Saves complete transcript with timestamps and speaker segments

## 🔍 Expected Results

### For Single Speaker Audio:
- **Speaker consistency**: Same person = same ID across all chunks
- **Final result**: 1 speaker (Speaker_00) throughout entire transcript
- **Processing method**: `chunked_with_consistent_diarization`

### For Multi-Speaker Audio:
- **Speaker consistency**: Each person gets consistent ID across chunks
- **Speaker count**: Actual number of speakers detected
- **Processing method**: `chunked_with_consistent_diarization`

## 📁 Output Files

The test creates two output files:

1. **`transcript_test_YYYYMMDD_HHMMSS.json`** - Complete API response
2. **`transcript_test_YYYYMMDD_HHMMSS_summary.txt`** - Human-readable summary

## 🐛 Troubleshooting

### Common Issues:

1. **"test.mp3 not found"**
   - Make sure test.mp3 is in the current directory
   - Check file permissions

2. **"Please update RUNPOD_ENDPOINT"**
   - Edit config.py with your actual endpoint URL

3. **"Please update API_KEY"**
   - Edit config.py with your actual RunPod API key

4. **"Please update HF_TOKEN"**
   - Edit config.py with your actual HuggingFace token

5. **Job timeout**
   - Increase MAX_WAIT_MINUTES in config.py
   - Check RunPod endpoint status

### Debug Mode:

The script provides detailed logging for each step. Look for:
- 🔍 Raw API responses
- 📊 Job status updates
- ✅ Success confirmations
- ❌ Error messages

## 🎯 Testing Different Scenarios

### Test Single Speaker:
```python
SINGLE_SPEAKER_MODE = True
SPEAKER_THRESHOLD = 0.35
```

### Test Multi-Speaker:
```python
SINGLE_SPEAKER_MODE = False
SPEAKER_THRESHOLD = 0.35
```

### Test Long Audio (>15 minutes):
- Use a long audio file
- Verify chunking is triggered
- Check speaker consistency across chunks

### Test Short Audio (<15 minutes):
- Use a short audio file
- Verify no chunking is used
- Check direct processing

## 🔄 Workflow Verification

The test validates the complete pipeline:

1. ✅ **Audio detection** - Duration and format
2. ✅ **Chunking decision** - >15 minutes triggers chunking
3. ✅ **Chunked transcription** - Each chunk processed individually
4. ✅ **Chunked diarization** - Pyannote runs on each chunk
5. ✅ **Speaker consistency** - Voice embeddings match speakers across chunks
6. ✅ **Result merging** - All chunks combined with consistent timestamps
7. ✅ **Output generation** - Complete transcript with metadata

## 🎉 Success Indicators

- **Processing method**: Contains "chunked" or "consistent_diarization"
- **Speaker count**: Reasonable number (1 for single speaker)
- **Chunks processed**: >1 for long audio files
- **Workflow**: Shows "chunked_transcription_with_consistent_diarization"
- **No errors**: Clean execution from start to finish

## 📞 Support

If you encounter issues:

1. Check the detailed error messages in the console
2. Verify your credentials in config.py
3. Ensure test.mp3 exists and is accessible
4. Check RunPod endpoint status and logs
5. Verify HuggingFace token has pyannote access

---

**Happy Testing! 🚀**
