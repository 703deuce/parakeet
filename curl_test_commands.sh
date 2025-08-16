#!/bin/bash

# Set your RunPod API key and endpoint
export RUNPOD_API_KEY="your_api_key_here"
export RUNPOD_ENDPOINT="7u304yobo6ytm9"

# Base64 encode your audio file (replace with your audio file path)
AUDIO_BASE64=$(base64 -w 0 your_audio_file.wav)

echo "Testing Parakeet API with Diarization Support"
echo "============================================="

# Test 1: Regular Transcription (existing functionality)
echo "🎯 Test 1: Regular Transcription"
curl -X POST "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": {
      \"audio_data\": \"${AUDIO_BASE64}\",
      \"audio_format\": \"wav\",
      \"include_timestamps\": true,
      \"use_diarization\": false
    }
  }" | jq '.'

echo -e "\n\n"

# Test 2: Speaker Diarization (auto-detect speakers)
echo "👥 Test 2: Speaker Diarization (Auto-detect)"
curl -X POST "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": {
      \"audio_data\": \"${AUDIO_BASE64}\",
      \"audio_format\": \"wav\",
      \"include_timestamps\": true,
      \"use_diarization\": true
    }
  }" | jq '.'

echo -e "\n\n"

# Test 3: Speaker Diarization (specify 2 speakers)
echo "🎭 Test 3: Speaker Diarization (2 speakers)"
curl -X POST "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": {
      \"audio_data\": \"${AUDIO_BASE64}\",
      \"audio_format\": \"wav\",
      \"include_timestamps\": true,
      \"use_diarization\": true,
      \"num_speakers\": 2
    }
  }" | jq '.'

echo "🎉 All tests completed!"
