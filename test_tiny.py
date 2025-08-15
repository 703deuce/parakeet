#!/usr/bin/env python3
"""
Test with minimal audio data
"""

import requests
import os
import base64

def test_minimal_audio():
    """Test with minimal audio data"""
    
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        print("Error: RUNPOD_API_KEY environment variable not set")
        return
    
    endpoint_url = "https://api.runpod.ai/v2/7u304yobo6ytm9/run"
    
    print(f"Testing minimal audio to: {endpoint_url}")
    
    # Create minimal audio data (just a few bytes)
    minimal_audio = b"RIFF" + b"\x00" * 40  # Minimal WAV header
    audio_b64 = base64.b64encode(minimal_audio).decode('utf-8')
    
    payload = {
        "input": {
            "audio_data": audio_b64,
            "audio_format": "wav",
            "include_timestamps": False,
            "chunk_duration": 1200
        }
    }
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    try:
        print(f"Audio data size: {len(audio_b64)} characters")
        print("Sending minimal audio request...")
        
        response = requests.post(endpoint_url, json=payload, headers=headers, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("\n✅ Minimal audio request successful!")
        else:
            print(f"\n❌ Failed with status {response.status_code}")
            
    except Exception as e:
        print(f"\n💥 Error: {e}")

if __name__ == "__main__":
    test_minimal_audio()
