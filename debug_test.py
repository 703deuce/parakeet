#!/usr/bin/env python3
"""
Debug version to see exactly what's happening with requests
"""

import requests
import os
import base64
import json

def debug_test():
    """Debug test to see what's happening"""
    
    # Get API key
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    print(f"✅ API Key: {api_key[:10]}...")
    
    # Test endpoint
    endpoint_url = "https://api.runpod.ai/v2/7u304yobo6ytm9/run"
    print(f"✅ Endpoint: {endpoint_url}")
    
    # Test with minimal payload first
    print("\n=== Testing minimal payload ===")
    minimal_payload = {
        "input": {
            "test": "debug_test"
        }
    }
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    try:
        print("Sending minimal request...")
        response = requests.post(endpoint_url, json=minimal_payload, headers=headers, timeout=30)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Job ID: {result.get('id', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Minimal test failed: {e}")
        return
    
    # Test with small audio file
    print("\n=== Testing with small audio ===")
    
    if not os.path.exists("test_clean.wav"):
        print("❌ test_clean.wav not found")
        return
    
    try:
        # Read and encode small audio file
        with open("test_clean.wav", 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
        
        print(f"Audio data size: {len(audio_data)} characters")
        
        audio_payload = {
            "input": {
                "audio_data": audio_data,
                "audio_format": "wav",
                "include_timestamps": False,
                "chunk_duration": 1200
            }
        }
        
        print("Sending audio request...")
        response = requests.post(endpoint_url, json=audio_payload, headers=headers, timeout=60)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Job ID: {result.get('id', 'N/A')}")
            
            # Save result
            with open("debug_audio_result.json", 'w') as f:
                json.dump(result, f, indent=2)
            print("✅ Result saved to debug_audio_result.json")
        
    except Exception as e:
        print(f"❌ Audio test failed: {e}")

if __name__ == "__main__":
    debug_test()
