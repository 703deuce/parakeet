#!/usr/bin/env python3
"""
Test MP3 file directly without conversion
"""

import requests
import os
import base64
import json

def test_mp3_direct():
    """Test MP3 file directly"""
    
    # Get API key
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    input_file = "test2.mp3"
    
    if not os.path.exists(input_file):
        print(f"❌ {input_file} not found")
        return
    
    print(f"Testing MP3 directly: {input_file}")
    print(f"File size: {os.path.getsize(input_file)} bytes ({os.path.getsize(input_file)/(1024*1024):.1f}MB)")
    
    # Check if file is too large for direct upload
    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    if file_size_mb > 10:
        print(f"⚠️  File is {file_size_mb:.1f}MB - may timeout during upload")
        print("Consider using chunked approach instead")
        return
    
    try:
        # Encode MP3 file
        print("Encoding MP3 to base64...")
        with open(input_file, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
        
        print(f"Base64 size: {len(audio_data)} characters")
        
        # Create payload for MP3
        payload = {
            "input": {
                "audio_data": audio_data,
                "audio_format": "mp3",  # Specify MP3 format
                "include_timestamps": True,
                "chunk_duration": 1200
            }
        }
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        endpoint_url = "https://api.runpod.ai/v2/7u304yobo6ytm9/run"
        
        print("Sending MP3 request to RunPod...")
        response = requests.post(endpoint_url, json=payload, headers=headers, timeout=120)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get('id')
            print(f"✅ Job queued: {job_id}")
            
            # Save result
            with open("mp3_direct_result.json", 'w') as f:
                json.dump(result, f, indent=2)
            print("✅ Result saved to mp3_direct_result.json")
            
            return job_id
        else:
            print(f"❌ Request failed")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    test_mp3_direct()
