#!/usr/bin/env python3
"""
Test script for the Parakeet Transcription API
"""

import base64
import json
import requests
import argparse
import os
from pathlib import Path

def encode_audio_file(file_path):
    """Encode audio file to base64"""
    try:
        with open(file_path, 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        return audio_data
    except Exception as e:
        print(f"Error encoding audio file: {e}")
        return None

def test_transcription(api_endpoint, audio_file_path, include_timestamps=False, chunk_duration=1200):
    """Test the transcription API"""
    
    # Check if audio file exists
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file '{audio_file_path}' not found")
        return None
    
    # Get file extension for format
    file_ext = Path(audio_file_path).suffix.lower().lstrip('.')
    if file_ext not in ['wav', 'mp3', 'flac', 'm4a', 'ogg']:
        print(f"Warning: Unsupported file format '{file_ext}', trying anyway...")
    
    print(f"Encoding audio file: {audio_file_path}")
    audio_data = encode_audio_file(audio_file_path)
    
    if not audio_data:
        return None
    
    # Prepare request payload (exactly matching handler.py expected format)
    payload = {
        "input": {
            "audio_data": audio_data,
            "audio_format": file_ext or "wav", 
            "include_timestamps": include_timestamps,
            "chunk_duration": chunk_duration
        }
    }
    
    print(f"Sending request to: {api_endpoint}")
    print(f"Include timestamps: {include_timestamps}")
    print(f"Chunk duration: {chunk_duration} seconds")
    
    try:
        # Get API key from environment
        api_key = os.getenv('RUNPOD_API_KEY')
        if not api_key:
            print("Error: RUNPOD_API_KEY environment variable not set")
            return None
        
        # Set headers with API key
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Send request
        print(f"Payload size: {len(str(payload))} characters")
        print(f"Audio data size: {len(audio_data)} characters")
        
        response = requests.post(api_endpoint, json=payload, headers=headers, timeout=300)  # 5 minute timeout
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response body: {result}")
            return result
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("Error: Request timed out (5 minutes)")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None

def print_results(result):
    """Print transcription results in a formatted way"""
    if not result:
        return
    
    print("\n" + "="*50)
    print("TRANSCRIPTION RESULTS")
    print("="*50)
    
    print(f"\nText: {result.get('text', 'N/A')}")
    
    if 'audio_duration_seconds' in result:
        duration = result['audio_duration_seconds']
        minutes = int(duration // 60)
        seconds = duration % 60
        print(f"Audio Duration: {minutes}m {seconds:.1f}s")
    
    if 'chunks_processed' in result:
        print(f"Chunks Processed: {result['chunks_processed']}")
    
    if 'model_used' in result:
        print(f"Model Used: {result['model_used']}")
    
    # Print timestamps if available
    if result.get('word_timestamps'):
        print(f"\nWord Timestamps: {len(result['word_timestamps'])} words")
        print("First 5 word timestamps:")
        for i, word_ts in enumerate(result['word_timestamps'][:5]):
            print(f"  {word_ts.get('word', 'N/A')}: {word_ts.get('start', 0):.2f}s - {word_ts.get('end', 0):.2f}s")
        if len(result['word_timestamps']) > 5:
            print("  ...")
    
    if result.get('segment_timestamps'):
        print(f"\nSegment Timestamps: {len(result['segment_timestamps'])} segments")
        print("First 3 segment timestamps:")
        for i, seg_ts in enumerate(result['segment_timestamps'][:3]):
            print(f"  {seg_ts.get('start', 0):.2f}s - {seg_ts.get('end', 0):.2f}s: {seg_ts.get('segment', 'N/A')}")
        if len(result['segment_timestamps']) > 3:
            print("  ...")

def main():
    parser = argparse.ArgumentParser(description='Test Parakeet Transcription API')
    parser.add_argument('audio_file', help='Path to audio file to transcribe')
    parser.add_argument('--api-endpoint', help='API endpoint URL (uses RUNPOD_ENDPOINT_URL env var if not provided)')
    parser.add_argument('--timestamps', action='store_true', help='Include timestamps in response')
    parser.add_argument('--chunk-duration', type=int, default=1200, help='Chunk duration in seconds (default: 1200)')
    parser.add_argument('--save-json', help='Save full response to JSON file')
    
    args = parser.parse_args()
    
    # Get API endpoint from args or environment variable
    api_endpoint = args.api_endpoint or os.getenv('RUNPOD_ENDPOINT_URL')
    if not api_endpoint:
        print("Error: API endpoint must be provided via --api-endpoint or RUNPOD_ENDPOINT_URL environment variable")
        return 1
    
    # If just endpoint ID is provided, construct full URL
    if not api_endpoint.startswith('http'):
        api_endpoint = f"https://api.runpod.ai/v2/{api_endpoint}/run"
        print(f"Constructed full URL: {api_endpoint}")
    
    # Test the API
    result = test_transcription(
        api_endpoint,
        args.audio_file,
        args.timestamps,
        args.chunk_duration
    )
    
    if result:
        print_results(result)
        
        # Save to JSON if requested
        if args.save_json:
            with open(args.save_json, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nFull response saved to: {args.save_json}")
    else:
        print("Transcription failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
