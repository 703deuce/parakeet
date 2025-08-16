#!/usr/bin/env python3
"""
Test script for the enhanced Parakeet API with speaker diarization
"""

import requests
import base64
import json
import os
import argparse

def test_transcription_api(audio_file, api_endpoint, use_diarization=False, num_speakers=None, include_timestamps=True):
    """Test the transcription API with optional diarization"""
    
    # Get API key
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        print("❌ RUNPOD_API_KEY environment variable not set")
        return None
    
    # Encode audio file
    print(f"🎵 Encoding audio file: {audio_file}")
    try:
        with open(audio_file, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ Error encoding audio: {e}")
        return None
    
    # Determine audio format
    audio_format = audio_file.split('.')[-1].lower()
    
    # Prepare payload
    payload = {
        "input": {
            "audio_data": audio_data,
            "audio_format": audio_format,
            "include_timestamps": include_timestamps,
            "use_diarization": use_diarization,
            "chunk_duration": 300
        }
    }
    
    # Add num_speakers if specified
    if num_speakers:
        payload["input"]["num_speakers"] = num_speakers
    
    # Prepare headers
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # Construct full URL if needed
    if not api_endpoint.startswith('http'):
        api_endpoint = f"https://api.runpod.ai/v2/{api_endpoint}/run"
    
    print(f"🚀 Sending request to: {api_endpoint}")
    print(f"📊 Mode: {'Diarization + Transcription' if use_diarization else 'Regular Transcription'}")
    if use_diarization and num_speakers:
        print(f"👥 Expected speakers: {num_speakers}")
    
    try:
        response = requests.post(api_endpoint, json=payload, headers=headers, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Request error: {e}")
        return None

def print_regular_results(result):
    """Print regular transcription results"""
    print("\n" + "="*60)
    print("📝 REGULAR TRANSCRIPTION RESULTS")
    print("="*60)
    
    print(f"\n🎯 Text: {result.get('text', 'N/A')}")
    
    if 'audio_duration_seconds' in result:
        duration = result['audio_duration_seconds']
        minutes = int(duration // 60)
        seconds = duration % 60
        print(f"⏱️  Duration: {minutes}m {seconds:.1f}s")
    
    print(f"📊 Chunks: {result.get('chunks_processed', 0)}")
    print(f"🤖 Model: {result.get('model_used', 'N/A')}")
    print(f"🔧 Method: {result.get('processing_method', 'N/A')}")
    
    if result.get('word_timestamps'):
        print(f"\n🔤 Words with timestamps: {len(result['word_timestamps'])}")

def print_diarized_results(result):
    """Print diarized transcription results"""
    print("\n" + "="*60)
    print("👥 SPEAKER DIARIZATION RESULTS")
    print("="*60)
    
    if 'audio_duration_seconds' in result:
        duration = result['audio_duration_seconds']
        minutes = int(duration // 60)
        seconds = duration % 60
        print(f"⏱️  Duration: {minutes}m {seconds:.1f}s")
    
    print(f"👤 Speakers detected: {result.get('speakers_detected', 0)}")
    print(f"📊 Segments: {result.get('segments_processed', 0)}")
    print(f"🤖 ASR Model: {result.get('model_used', 'N/A')}")
    print(f"🎭 Diarization Model: {result.get('diarization_model', 'N/A')}")
    
    if 'merged_text' in result:
        print(f"\n📝 Complete transcript:")
        print(f"{result['merged_text']}")
    
    if 'diarized_transcript' in result:
        print(f"\n👥 Speaker-by-speaker breakdown:")
        print("-" * 40)
        
        for i, segment in enumerate(result['diarized_transcript']):
            speaker = segment.get('speaker', 'Unknown')
            start_time = segment.get('start_time', 0)
            end_time = segment.get('end_time', 0)
            text = segment.get('text', '')
            
            print(f"\n🗣️  {speaker} ({start_time:.1f}s - {end_time:.1f}s):")
            print(f"   {text}")

def main():
    parser = argparse.ArgumentParser(description='Test Parakeet API with Diarization')
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--endpoint', default='7u304yobo6ytm9', help='RunPod endpoint ID or full URL')
    parser.add_argument('--diarization', action='store_true', help='Enable speaker diarization')
    parser.add_argument('--speakers', type=int, help='Expected number of speakers')
    parser.add_argument('--no-timestamps', action='store_true', help='Disable word timestamps')
    parser.add_argument('--save-json', help='Save response to JSON file')
    
    args = parser.parse_args()
    
    print("🎤 PARAKEET API TESTER WITH DIARIZATION")
    print("=" * 60)
    
    # Test the API
    result = test_transcription_api(
        audio_file=args.audio_file,
        api_endpoint=args.endpoint,
        use_diarization=args.diarization,
        num_speakers=args.speakers,
        include_timestamps=not args.no_timestamps
    )
    
    if result:
        # Check if it's a regular transcription or diarized result
        if 'diarized_transcript' in result:
            print_diarized_results(result)
        else:
            print_regular_results(result)
        
        # Save to JSON if requested
        if args.save_json:
            with open(args.save_json, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Results saved to: {args.save_json}")
    else:
        print("\n❌ Transcription failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
