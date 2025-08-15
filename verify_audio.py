#!/usr/bin/env python3
"""
Verify that audio files can be properly read
"""

import soundfile as sf
import os
from pydub import AudioSegment

def verify_audio_file(file_path):
    """Verify that an audio file can be read properly"""
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"Verifying: {file_path}")
    print(f"File size: {os.path.getsize(file_path)} bytes")
    
    # Test 1: Try with soundfile
    try:
        data, samplerate = sf.read(file_path)
        duration = len(data) / samplerate
        channels = 1 if len(data.shape) == 1 else data.shape[1]
        
        print(f"✅ soundfile: Duration = {duration:.2f}s, Sample rate = {samplerate}Hz, Channels = {channels}")
        
        # Check for common issues
        if samplerate < 8000:
            print("⚠️  Warning: Very low sample rate, may affect transcription quality")
        if duration < 0.1:
            print("⚠️  Warning: Very short audio file")
        if duration > 1800:  # 30 minutes
            print("⚠️  Warning: Very long audio file, will be chunked")
            
    except Exception as e:
        print(f"❌ soundfile failed: {e}")
        return False
    
    # Test 2: Try with pydub
    try:
        audio = AudioSegment.from_file(file_path)
        duration_ms = len(audio)
        sample_rate = audio.frame_rate
        channels = audio.channels
        
        print(f"✅ pydub: Duration = {duration_ms/1000:.2f}s, Sample rate = {sample_rate}Hz, Channels = {channels}")
        
    except Exception as e:
        print(f"❌ pydub failed: {e}")
        return False
    
    # Test 3: Check if it has actual audio content
    try:
        # Check if audio has any significant content (not just silence)
        max_amplitude = max(abs(data.min()), abs(data.max()))
        print(f"✅ Audio content: Max amplitude = {max_amplitude:.4f}")
        
        if max_amplitude < 0.001:
            print("⚠️  Warning: Audio appears to be very quiet or silent")
        
    except Exception as e:
        print(f"⚠️  Could not analyze audio content: {e}")
    
    print("✅ Audio file verification passed!")
    return True

def main():
    files_to_check = ["test.wav", "test_clean.wav", "test2.wav"]
    
    for file_path in files_to_check:
        print("=" * 60)
        if verify_audio_file(file_path):
            print(f"✅ {file_path} is ready for transcription")
        else:
            print(f"❌ {file_path} has issues and may not work for transcription")
        print()

if __name__ == "__main__":
    main()
