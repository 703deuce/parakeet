#!/usr/bin/env python3
"""
Fix the test.wav file to make it properly decodable
"""

import soundfile as sf
import numpy as np
import os
from pydub import AudioSegment
from pydub.utils import which

def fix_audio_file(input_file, output_file):
    """Fix audio file by re-encoding it properly"""
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        return False
    
    print(f"Fixing audio file: {input_file}")
    print(f"Original file size: {os.path.getsize(input_file)} bytes")
    
    try:
        # Method 1: Try with pydub first (handles more formats)
        print("Attempting to load with pydub...")
        audio = AudioSegment.from_file(input_file)
        
        # Convert to standard format
        # - 16kHz sample rate (good for speech recognition)
        # - Mono channel
        # - 16-bit PCM
        audio = audio.set_frame_rate(16000)
        audio = audio.set_channels(1)  # Mono
        audio = audio.set_sample_width(2)  # 16-bit
        
        # Export as WAV
        audio.export(output_file, format="wav")
        
        print(f"✅ Successfully fixed with pydub")
        print(f"New file size: {os.path.getsize(output_file)} bytes")
        
        # Verify the fixed file can be read by soundfile
        try:
            data, samplerate = sf.read(output_file)
            duration = len(data) / samplerate
            print(f"✅ Verification: Duration = {duration:.2f}s, Sample rate = {samplerate}Hz")
            return True
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ pydub failed: {e}")
        
        # Method 2: Try with soundfile directly
        try:
            print("Attempting to load with soundfile...")
            data, samplerate = sf.read(input_file)
            
            # Resample to 16kHz if needed
            if samplerate != 16000:
                print(f"Resampling from {samplerate}Hz to 16000Hz...")
                # Simple resampling (for more complex audio, use librosa)
                ratio = 16000 / samplerate
                new_length = int(len(data) * ratio)
                data = np.interp(np.linspace(0, len(data), new_length), np.arange(len(data)), data)
                samplerate = 16000
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                print("Converting to mono...")
                data = np.mean(data, axis=1)
            
            # Save fixed file
            sf.write(output_file, data, samplerate)
            
            print(f"✅ Successfully fixed with soundfile")
            print(f"New file size: {os.path.getsize(output_file)} bytes")
            return True
            
        except Exception as e2:
            print(f"❌ soundfile also failed: {e2}")
            return False

def main():
    input_file = "test.wav"
    output_file = "test_fixed.wav"
    
    # Check if ffmpeg is available (needed by pydub for some formats)
    if which("ffmpeg") is None:
        print("Warning: ffmpeg not found. Some audio formats may not be supported.")
    
    if fix_audio_file(input_file, output_file):
        print(f"\n🎉 Audio file fixed successfully!")
        print(f"Original: {input_file}")
        print(f"Fixed: {output_file}")
        
        # Replace original with fixed version
        if input("Replace original file? (y/n): ").lower() == 'y':
            os.replace(output_file, input_file)
            print(f"✅ Replaced {input_file} with fixed version")
        else:
            print(f"✅ Fixed file saved as {output_file}")
    else:
        print(f"\n❌ Failed to fix audio file")

if __name__ == "__main__":
    main()
