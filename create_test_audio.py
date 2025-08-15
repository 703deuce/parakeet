#!/usr/bin/env python3
"""
Create a proper test audio file that can be decoded
"""

import numpy as np
import soundfile as sf
import os

def create_test_audio():
    """Create a simple test audio file"""
    
    # Generate a 5-second sine wave at 440Hz (A note)
    sample_rate = 16000  # 16kHz sample rate (good for speech)
    duration = 5  # 5 seconds
    frequency = 440  # A note
    
    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate sine wave
    audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Save as WAV file
    output_file = "test_clean.wav"
    sf.write(output_file, audio_data, sample_rate)
    
    print(f"Created test audio file: {output_file}")
    print(f"Duration: {duration} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"File size: {os.path.getsize(output_file)} bytes")
    
    return output_file

if __name__ == "__main__":
    create_test_audio()
