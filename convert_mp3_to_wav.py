#!/usr/bin/env python3
"""
Convert MP3 to properly formatted WAV for transcription
"""

from pydub import AudioSegment
import os

def convert_mp3_to_wav(input_file, output_file):
    """Convert MP3 to WAV with optimal settings for speech recognition"""
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return False
    
    print(f"Converting: {input_file}")
    print(f"Input file size: {os.path.getsize(input_file)} bytes")
    
    try:
        # Load the MP3 file
        print("Loading MP3 file...")
        audio = AudioSegment.from_mp3(input_file)
        
        print(f"Original format:")
        print(f"  Duration: {len(audio)/1000:.2f} seconds")
        print(f"  Sample rate: {audio.frame_rate} Hz")
        print(f"  Channels: {audio.channels}")
        print(f"  Sample width: {audio.sample_width} bytes")
        
        # Convert to optimal format for speech recognition
        print("Converting to optimal format...")
        
        # Set to 16kHz sample rate (optimal for speech)
        audio = audio.set_frame_rate(16000)
        
        # Convert to mono
        audio = audio.set_channels(1)
        
        # Set to 16-bit (2 bytes per sample)
        audio = audio.set_sample_width(2)
        
        # Export as WAV
        print(f"Exporting to: {output_file}")
        audio.export(output_file, format="wav")
        
        print(f"✅ Conversion successful!")
        print(f"Output file size: {os.path.getsize(output_file)} bytes")
        
        # Verify the output
        print("Verifying output...")
        verify_audio = AudioSegment.from_wav(output_file)
        print(f"Final format:")
        print(f"  Duration: {len(verify_audio)/1000:.2f} seconds")
        print(f"  Sample rate: {verify_audio.frame_rate} Hz")
        print(f"  Channels: {verify_audio.channels}")
        print(f"  Sample width: {verify_audio.sample_width} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def main():
    input_file = "test2.mp3"
    output_file = "test2.wav"
    
    if convert_mp3_to_wav(input_file, output_file):
        print(f"\n🎉 Successfully converted {input_file} to {output_file}")
        print(f"The WAV file is now ready for transcription!")
    else:
        print(f"\n❌ Failed to convert {input_file}")

if __name__ == "__main__":
    main()
