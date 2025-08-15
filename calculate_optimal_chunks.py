#!/usr/bin/env python3
"""
Calculate optimal chunk sizes to stay under RunPod's 10MiB limit
"""

import os
from pydub import AudioSegment

def calculate_optimal_chunk_size(input_file, target_mb=7.5):
    """Calculate optimal chunk duration to stay under target size"""
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return None
    
    print(f"Analyzing: {input_file}")
    
    # Load audio to get duration
    audio = AudioSegment.from_file(input_file)
    total_duration_seconds = len(audio) / 1000
    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    
    print(f"📊 File Analysis:")
    print(f"  Duration: {total_duration_seconds:.1f} seconds ({total_duration_seconds/60:.1f} minutes)")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Format: {input_file.split('.')[-1].upper()}")
    
    # Calculate MB per second
    mb_per_second = file_size_mb / total_duration_seconds
    print(f"  Data rate: {mb_per_second:.3f} MB/second")
    
    # Calculate optimal chunk duration to stay under target
    optimal_chunk_seconds = target_mb / mb_per_second
    optimal_chunk_minutes = optimal_chunk_seconds / 60
    
    print(f"\n🎯 Optimal Chunk Size (to stay under {target_mb}MB):")
    print(f"  Chunk duration: {optimal_chunk_seconds:.0f} seconds ({optimal_chunk_minutes:.1f} minutes)")
    
    # Calculate how many chunks this would create
    num_chunks = total_duration_seconds / optimal_chunk_seconds
    print(f"  Number of chunks: {num_chunks:.1f}")
    
    # Show different chunk size options
    print(f"\n📋 Chunk Size Options:")
    
    chunk_options = [60, 120, 180, 240, 300, 360, 480, 600]  # 1-10 minutes
    
    for chunk_sec in chunk_options:
        chunk_min = chunk_sec / 60
        estimated_size_mb = chunk_sec * mb_per_second
        base64_size_mb = estimated_size_mb * 1.33  # Base64 overhead
        num_chunks = total_duration_seconds / chunk_sec
        
        status = "✅" if base64_size_mb < 10 else "❌"
        
        print(f"  {status} {chunk_min:2.0f}min chunks: ~{estimated_size_mb:.1f}MB → ~{base64_size_mb:.1f}MB base64 ({num_chunks:.1f} chunks)")
    
    # Recommend the largest safe chunk size
    safe_chunk_seconds = None
    for chunk_sec in reversed(chunk_options):
        estimated_size_mb = chunk_sec * mb_per_second
        base64_size_mb = estimated_size_mb * 1.33
        if base64_size_mb < 9.5:  # Leave some safety margin
            safe_chunk_seconds = chunk_sec
            break
    
    if safe_chunk_seconds:
        print(f"\n🏆 RECOMMENDED: {safe_chunk_seconds/60:.0f}-minute chunks")
        print(f"   This will create {total_duration_seconds/safe_chunk_seconds:.1f} chunks")
        print(f"   Each chunk: ~{safe_chunk_seconds * mb_per_second:.1f}MB → ~{safe_chunk_seconds * mb_per_second * 1.33:.1f}MB base64")
    
    return safe_chunk_seconds

def main():
    files_to_analyze = ["test2.mp3", "test2.wav", "test.wav"]
    
    print("🔍 OPTIMAL CHUNK SIZE CALCULATOR")
    print("=" * 60)
    print("Target: Stay under 10MiB RunPod limit after base64 encoding")
    print()
    
    for file_path in files_to_analyze:
        if os.path.exists(file_path):
            print("=" * 60)
            optimal_seconds = calculate_optimal_chunk_size(file_path)
            print()
        else:
            print(f"⚠️  {file_path} not found, skipping...")

if __name__ == "__main__":
    main()
