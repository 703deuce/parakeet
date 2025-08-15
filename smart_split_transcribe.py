#!/usr/bin/env python3
"""
Smart audio splitting that finds silence points for natural chunk boundaries
"""

import os
import requests
import base64
import json
import time
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def find_silence_split_points(audio, target_chunk_duration_ms, silence_thresh=-40, min_silence_len=500):
    """
    Find optimal split points near target duration that fall on silence
    
    Args:
        audio: AudioSegment object
        target_chunk_duration_ms: Target chunk length in milliseconds
        silence_thresh: Silence threshold in dBFS (default -40dB)
        min_silence_len: Minimum silence length in ms to consider (default 500ms)
    """
    
    print(f"🔍 Analyzing audio for silence points...")
    print(f"  Target chunk duration: {target_chunk_duration_ms/1000:.1f}s")
    print(f"  Silence threshold: {silence_thresh}dBFS")
    print(f"  Minimum silence length: {min_silence_len}ms")
    
    # Detect non-silent segments (which gives us the silent gaps between them)
    nonsilent_ranges = detect_nonsilent(
        audio, 
        min_silence_len=min_silence_len, 
        silence_thresh=silence_thresh
    )
    
    if not nonsilent_ranges:
        print("⚠️  No silence detected, falling back to time-based splitting")
        return list(range(0, len(audio), target_chunk_duration_ms))
    
    print(f"✅ Found {len(nonsilent_ranges)} speech segments")
    
    # Calculate silence points (gaps between speech)
    silence_points = []
    
    # Add start if there's initial silence
    if nonsilent_ranges[0][0] > 0:
        silence_points.append(nonsilent_ranges[0][0] // 2)  # Middle of initial silence
    
    # Add points between speech segments
    for i in range(len(nonsilent_ranges) - 1):
        end_of_current = nonsilent_ranges[i][1]
        start_of_next = nonsilent_ranges[i + 1][0]
        
        if start_of_next - end_of_current >= min_silence_len:
            # Middle of the silence gap
            silence_point = end_of_current + (start_of_next - end_of_current) // 2
            silence_points.append(silence_point)
    
    # Add end if there's final silence
    if nonsilent_ranges[-1][1] < len(audio):
        silence_points.append(nonsilent_ranges[-1][1] + (len(audio) - nonsilent_ranges[-1][1]) // 2)
    
    print(f"✅ Found {len(silence_points)} potential silence split points")
    
    # Now find the best split points near our target durations
    split_points = [0]  # Always start at beginning
    current_pos = 0
    
    while current_pos < len(audio):
        target_next_split = current_pos + target_chunk_duration_ms
        
        if target_next_split >= len(audio):
            # We're at the end
            break
        
        # Find the silence point closest to our target
        best_split = target_next_split
        min_distance = float('inf')
        
        for silence_point in silence_points:
            if silence_point > current_pos:  # Must be after current position
                distance = abs(silence_point - target_next_split)
                
                # Prefer splits that are within reasonable range of target
                max_deviation = target_chunk_duration_ms * 0.3  # Allow 30% deviation
                
                if distance < min_distance and distance <= max_deviation:
                    min_distance = distance
                    best_split = silence_point
        
        split_points.append(best_split)
        current_pos = best_split
        
        deviation_seconds = (best_split - target_next_split) / 1000
        print(f"  Split point: {best_split/1000:.1f}s (deviation: {deviation_seconds:+.1f}s)")
    
    # Ensure we end at the actual end of audio
    if split_points[-1] < len(audio):
        split_points.append(len(audio))
    
    return split_points

def smart_split_audio(input_file, target_chunk_duration=300, output_dir="smart_chunks"):
    """Split audio at silence points near target duration"""
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎵 Loading audio file: {input_file}")
    audio = AudioSegment.from_file(input_file)
    
    total_duration = len(audio) / 1000
    target_chunk_duration_ms = target_chunk_duration * 1000
    
    print(f"📊 Audio info:")
    print(f"  Total duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")
    print(f"  Target chunk duration: {target_chunk_duration}s")
    
    # Find smart split points
    split_points = find_silence_split_points(audio, target_chunk_duration_ms)
    
    print(f"\n📂 Creating {len(split_points)-1} smart chunks...")
    
    chunks = []
    
    for i in range(len(split_points) - 1):
        start_ms = split_points[i]
        end_ms = split_points[i + 1]
        
        chunk = audio[start_ms:end_ms]
        
        # Save chunk as MP3 for efficiency
        chunk_filename = f"smart_chunk_{i:03d}.mp3"
        chunk_path = os.path.join(output_dir, chunk_filename)
        
        # Export with optimal settings
        chunk.export(
            chunk_path,
            format="mp3",
            parameters=["-ar", "16000", "-ac", "1"]  # 16kHz, mono
        )
        
        chunk_info = {
            'file': chunk_path,
            'start_time': start_ms / 1000,
            'end_time': end_ms / 1000,
            'duration': (end_ms - start_ms) / 1000,
            'size_mb': os.path.getsize(chunk_path) / (1024 * 1024)
        }
        
        chunks.append(chunk_info)
        
        print(f"✅ Chunk {i+1}: {chunk_info['start_time']:.1f}s-{chunk_info['end_time']:.1f}s ({chunk_info['duration']:.1f}s, {chunk_info['size_mb']:.1f}MB)")
    
    print(f"\n🎉 Created {len(chunks)} smart chunks in {output_dir}/")
    return chunks

def transcribe_chunk(chunk_file, api_key, endpoint_url):
    """Transcribe a single audio chunk"""
    
    try:
        # Encode audio file
        with open(chunk_file, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Detect format from filename
        audio_format = "mp3" if chunk_file.lower().endswith('.mp3') else "wav"
        
        payload = {
            "input": {
                "audio_data": audio_data,
                "audio_format": audio_format,
                "include_timestamps": True,
                "chunk_duration": 1200
            }
        }
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        print(f"  📤 Uploading chunk ({len(audio_data)} chars)...")
        response = requests.post(endpoint_url, json=payload, headers=headers, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get('id')
            print(f"  ✅ Job queued: {job_id}")
            return job_id
        else:
            print(f"  ❌ Upload failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def wait_for_job_completion(job_id, api_key, endpoint_base):
    """Wait for a job to complete and return results"""
    
    status_url = f"{endpoint_base}/status/{job_id}"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    max_attempts = 30
    attempt = 0
    
    while attempt < max_attempts:
        try:
            response = requests.get(status_url, headers=headers)
            if response.status_code == 200:
                result = response.json()
                status = result.get('status')
                
                if status == 'COMPLETED':
                    return result
                elif status == 'FAILED':
                    print(f"  ❌ Job failed: {result.get('error', 'Unknown error')}")
                    return None
                else:
                    print(f"  ⏳ Status: {status}")
                    time.sleep(10)
            else:
                print(f"  ❌ Status check failed: {response.status_code}")
                time.sleep(10)
                
        except Exception as e:
            print(f"  ❌ Status check error: {e}")
            time.sleep(10)
        
        attempt += 1
    
    print(f"  ⏰ Timeout waiting for job completion")
    return None

def merge_transcriptions(chunk_results, chunk_info_list):
    """Merge transcription results from multiple chunks with smart timing"""
    
    merged_text = []
    merged_word_timestamps = []
    merged_segment_timestamps = []
    
    for i, (result, chunk_info) in enumerate(zip(chunk_results, chunk_info_list)):
        if not result or 'output' not in result:
            print(f"⚠️  Skipping chunk {i}: No valid transcription")
            continue
        
        output = result['output']
        text = output.get('text', '').strip()
        
        if text:
            merged_text.append(text)
            
            # Use actual chunk start time for timestamp adjustment
            start_offset = chunk_info['start_time']
            
            # Merge word timestamps
            for word_ts in output.get('word_timestamps', []):
                adjusted_word = word_ts.copy()
                adjusted_word['start'] += start_offset
                adjusted_word['end'] += start_offset
                merged_word_timestamps.append(adjusted_word)
            
            # Merge segment timestamps
            for seg_ts in output.get('segment_timestamps', []):
                adjusted_seg = seg_ts.copy()
                adjusted_seg['start'] += start_offset
                adjusted_seg['end'] += start_offset
                merged_segment_timestamps.append(adjusted_seg)
    
    return {
        'text': ' '.join(merged_text),
        'word_timestamps': merged_word_timestamps,
        'segment_timestamps': merged_segment_timestamps,
        'chunks_processed': len([r for r in chunk_results if r]),
        'total_chunks': len(chunk_results)
    }

def main():
    input_file = "test2.mp3"
    target_chunk_duration = 300  # 5 minutes target, but will adjust for silence
    
    # Get API credentials
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        print("❌ RUNPOD_API_KEY environment variable not set")
        return
    
    endpoint_url = "https://api.runpod.ai/v2/7u304yobo6ytm9/run"
    endpoint_base = "https://api.runpod.ai/v2/7u304yobo6ytm9"
    
    print("🎯 SMART SILENCE-BASED CHUNKED TRANSCRIPTION")
    print("=" * 60)
    print("Splits at natural silence points for better transcription quality")
    
    # Step 1: Smart split audio at silence points
    print(f"\n📂 Step 1: Smart splitting audio file...")
    chunks = smart_split_audio(input_file, target_chunk_duration)
    
    if not chunks:
        print("❌ Failed to create chunks")
        return
    
    # Step 2: Transcribe each chunk
    print(f"\n🎤 Step 2: Transcribing {len(chunks)} smart chunks...")
    job_ids = []
    
    for i, chunk_info in enumerate(chunks):
        print(f"\n📝 Chunk {i+1}/{len(chunks)}: {chunk_info['file']}")
        print(f"   Duration: {chunk_info['duration']:.1f}s, Size: {chunk_info['size_mb']:.1f}MB")
        job_id = transcribe_chunk(chunk_info['file'], api_key, endpoint_url)
        job_ids.append(job_id)
        
        time.sleep(2)  # Small delay between requests
    
    # Step 3: Wait for all jobs to complete
    print(f"\n⏳ Step 3: Waiting for transcriptions to complete...")
    chunk_results = []
    
    for i, job_id in enumerate(job_ids):
        if job_id:
            print(f"\n⏰ Waiting for chunk {i+1} (Job: {job_id})...")
            result = wait_for_job_completion(job_id, api_key, endpoint_base)
            chunk_results.append(result)
        else:
            chunk_results.append(None)
    
    # Step 4: Merge results
    print(f"\n🔗 Step 4: Merging transcription results...")
    final_result = merge_transcriptions(chunk_results, chunks)
    
    # Step 5: Save results
    print(f"\n💾 Step 5: Saving results...")
    
    # Save to JSON
    with open("smart_transcription.json", 'w') as f:
        json.dump(final_result, f, indent=2)
    
    # Save to text file
    with open("smart_transcription.txt", 'w', encoding='utf-8') as f:
        f.write("SMART SILENCE-BASED TRANSCRIPTION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write("FULL TRANSCRIPTION:\n")
        f.write("-" * 20 + "\n")
        f.write(final_result['text'] + "\n\n")
        
        f.write("METADATA:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total chunks processed: {final_result['chunks_processed']}/{final_result['total_chunks']}\n")
        f.write(f"Total words with timestamps: {len(final_result['word_timestamps'])}\n")
        f.write(f"Total segments with timestamps: {len(final_result['segment_timestamps'])}\n\n")
        
        # Show chunk boundaries
        f.write("CHUNK BOUNDARIES:\n")
        f.write("-" * 20 + "\n")
        for i, chunk_info in enumerate(chunks):
            f.write(f"Chunk {i+1}: {chunk_info['start_time']:.1f}s - {chunk_info['end_time']:.1f}s ({chunk_info['duration']:.1f}s)\n")
        f.write("\n")
        
        if final_result['word_timestamps']:
            f.write("WORD TIMESTAMPS (first 20):\n")
            f.write("-" * 20 + "\n")
            for word_ts in final_result['word_timestamps'][:20]:
                f.write(f"{word_ts.get('start', 0):.2f}s - {word_ts.get('end', 0):.2f}s: {word_ts.get('word', 'N/A')}\n")
            if len(final_result['word_timestamps']) > 20:
                f.write(f"... and {len(final_result['word_timestamps']) - 20} more words\n")
    
    print(f"✅ Results saved to:")
    print(f"  - smart_transcription.json")
    print(f"  - smart_transcription.txt")
    print(f"\n🎉 Smart transcription complete!")
    print(f"📊 Processed {final_result['chunks_processed']}/{final_result['total_chunks']} chunks successfully")
    print(f"🎯 Split at natural silence points for optimal quality!")

if __name__ == "__main__":
    main()
