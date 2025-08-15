#!/usr/bin/env python3
"""
Split large audio files into chunks and transcribe each chunk
"""

import os
import requests
import base64
import json
import time
from pydub import AudioSegment

def split_audio_file(input_file, chunk_duration_seconds=120, output_dir="chunks"):
    """Split audio file into smaller chunks"""
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading audio file: {input_file}")
    audio = AudioSegment.from_file(input_file)
    
    total_duration = len(audio) / 1000  # Convert to seconds
    chunk_duration_ms = chunk_duration_seconds * 1000
    
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Chunk duration: {chunk_duration_seconds} seconds")
    
    chunks = []
    chunk_count = 0
    
    for start_ms in range(0, len(audio), chunk_duration_ms):
        end_ms = min(start_ms + chunk_duration_ms, len(audio))
        chunk = audio[start_ms:end_ms]
        
        # Save chunk as MP3 to test MP3 support
        chunk_filename = f"chunk_{chunk_count:03d}.mp3"
        chunk_path = os.path.join(output_dir, chunk_filename)
        
        # Export as MP3 with optimal settings for speech recognition
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
        chunk_count += 1
        
        print(f"Created chunk {chunk_count}: {chunk_info['duration']:.1f}s, {chunk_info['size_mb']:.1f}MB")
    
    print(f"✅ Created {len(chunks)} chunks in {output_dir}/")
    return chunks

def transcribe_chunk(chunk_file, api_key, endpoint_url):
    """Transcribe a single audio chunk"""
    
    try:
        # Encode audio file
        with open(chunk_file, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Detect audio format from file extension
        audio_format = "wav"  # Default to wav since we export chunks as wav
        if chunk_file.lower().endswith('.mp3'):
            audio_format = "mp3"
        elif chunk_file.lower().endswith('.flac'):
            audio_format = "flac"
        
        payload = {
            "input": {
                "audio_data": audio_data,
                "audio_format": audio_format,
                "include_timestamps": True,  # Get timestamps for merging
                "chunk_duration": 1200  # This won't be used since we're pre-chunking
            }
        }
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        print(f"  Uploading chunk ({len(audio_data)} chars)...")
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
    
    max_attempts = 30  # 5 minutes max wait
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
                    time.sleep(10)  # Wait 10 seconds
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
    """Merge transcription results from multiple chunks"""
    
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
            
            # Adjust timestamps by adding chunk start time
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
    input_file = "test2.mp3"  # Using MP3 directly
    
    # Optimal chunk sizes based on format
    if input_file.lower().endswith('.mp3'):
        chunk_duration = 300  # 5 minutes for MP3 (stays under 10MiB limit)
        print("🎵 Using MP3 format - 5-minute chunks for optimal efficiency")
    else:
        chunk_duration = 180  # 3 minutes for WAV (stays under 10MiB limit)
        print("🎵 Using WAV format - 3-minute chunks for optimal efficiency")
    
    # Get API credentials
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        print("❌ RUNPOD_API_KEY environment variable not set")
        return
    
    endpoint_url = "https://api.runpod.ai/v2/7u304yobo6ytm9/run"
    endpoint_base = "https://api.runpod.ai/v2/7u304yobo6ytm9"
    
    print("🎵 CHUNKED TRANSCRIPTION PROCESS")
    print("=" * 50)
    
    # Step 1: Split audio into chunks
    print("\n📂 Step 1: Splitting audio file...")
    chunks = split_audio_file(input_file, chunk_duration)
    
    if not chunks:
        print("❌ Failed to create chunks")
        return
    
    # Step 2: Transcribe each chunk
    print(f"\n🎤 Step 2: Transcribing {len(chunks)} chunks...")
    job_ids = []
    
    for i, chunk_info in enumerate(chunks):
        print(f"\nChunk {i+1}/{len(chunks)}: {chunk_info['file']}")
        job_id = transcribe_chunk(chunk_info['file'], api_key, endpoint_url)
        job_ids.append(job_id)
        
        # Small delay between requests
        time.sleep(2)
    
    # Step 3: Wait for all jobs to complete
    print(f"\n⏳ Step 3: Waiting for transcriptions to complete...")
    chunk_results = []
    
    for i, job_id in enumerate(job_ids):
        if job_id:
            print(f"\nWaiting for chunk {i+1} (Job: {job_id})...")
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
    with open("mp3_chunked_transcription.json", 'w') as f:
        json.dump(final_result, f, indent=2)
    
    # Save to text file
    with open("mp3_chunked_transcription.txt", 'w', encoding='utf-8') as f:
        f.write("CHUNKED TRANSCRIPTION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write("FULL TRANSCRIPTION:\n")
        f.write("-" * 20 + "\n")
        f.write(final_result['text'] + "\n\n")
        
        f.write("METADATA:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total chunks processed: {final_result['chunks_processed']}/{final_result['total_chunks']}\n")
        f.write(f"Total words with timestamps: {len(final_result['word_timestamps'])}\n")
        f.write(f"Total segments with timestamps: {len(final_result['segment_timestamps'])}\n\n")
        
        if final_result['word_timestamps']:
            f.write("WORD TIMESTAMPS (first 20):\n")
            f.write("-" * 20 + "\n")
            for word_ts in final_result['word_timestamps'][:20]:
                f.write(f"{word_ts.get('start', 0):.2f}s - {word_ts.get('end', 0):.2f}s: {word_ts.get('word', 'N/A')}\n")
            if len(final_result['word_timestamps']) > 20:
                f.write(f"... and {len(final_result['word_timestamps']) - 20} more words\n")
    
    print(f"✅ Results saved to:")
    print(f"  - mp3_chunked_transcription.json")
    print(f"  - mp3_chunked_transcription.txt")
    print(f"\n🎉 Transcription complete!")
    print(f"📊 Processed {final_result['chunks_processed']}/{final_result['total_chunks']} chunks successfully")

if __name__ == "__main__":
    main()
