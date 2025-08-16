#!/usr/bin/env python3
"""
UNIFIED DIARIZATION PIPELINE
One script that does everything: test diarization, extract results, and merge chunks
"""

import os
import requests
import base64
import json
import time
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from typing import List, Dict, Any, Tuple

def smart_split_audio(input_file, target_chunk_duration=300, output_dir="unified_chunks"):
    """Split audio at silence points near target duration"""
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎵 Loading audio file: {input_file}")
    
    # Handle different audio formats
    if input_file.lower().endswith('.mp4'):
        print("🎬 Converting MP4 to audio for processing...")
        audio = AudioSegment.from_file(input_file, format="mp4")
    else:
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
        chunk_filename = f"unified_chunk_{i:03d}.mp3"
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

def find_silence_split_points(audio, target_chunk_duration_ms, silence_thresh=-40, min_silence_len=500):
    """Find optimal split points near target duration that fall on silence"""
    
    print(f"🔍 Analyzing audio for silence points...")
    print(f"  Target chunk duration: {target_chunk_duration_ms/1000:.1f}s")
    print(f"  Silence threshold: {silence_thresh}dBFS")
    print(f"  Minimum silence length: {min_silence_len}ms")
    
    # Detect non-silent segments
    nonsilent_ranges = detect_nonsilent(
        audio, 
        min_silence_len=min_silence_len, 
        silence_thresh=silence_thresh
    )
    
    if not nonsilent_ranges:
        print("⚠️  No silence detected, falling back to time-based splitting")
        return list(range(0, len(audio), target_chunk_duration_ms))
    
    print(f"✅ Found {len(nonsilent_ranges)} speech segments")
    
    # Calculate silence points
    silence_points = []
    
    # Add start if there's initial silence
    if nonsilent_ranges[0][0] > 0:
        silence_points.append(nonsilent_ranges[0][0] // 2)
    
    # Add points between speech segments
    for i in range(len(nonsilent_ranges) - 1):
        end_of_current = nonsilent_ranges[i][1]
        start_of_next = nonsilent_ranges[i + 1][0]
        
        if start_of_next - end_of_current >= min_silence_len:
            silence_point = end_of_current + (start_of_next - end_of_current) // 2
            silence_points.append(silence_point)
    
    # Add end if there's final silence
    if nonsilent_ranges[-1][1] < len(audio):
        silence_points.append(nonsilent_ranges[-1][1] + (len(audio) - nonsilent_ranges[-1][1]) // 2)
    
    print(f"✅ Found {len(silence_points)} potential silence split points")
    
    # Find best split points near target durations
    split_points = [0]
    current_pos = 0
    
    while current_pos < len(audio):
        target_next_split = current_pos + target_chunk_duration_ms
        
        if target_next_split >= len(audio):
            break
        
        # Find silence point closest to target
        best_split = target_next_split
        min_distance = float('inf')
        
        for silence_point in silence_points:
            if silence_point > current_pos:
                distance = abs(silence_point - target_next_split)
                max_deviation = target_chunk_duration_ms * 0.3
                
                if distance < min_distance and distance <= max_deviation:
                    min_distance = distance
                    best_split = silence_point
        
        split_points.append(best_split)
        current_pos = best_split
        
        deviation_seconds = (best_split - target_next_split) / 1000
        print(f"  Split point: {best_split/1000:.1f}s (deviation: {deviation_seconds:+.1f}s)")
    
    # Ensure we end at actual end of audio
    if split_points[-1] < len(audio):
        split_points.append(len(audio))
    
    return split_points

def test_chunk_diarization(chunk_file, api_key, endpoint_url, num_speakers=None):
    """Test diarization on a single chunk"""
    
    try:
        # Encode audio chunk
        with open(chunk_file, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Check size
        chunk_size_mb = len(audio_data) / (1024 * 1024)
        print(f"  📊 Chunk size: {chunk_size_mb:.2f}MB (base64)")
        
        if chunk_size_mb > 9:
            print(f"  ⚠️  Chunk still too large ({chunk_size_mb:.2f}MB), skipping")
            return None
        
        payload = {
            "input": {
                "audio_data": audio_data,
                "audio_format": "mp3",
                "include_timestamps": True,
                "use_diarization": True,
                "num_speakers": num_speakers,
                "chunk_duration": 300
            }
        }
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        speaker_info = f" ({num_speakers} speakers)" if num_speakers else " (auto-detect)"
        print(f"  📤 Testing diarization on chunk{speaker_info}...")
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

def stitch_diarized_chunks(chunk_outputs: List[Dict[str, Any]]) -> Tuple[str, Dict[str, str], List[Dict[str, Any]]]:
    """Merge multiple chunk diarization results into one global transcript"""
    
    print("🔗 Starting diarization chunk merging process...")
    
    # 1. Gather and flatten all segments from all chunks
    all_segments = []
    total_chunks = len(chunk_outputs)
    
    for chunk_idx, output in enumerate(chunk_outputs):
        if 'diarized_transcript' in output:
            chunk_segments = output.get('diarized_transcript', [])
            print(f"  📝 Chunk {chunk_idx + 1}: Found {len(chunk_segments)} segments")
            
            # Get chunk boundaries to adjust timestamps
            chunk_boundaries = output.get('chunk_boundaries', [])
            chunk_start_time = 0
            if chunk_boundaries and len(chunk_boundaries) > chunk_idx:
                chunk_start_time = chunk_boundaries[chunk_idx].get('start', 0)
            
            for seg in chunk_segments:
                if all(key in seg for key in ['speaker', 'start_time', 'end_time', 'text']):
                    # Adjust timestamps to be absolute from original audio
                    adjusted_seg = seg.copy()
                    adjusted_seg['start_time'] = seg['start_time'] + chunk_start_time
                    adjusted_seg['end_time'] = seg['end_time'] + chunk_start_time
                    adjusted_seg['source_chunk'] = chunk_idx + 1
                    all_segments.append(adjusted_seg)
                else:
                    print(f"    ⚠️  Skipping incomplete segment: {seg}")
        else:
            print(f"  ⚠️  Chunk {chunk_idx + 1}: No diarized_transcript found")
    
    print(f"✅ Total segments collected: {len(all_segments)}")
    
    if not all_segments:
        print("❌ No valid segments found to merge")
        return "", {}, []
    
    # 2. Sort by absolute start_time to maintain chronological order
    all_segments.sort(key=lambda seg: seg['start_time'])
    print(f"📅 Sorted {len(all_segments)} segments chronologically")
    
    # 3. Assign global speaker labels by order of appearance
    speaker_map = {}
    speaker_counter = 1
    
    print("🎭 Assigning global speaker labels...")
    for seg in all_segments:
        spk_label = seg['speaker']
        if spk_label not in speaker_map and spk_label != "UNKNOWN":
            speaker_map[spk_label] = f"Speaker {speaker_counter}"
            speaker_counter += 1
            print(f"  🗣️  {spk_label} → {speaker_map[spk_label]}")
        
        seg['global_speaker'] = speaker_map.get(spk_label, "Unknown")
    
    print(f"✅ Mapped {len(speaker_map)} unique speakers")
    
    # 4. Assemble final transcript
    stitched_lines = []
    for seg in all_segments:
        label = seg['global_speaker']
        st = seg['start_time']
        et = seg['end_time']
        txt = seg['text'].strip()
        
        if txt:
            stitched_lines.append(f"[{st:.1f}s-{et:.1f}s] {label}: {txt}")
    
    final_text = "\n".join(stitched_lines)
    print(f"✅ Created merged transcript with {len(stitched_lines)} segments")
    
    return final_text, speaker_map, all_segments

def save_final_transcript(final_text: str, speaker_map: Dict[str, str], 
                         all_segments: List[Dict[str, Any]], 
                         input_file: str) -> None:
    """Save the final merged transcript"""
    
    print(f"\n💾 Saving final unified transcript...")
    
    # Generate output filename based on input file
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"FINAL_UNIFIED_{base_name}_transcript.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("🎤 FINAL UNIFIED DIARIZATION TRANSCRIPT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Source file: {input_file}\n")
        f.write(f"Total segments: {len(all_segments)}\n")
        f.write(f"Total speakers: {len(speaker_map)}\n")
        f.write(f"Speaker mapping: {speaker_map}\n\n")
        f.write("COMPLETE TRANSCRIPT:\n")
        f.write("-" * 20 + "\n\n")
        f.write(final_text)
    
    print(f"✅ Final transcript saved to: {output_file}")
    
    # Also save a compact version
    compact_file = f"FINAL_COMPACT_{base_name}_transcript.txt"
    with open(compact_file, 'w', encoding='utf-8') as f:
        f.write("🎤 COMPACT TRANSCRIPT (Speaker Only)\n")
        f.write("=" * 50 + "\n\n")
        for seg in all_segments:
            label = seg['global_speaker']
            txt = seg['text'].strip()
            if txt:
                f.write(f"{label}: {txt}\n\n")
    
    print(f"✅ Compact transcript saved to: {compact_file}")

def main():
    """Main unified pipeline function"""
    
    print("🎤 UNIFIED DIARIZATION PIPELINE")
    print("=" * 60)
    print("One script that does everything: test, extract, and merge!")
    
    # Configuration
    input_file = input("Enter audio file path (e.g., test4.mp3): ").strip()
    if not input_file:
        input_file = "test4.mp3"  # Default
    
    num_speakers = input("Enter number of speakers (or press Enter for auto-detect): ").strip()
    if num_speakers:
        try:
            num_speakers = int(num_speakers)
        except ValueError:
            num_speakers = None
            print("⚠️  Invalid number, using auto-detect")
    
    # Get API credentials
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        print("❌ RUNPOD_API_KEY environment variable not set")
        return
    
    endpoint_url = "https://api.runpod.ai/v2/7u304yobo6ytm9/run"
    endpoint_base = "https://api.runpod.ai/v2/7u304yobo6ytm9"
    
    print(f"\n🎯 CONFIGURATION:")
    print(f"  Input file: {input_file}")
    print(f"  Speakers: {num_speakers if num_speakers else 'Auto-detect'}")
    print(f"  Endpoint: {endpoint_base}")
    
    # Step 1: Smart split audio
    print(f"\n📂 STEP 1: Smart splitting audio file...")
    chunks = smart_split_audio(input_file, 300 if input_file.lower().endswith('.mp3') else 180)
    
    if not chunks:
        print("❌ Failed to create chunks")
        return
    
    # Step 2: Test diarization on each chunk
    print(f"\n🎤 STEP 2: Testing diarization on {len(chunks)} chunks...")
    job_ids = []
    
    for i, chunk_info in enumerate(chunks):
        print(f"\n📝 Chunk {i+1}/{len(chunks)}: {chunk_info['file']}")
        print(f"   Duration: {chunk_info['duration']:.1f}s, Size: {chunk_info['size_mb']:.1f}MB")
        
        job_id = test_chunk_diarization(chunk_info['file'], api_key, endpoint_url, num_speakers)
        job_ids.append(job_id)
        
        if job_id:
            time.sleep(2)  # Small delay between requests
    
    # Step 3: Wait for all jobs to complete
    print(f"\n⏳ STEP 3: Waiting for diarization jobs to complete...")
    chunk_results = []
    
    for i, job_id in enumerate(job_ids):
        if job_id:
            print(f"\n⏰ Waiting for chunk {i+1} (Job: {job_id})...")
            result = wait_for_job_completion(job_id, api_key, endpoint_base)
            chunk_results.append(result)
        else:
            chunk_results.append(None)
    
    # Step 4: Extract and merge results
    print(f"\n🔍 STEP 4: Extracting and merging results...")
    
    successful_results = [r for r in chunk_results if r and 'output' in r]
    print(f"✅ Successful diarization jobs: {len(successful_results)}/{len(chunk_results)}")
    
    if successful_results:
        # Save raw results
        results_file = f"unified_diarization_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'chunks': chunks,
                'results': chunk_results,
                'summary': {
                    'total_chunks': len(chunks),
                    'successful_jobs': len(successful_results),
                    'failed_jobs': len(chunk_results) - len(successful_results)
                }
            }, f, indent=2)
        
        print(f"💾 Raw results saved to: {results_file}")
        
        # Extract chunk outputs for merging
        chunk_outputs = []
        for result in successful_results:
            if 'output' in result:
                chunk_outputs.append(result['output'])
        
        # DEBUG: Show what we're getting from the API
        print(f"\n🔍 DEBUG: What the API is returning:")
        for i, output in enumerate(chunk_outputs):
            print(f"  Chunk {i+1} output keys: {list(output.keys())}")
            if 'diarized_transcript' in output:
                print(f"  Chunk {i+1} diarized_transcript: {len(output['diarized_transcript'])} segments")
                for j, seg in enumerate(output['diarized_transcript']):
                    print(f"    Segment {j+1}: speaker='{seg.get('speaker', 'MISSING')}', text='{seg.get('text', '')[:50]}...'")
            else:
                print(f"  Chunk {i+1}: No diarized_transcript found!")
        
        # Merge all chunks into one transcript
        print(f"\n🔗 STEP 5: Merging all chunks into unified transcript...")
        final_text, speaker_map, all_segments = stitch_diarized_chunks(chunk_outputs)
        
        if final_text:
            # Save final unified transcript
            save_final_transcript(final_text, speaker_map, all_segments, input_file)
            
            # Show preview
            print(f"\n🎯 FINAL UNIFIED TRANSCRIPT PREVIEW (first 1000 chars):")
            print("=" * 60)
            print(final_text[:1000])
            if len(final_text) > 1000:
                print(f"\n... and {len(final_text) - 1000} more characters")
            
            print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"📊 Total chunks processed: {len(chunks)}")
            print(f"📝 Total segments: {len(all_segments)}")
            print(f"👥 Speakers: {list(speaker_map.values())}")
            print(f"📄 Final transcript saved with consistent speaker labels!")
        else:
            print("❌ Failed to merge transcript")
    else:
        print("❌ No successful diarization jobs")

if __name__ == "__main__":
    main()
