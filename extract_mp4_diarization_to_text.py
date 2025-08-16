#!/usr/bin/env python3
"""
Extract MP4 diarization results from JSON and save to readable text file
"""

import json
import os

def extract_mp4_diarization_to_text(json_file, output_file):
    """Extract MP4 diarization results to readable text format"""
    
    if not os.path.exists(json_file):
        print(f"❌ JSON file not found: {json_file}")
        return
    
    print(f"📖 Reading MP4 diarization results from: {json_file}")
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")
        return
    
    print(f"📝 Extracting results to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("🎤 MP4 DIARIZATION RESULTS (test4.mp3)\n")
        f.write("=" * 60 + "\n\n")
        
        # Summary
        summary = data.get('summary', {})
        f.write("📊 SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total chunks processed: {summary.get('total_chunks', 0)}\n")
        f.write(f"Successful jobs: {summary.get('successful_jobs', 0)}\n")
        f.write(f"Failed jobs: {summary.get('failed_jobs', 0)}\n\n")
        
        # Chunk information
        chunks = data.get('chunks', [])
        f.write("📂 CHUNK INFORMATION\n")
        f.write("-" * 20 + "\n")
        for i, chunk in enumerate(chunks):
            f.write(f"Chunk {i+1}:\n")
            f.write(f"  Time: {chunk.get('start_time', 0):.1f}s - {chunk.get('end_time', 0):.1f}s\n")
            f.write(f"  Duration: {chunk.get('duration', 0):.1f}s\n")
            f.write(f"  Size: {chunk.get('size_mb', 0):.2f}MB\n")
            f.write(f"  File: {chunk.get('file', 'N/A')}\n\n")
        
        # Diarization results
        results = data.get('results', [])
        f.write("🎭 DIARIZATION RESULTS BY CHUNK\n")
        f.write("=" * 60 + "\n\n")
        
        for i, result in enumerate(results):
            if not result or 'output' not in result:
                f.write(f"❌ Chunk {i+1}: No result\n\n")
                continue
            
            output = result.get('output', {})
            f.write(f"📝 CHUNK {i+1} RESULTS\n")
            f.write("-" * 30 + "\n")
            
            # Check if diarization was successful
            if 'diarized_transcript' in output:
                f.write(f"🎭 SPEAKER DIARIZATION ENABLED\n")
                f.write(f"Speakers detected: {output.get('speakers_detected', 0)}\n")
                f.write(f"Segments processed: {output.get('segments_processed', 0)}\n")
                f.write(f"Processing method: {output.get('processing_method', 'N/A')}\n\n")
                
                # Show each speaker segment
                diarized_transcript = output.get('diarized_transcript', [])
                for j, segment in enumerate(diarized_transcript):
                    speaker = segment.get('speaker', 'Unknown')
                    start_time = segment.get('start_time', 0)
                    end_time = segment.get('end_time', 0)
                    duration = segment.get('duration', 0)
                    text = segment.get('text', '')
                    source_chunk = segment.get('source_chunk', i+1)
                    
                    f.write(f"🗣️  Segment {j+1} - Speaker: {speaker}\n")
                    f.write(f"   Time: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s)\n")
                    f.write(f"   Source Chunk: {source_chunk}\n")
                    f.write(f"   Text: {text}\n\n")
                
                # Show merged text
                if 'merged_text' in output:
                    f.write(f"📝 COMPLETE TRANSCRIPT (Chunk {i+1}):\n")
                    f.write(f"{output['merged_text']}\n\n")
                    
            else:
                # Regular transcription
                f.write(f"📝 REGULAR TRANSCRIPTION\n")
                f.write(f"Processing method: {output.get('processing_method', 'N/A')}\n")
                f.write(f"Chunks processed: {output.get('chunks_processed', 0)}\n")
                f.write(f"Model used: {output.get('model_used', 'N/A')}\n\n")
                
                # Show transcription text
                text = output.get('text', '')
                if text:
                    f.write(f"📝 TRANSCRIPTION TEXT:\n")
                    f.write(f"{text}\n\n")
                
                # Show word timestamps if available
                word_timestamps = output.get('word_timestamps', [])
                if word_timestamps:
                    f.write(f"🔤 WORD TIMESTAMPS (first 20):\n")
                    for k, word_ts in enumerate(word_timestamps[:20]):
                        word = word_ts.get('word', 'N/A')
                        start = word_ts.get('start', 0)
                        end = word_ts.get('end', 0)
                        f.write(f"   {start:.2f}s - {end:.2f}s: {word}\n")
                    
                    if len(word_timestamps) > 20:
                        f.write(f"   ... and {len(word_timestamps) - 20} more words\n")
                    f.write("\n")
            
            f.write("=" * 60 + "\n\n")
        
        # Overall summary
        f.write("🎯 OVERALL ASSESSMENT\n")
        f.write("=" * 30 + "\n")
        
        successful_results = [r for r in results if r and 'output' in r]
        diarization_results = [r for r in successful_results if 'diarized_transcript' in r.get('output', {})]
        regular_results = [r for r in successful_results if 'diarized_transcript' not in r.get('output', {})]
        
        f.write(f"✅ Total successful jobs: {len(successful_results)}\n")
        f.write(f"🎭 Diarization jobs: {len(diarization_results)}\n")
        f.write(f"📝 Regular transcription jobs: {len(regular_results)}\n")
        
        if diarization_results:
            total_speakers = 0
            total_segments = 0
            for result in diarization_results:
                output = result.get('output', {})
                total_speakers += output.get('speakers_detected', 0)
                total_segments += output.get('segments_processed', 0)
            
            f.write(f"👥 Total speakers detected across all chunks: {total_speakers}\n")
            f.write(f"📊 Total segments processed: {total_segments}\n")
        
        f.write(f"\n🎉 MP4 diarization test completed successfully!\n")
    
    print(f"✅ Results extracted to: {output_file}")

def main():
    json_file = "mp4_chunked_diarization_results.json"
    output_file = "mp4_diarization_results.txt"
    
    extract_mp4_diarization_to_text(json_file, output_file)
    
    # Also show a quick summary
    if os.path.exists(output_file):
        print(f"\n📋 QUICK SUMMARY:")
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[:20]:  # Show first 20 lines
                print(line.rstrip())

if __name__ == "__main__":
    main()
