3#!/usr/bin/env python3
"""
Merge multiple chunked diarization results into one unified transcript
Stitches all chunks back together with consistent speaker labels and chronological ordering
"""

import json
import os
import re
from typing import List, Dict, Any, Tuple

def parse_transcript_to_segments(transcript_file: str) -> List[Dict[str, Any]]:
    """
    Parse a transcript file with word-level diarization and convert to segment-level
    Groups consecutive words from the same speaker into segments
    """
    print(f"📖 Parsing transcript file: {transcript_file}")
    
    if not os.path.exists(transcript_file):
        print(f"❌ Transcript file not found: {transcript_file}")
        return []
    
    with open(transcript_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the SPEAKER DIARIZATION section
    diarization_match = re.search(r'SPEAKER DIARIZATION:\s*-+\s*(.*?)(?=\n\n|\nPROCESSING METADATA|$)', content, re.DOTALL)
    if not diarization_match:
        print("❌ No SPEAKER DIARIZATION section found in transcript")
        return []
    
    diarization_text = diarization_match.group(1)
    
    # Parse each line: [timestamp] SPEAKER_XX: word
    lines = diarization_text.strip().split('\n')
    word_entries = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Match pattern: [start_time - end_time] SPEAKER_XX: word
        match = re.match(r'\[([0-9.]+)s\s*-\s*([0-9.]+)s\]\s+(SPEAKER_\d+):\s*(.+)', line)
        if match:
            start_time = float(match.group(1))
            end_time = float(match.group(2))
            speaker = match.group(3)
            word = match.group(4).strip()
            
            word_entries.append({
                'start_time': start_time,
                'end_time': end_time,
                'speaker': speaker,
                'word': word
            })
    
    print(f"  📝 Parsed {len(word_entries)} word entries")
    
    if not word_entries:
        print("❌ No word entries found")
        return []
    
    # Group words into segments based on speaker changes and punctuation
    segments = []
    current_segment = None
    
    # Punctuation that ends a segment
    segment_endings = ['.', '!', '?', ':', ';']
    
    for entry in word_entries:
        word = entry['word']
        speaker = entry['speaker']
        
        # Start new segment if speaker changed or this is the first word
        if current_segment is None or current_segment['speaker'] != speaker:
            # Save previous segment if exists
            if current_segment is not None:
                segments.append(current_segment)
            
            # Start new segment
            current_segment = {
                'speaker': speaker,
                'start_time': entry['start_time'],
                'end_time': entry['end_time'],
                'words': [word],
                'text': word
            }
        else:
            # Add word to current segment
            current_segment['words'].append(word)
            current_segment['text'] += ' ' + word
            current_segment['end_time'] = entry['end_time']
        
        # Check if word ends with punctuation that should end the segment
        if any(word.rstrip().endswith(punct) for punct in segment_endings):
            # End current segment
            if current_segment is not None:
                segments.append(current_segment)
                current_segment = None
    
    # Add any remaining segment
    if current_segment is not None:
        segments.append(current_segment)
    
    # Convert to the format expected by the merge function
    formatted_segments = []
    for segment in segments:
        formatted_segments.append({
            'speaker': segment['speaker'],
            'start_time': segment['start_time'],
            'end_time': segment['end_time'],
            'text': segment['text'].strip()
        })
    
    print(f"✅ Created {len(formatted_segments)} segments from word-level data")
    return formatted_segments

def stitch_diarized_chunks(chunk_outputs: List[Dict[str, Any]]) -> Tuple[str, Dict[str, str], List[Dict[str, Any]]]:
    """
    Merges multiple chunk diarization results into one global transcript.
    All times are assumed to be absolute already.
    Speakers are labeled as Speaker 1, 2, ... in order of appearance.
    
    Args:
        chunk_outputs: List of API outputs from each chunk
        
    Returns:
        final_text: Complete merged transcript as text
        speaker_map: Mapping from original labels to global labels
        all_segments: All segments with global speaker labels
    """
    print("🔗 Starting diarization chunk merging process...")
    
    # 1. Gather and flatten all segments from all chunks
    all_segments = []
    total_chunks = len(chunk_outputs)
    
    for chunk_idx, output in enumerate(chunk_outputs):
        if 'diarized_transcript' in output:
            chunk_segments = output.get('diarized_transcript', [])
            print(f"  📝 Chunk {chunk_idx + 1}: Found {len(chunk_segments)} segments")
            
            for seg in chunk_segments:
                # Ensure we have all required fields
                if all(key in seg for key in ['speaker', 'start_time', 'end_time', 'text']):
                    all_segments.append(seg)
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
    
    # 4. Assemble final transcript with multiple output formats
    stitched_lines = []
    stitched_compact = []
    
    for seg in all_segments:
        label = seg['global_speaker']
        st = seg['start_time']
        et = seg['end_time']
        txt = seg['text'].strip()
        
        if txt:
            # Detailed format with timestamps
            stitched_lines.append(f"[{st:.1f}s-{et:.1f}s] {label}: {txt}")
            
            # Compact format for easier reading
            stitched_compact.append(f"{label}: {txt}")
    
    # Create final text blocks
    final_text_detailed = "\n".join(stitched_lines)
    final_text_compact = "\n\n".join(stitched_compact)
    
    print(f"✅ Created merged transcript with {len(stitched_lines)} segments")
    
    return final_text_detailed, speaker_map, all_segments

def save_merged_transcript(final_text: str, speaker_map: Dict[str, str], 
                          all_segments: List[Dict[str, Any]], 
                          output_prefix: str = "merged_transcript") -> None:
    """
    Save the merged transcript in multiple formats
    """
    print(f"\n💾 Saving merged transcript to files...")
    
    # 1. Detailed transcript with timestamps
    detailed_file = f"{output_prefix}_detailed.txt"
    with open(detailed_file, 'w', encoding='utf-8') as f:
        f.write("🎤 MERGED DIARIZATION TRANSCRIPT (All Chunks Combined)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total segments: {len(all_segments)}\n")
        f.write(f"Total speakers: {len(speaker_map)}\n")
        f.write(f"Speaker mapping: {speaker_map}\n\n")
        f.write("TRANSCRIPT:\n")
        f.write("-" * 20 + "\n\n")
        f.write(final_text)
    
    print(f"  📄 Detailed transcript: {detailed_file}")
    
    # 2. Compact transcript (speaker only)
    compact_file = f"{output_prefix}_compact.txt"
    with open(compact_file, 'w', encoding='utf-8') as f:
        f.write("🎤 MERGED TRANSCRIPT - COMPACT FORMAT\n")
        f.write("=" * 50 + "\n\n")
        for seg in all_segments:
            label = seg['global_speaker']
            txt = seg['text'].strip()
            if txt:
                f.write(f"{label}: {txt}\n\n")
    
    print(f"  📄 Compact transcript: {compact_file}")
    
    # 3. JSON summary with metadata
    json_file = f"{output_prefix}_summary.json"
    summary_data = {
        'metadata': {
            'total_segments': len(all_segments),
            'total_speakers': len(speaker_map),
            'speaker_mapping': speaker_map,
            'total_duration': max(seg['end_time'] for seg in all_segments) if all_segments else 0
        },
        'segments': all_segments,
        'transcript_detailed': final_text
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"  📄 JSON summary: {json_file}")

def merge_from_json_file(json_file: str, output_prefix: str = None) -> None:
    """
    Load results from a JSON file and merge them
    """
    if not os.path.exists(json_file):
        print(f"❌ JSON file not found: {json_file}")
        return
    
    print(f"📖 Loading diarization results from: {json_file}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract chunk outputs
        if 'results' in data:
            chunk_outputs = []
            for result in data['results']:
                if result and 'output' in result:
                    chunk_outputs.append(result['output'])
            
            if not chunk_outputs:
                print("❌ No valid outputs found in results")
                return
            
            print(f"✅ Found {len(chunk_outputs)} chunk outputs to merge")
            
            # Set output prefix if not provided
            if output_prefix is None:
                base_name = os.path.splitext(os.path.basename(json_file))[0]
                output_prefix = f"merged_{base_name}"
            
            # Merge the chunks
            final_text, speaker_map, all_segments = stitch_diarized_chunks(chunk_outputs)
            
            if final_text:
                # Save the merged transcript
                save_merged_transcript(final_text, speaker_map, all_segments, output_prefix)
                
                # Show preview
                print(f"\n🎯 MERGED TRANSCRIPT PREVIEW (first 1000 chars):")
                print("=" * 60)
                print(final_text[:1000])
                if len(final_text) > 1000:
                    print(f"\n... and {len(final_text) - 1000} more characters")
                
                print(f"\n🎉 Successfully merged {len(chunk_outputs)} chunks into unified transcript!")
                print(f"📊 Total segments: {len(all_segments)}")
                print(f"👥 Speakers: {list(speaker_map.values())}")
            else:
                print("❌ Failed to merge transcript")
                
        else:
            print("❌ No 'results' field found in JSON file")
            
    except Exception as e:
        print(f"❌ Error processing JSON file: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function - process available diarization result files
    """
    print("🎤 DIARIZATION CHUNK MERGER")
    print("=" * 50)
    print("Merges multiple chunked diarization results into one unified transcript")
    
    # Look for available result files
    available_files = []
    for file in os.listdir('.'):
        if file.endswith('_diarization_results.json') or file.endswith('_speakers_diarization_results.json'):
            available_files.append(file)
    
    if not available_files:
        print("❌ No diarization result files found in current directory")
        print("   Looking for files ending with '_diarization_results.json'")
        return
    
    print(f"📁 Found {len(available_files)} result files:")
    for i, file in enumerate(available_files):
        print(f"  {i+1}. {file}")
    
    # Process the first available file (you can modify this logic)
    if available_files:
        selected_file = available_files[0]
        print(f"\n🔄 Processing: {selected_file}")
        
        # Generate output prefix
        base_name = os.path.splitext(selected_file)[0]
        output_prefix = f"merged_{base_name}"
        
        merge_from_json_file(selected_file, output_prefix)
    else:
        print("❌ No valid files to process")

if __name__ == "__main__":
    main()
