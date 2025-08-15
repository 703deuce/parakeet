#!/usr/bin/env python3
"""
Compare transcription results to understand word count differences
"""

import json
import os
import re

def analyze_transcription_file(json_file, txt_file):
    """Analyze a transcription file"""
    
    if not os.path.exists(json_file) or not os.path.exists(txt_file):
        print(f"❌ Files not found: {json_file} or {txt_file}")
        return None
    
    # Read JSON metadata
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Read text file
    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract the transcription text (between "FULL TRANSCRIPTION:" and "METADATA:")
    text_start = content.find("FULL TRANSCRIPTION:\n--------------------\n")
    text_end = content.find("\n\nMETADATA:")
    
    if text_start != -1 and text_end != -1:
        transcription_text = content[text_start + len("FULL TRANSCRIPTION:\n--------------------\n"):text_end]
    else:
        transcription_text = ""
    
    # Count words in actual transcription
    actual_words = len(transcription_text.split()) if transcription_text else 0
    
    # Get metadata
    json_word_count = data.get('total_words', 0) if isinstance(data, dict) else 0
    json_chunks = data.get('total_chunks', 0) if isinstance(data, dict) else 0
    json_word_timestamps = len(data.get('word_timestamps', [])) if isinstance(data, dict) else 0
    
    return {
        'file': json_file,
        'actual_word_count': actual_words,
        'json_word_timestamps': json_word_timestamps,
        'chunks': json_chunks,
        'transcription_preview': transcription_text[:200] + "..." if transcription_text else "No text found"
    }

def main():
    print("🔍 TRANSCRIPTION COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Files to compare
    files_to_compare = [
        ("chunked_transcription.json", "chunked_transcription.txt", "WAV 2-min chunks"),
        ("mp3_chunked_transcription.json", "mp3_chunked_transcription.txt", "MP3 5-min chunks")
    ]
    
    results = []
    
    for json_file, txt_file, description in files_to_compare:
        print(f"\n📊 Analyzing: {description}")
        result = analyze_transcription_file(json_file, txt_file)
        if result:
            result['description'] = description
            results.append(result)
            print(f"  Actual words in text: {result['actual_word_count']}")
            print(f"  Word timestamps: {result['json_word_timestamps']}")
            print(f"  Chunks: {result['chunks']}")
            print(f"  Preview: {result['transcription_preview'][:100]}...")
    
    # Compare results
    if len(results) >= 2:
        print(f"\n🔍 COMPARISON:")
        print("=" * 60)
        
        for i, result in enumerate(results):
            print(f"{i+1}. {result['description']}")
            print(f"   Words: {result['actual_word_count']}")
            print(f"   Timestamps: {result['json_word_timestamps']}")
            print(f"   Chunks: {result['chunks']}")
        
        # Calculate differences
        if len(results) == 2:
            word_diff = results[1]['actual_word_count'] - results[0]['actual_word_count']
            timestamp_diff = results[1]['json_word_timestamps'] - results[0]['json_word_timestamps']
            
            print(f"\n📈 DIFFERENCES:")
            print(f"   Word count difference: {word_diff:+d}")
            print(f"   Timestamp difference: {timestamp_diff:+d}")
            print(f"   Percentage difference: {(word_diff/results[0]['actual_word_count']*100):+.2f}%")
            
            # Analysis
            print(f"\n💡 ANALYSIS:")
            if abs(word_diff) <= 10:
                print("   ✅ Word count difference is minimal (<1%)")
                print("   ✅ Both transcriptions are essentially equivalent")
            else:
                print("   ⚠️  Significant word count difference detected")
            
            if results[1]['chunks'] < results[0]['chunks']:
                print(f"   📊 Fewer chunks ({results[1]['chunks']} vs {results[0]['chunks']}) = faster processing")
            
            print(f"   🎯 Recommendation: Use the version with fewer chunks for better efficiency")

if __name__ == "__main__":
    main()
