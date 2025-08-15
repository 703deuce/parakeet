#!/usr/bin/env python3
"""
Find timestamps for specific words in transcription results
"""

import json
import os

def find_word_timestamps(json_file, target_words):
    """Find timestamps for specific words"""
    
    if not os.path.exists(json_file):
        print(f"❌ File not found: {json_file}")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    word_timestamps = data.get('word_timestamps', [])
    
    if not word_timestamps:
        print(f"❌ No word timestamps found in {json_file}")
        return
    
    print(f"🔍 Searching for words in {json_file}:")
    print(f"Total words with timestamps: {len(word_timestamps)}")
    
    found_words = []
    
    for target_word in target_words:
        print(f"\n📍 Looking for: '{target_word}'")
        matches = []
        
        for i, word_ts in enumerate(word_timestamps):
            word = word_ts.get('word', '').lower()
            target_lower = target_word.lower()
            
            # Check for exact match or partial match (for names)
            if word == target_lower or target_lower in word or word in target_lower:
                matches.append({
                    'position': i,
                    'word': word_ts.get('word', ''),
                    'start': word_ts.get('start', 0),
                    'end': word_ts.get('end', 0),
                    'context_before': [w.get('word', '') for w in word_timestamps[max(0, i-3):i]],
                    'context_after': [w.get('word', '') for w in word_timestamps[i+1:i+4]]
                })
        
        if matches:
            print(f"✅ Found {len(matches)} match(es):")
            for match in matches:
                context_before = ' '.join(match['context_before'])
                context_after = ' '.join(match['context_after'])
                print(f"  Position {match['position']}: '{match['word']}'")
                print(f"  Time: {match['start']:.2f}s - {match['end']:.2f}s")
                print(f"  Context: ...{context_before} [{match['word']}] {context_after}...")
                found_words.append(match)
        else:
            print(f"❌ No matches found for '{target_word}'")
    
    return found_words

def main():
    print("🎧 WORD TIMESTAMP FINDER")
    print("=" * 60)
    
    # The 3 key different words we found
    target_words = ["Supsang", "Sufsang", "Reha", "Rebecca", "Khanna", "Conna"]
    
    # Check both transcription files
    files_to_check = [
        ("chunked_transcription.json", "WAV (4 chunks)"),
        ("mp3_chunked_transcription.json", "MP3 (2 chunks)")
    ]
    
    all_results = {}
    
    for json_file, description in files_to_check:
        print(f"\n{'='*60}")
        print(f"📊 ANALYZING: {description}")
        print(f"File: {json_file}")
        print('='*60)
        
        results = find_word_timestamps(json_file, target_words)
        all_results[description] = results
    
    # Summary of key timestamps
    print(f"\n{'='*60}")
    print("🎯 KEY TIMESTAMPS FOR LISTENING")
    print('='*60)
    
    key_words = [
        ("Supsang/Sufsang", "Reporter's last name"),
        ("Reha/Rebecca", "Host's name"), 
        ("Khanna/Conna", "Congressman's name")
    ]
    
    for word_pair, description in key_words:
        print(f"\n🔍 {description} ({word_pair}):")
        
        for file_desc, results in all_results.items():
            if results:
                # Find matches for this word pair
                word1, word2 = word_pair.split('/')
                matches = [r for r in results if word1.lower() in r['word'].lower() or word2.lower() in r['word'].lower()]
                
                if matches:
                    for match in matches:
                        print(f"  {file_desc}: '{match['word']}' at {match['start']:.2f}s - {match['end']:.2f}s")
                else:
                    print(f"  {file_desc}: Not found")

if __name__ == "__main__":
    main()
