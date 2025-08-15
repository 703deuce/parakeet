#!/usr/bin/env python3
"""
Clean word-by-word comparison to find exact differences
"""

import os

def extract_transcription_text(txt_file):
    """Extract just the transcription text from the file"""
    
    if not os.path.exists(txt_file):
        return None
    
    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract the transcription text (between "FULL TRANSCRIPTION:" and "METADATA:")
    text_start = content.find("FULL TRANSCRIPTION:\n--------------------\n")
    text_end = content.find("\n\nMETADATA:")
    
    if text_start != -1 and text_end != -1:
        transcription_text = content[text_start + len("FULL TRANSCRIPTION:\n--------------------\n"):text_end]
        return transcription_text.strip()
    
    return None

def find_exact_differences(text1, text2, label1, label2):
    """Find exact word differences with clean output"""
    
    words1 = text1.split()
    words2 = text2.split()
    
    print(f"📊 Comparing:")
    print(f"  {label1}: {len(words1)} words")
    print(f"  {label2}: {len(words2)} words")
    
    # Find differences by comparing word by word
    differences = []
    max_len = max(len(words1), len(words2))
    
    i = 0
    while i < max_len:
        if i >= len(words1):
            # Extra words in text2
            differences.append({
                'type': 'added',
                'position': i,
                'word': words2[i],
                'context': ' '.join(words2[max(0, i-3):i+4])
            })
        elif i >= len(words2):
            # Extra words in text1
            differences.append({
                'type': 'removed',
                'position': i,
                'word': words1[i],
                'context': ' '.join(words1[max(0, i-3):i+4])
            })
        elif words1[i] != words2[i]:
            # Different words
            differences.append({
                'type': 'changed',
                'position': i,
                'word1': words1[i],
                'word2': words2[i],
                'context1': ' '.join(words1[max(0, i-3):i+4]),
                'context2': ' '.join(words2[max(0, i-3):i+4])
            })
        
        i += 1
    
    if not differences:
        print("✅ No differences found!")
        return
    
    print(f"\n🔍 FOUND {len(differences)} DIFFERENCE(S):")
    print("=" * 60)
    
    for i, diff in enumerate(differences, 1):
        print(f"\n{i}. Position {diff['position']}:")
        
        if diff['type'] == 'added':
            print(f"   ✅ Added in {label2}: '{diff['word']}'")
            print(f"   Context: {diff['context']}")
        
        elif diff['type'] == 'removed':
            print(f"   ❌ Removed from {label1}: '{diff['word']}'")
            print(f"   Context: {diff['context']}")
        
        elif diff['type'] == 'changed':
            print(f"   🔄 Changed:")
            print(f"   {label1}: '{diff['word1']}'")
            print(f"   {label2}: '{diff['word2']}'")
            print(f"   Context {label1}: {diff['context1']}")
            print(f"   Context {label2}: {diff['context2']}")

def main():
    print("🔍 CLEAN WORD-BY-WORD COMPARISON")
    print("=" * 60)
    
    # Files to compare
    file1 = "chunked_transcription.txt"
    file2 = "mp3_chunked_transcription.txt"
    label1 = "WAV (4 chunks)"
    label2 = "MP3 (2 chunks)"
    
    # Extract transcription texts
    text1 = extract_transcription_text(file1)
    text2 = extract_transcription_text(file2)
    
    if not text1 or not text2:
        print("❌ Could not extract text from files")
        return
    
    # Find differences
    find_exact_differences(text1, text2, label1, label2)

if __name__ == "__main__":
    main()
