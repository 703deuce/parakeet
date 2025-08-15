#!/usr/bin/env python3
"""
Find the exact word difference between two transcriptions
"""

import os
import difflib

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

def find_word_differences(text1, text2, label1, label2):
    """Find the exact word differences between two texts"""
    
    words1 = text1.split()
    words2 = text2.split()
    
    print(f"📊 Word counts:")
    print(f"  {label1}: {len(words1)} words")
    print(f"  {label2}: {len(words2)} words")
    print(f"  Difference: {len(words2) - len(words1):+d} words")
    
    # Use difflib to find differences
    differ = difflib.unified_diff(
        words1, 
        words2, 
        fromfile=label1, 
        tofile=label2, 
        lineterm='',
        n=3  # Show 3 words of context
    )
    
    diff_lines = list(differ)
    
    if len(diff_lines) <= 2:  # Only headers, no differences
        print("\n✅ No word differences found!")
        return
    
    print(f"\n🔍 WORD DIFFERENCES:")
    print("=" * 50)
    
    # Parse the diff output
    added_words = []
    removed_words = []
    context_before = []
    context_after = []
    
    for line in diff_lines[2:]:  # Skip headers
        if line.startswith('-'):
            removed_words.append(line[1:])
        elif line.startswith('+'):
            added_words.append(line[1:])
        elif line.startswith(' '):
            if not added_words and not removed_words:
                context_before.append(line[1:])
            else:
                context_after.append(line[1:])
    
    # Show the differences with context
    if removed_words or added_words:
        print("Context before:", " ".join(context_before[-3:]))
        
        if removed_words:
            print(f"❌ Removed from {label1}: {' '.join(removed_words)}")
        
        if added_words:
            print(f"✅ Added in {label2}: {' '.join(added_words)}")
        
        print("Context after:", " ".join(context_after[:3]))
    
    # Also try character-level diff for more precision
    print(f"\n🔍 CHARACTER-LEVEL ANALYSIS:")
    print("=" * 50)
    
    # Find the first difference
    min_len = min(len(text1), len(text2))
    first_diff_pos = -1
    
    for i in range(min_len):
        if text1[i] != text2[i]:
            first_diff_pos = i
            break
    
    if first_diff_pos != -1:
        start = max(0, first_diff_pos - 50)
        end = min(len(text1), first_diff_pos + 50)
        
        print(f"First difference at position {first_diff_pos}:")
        print(f"{label1}: ...{text1[start:end]}...")
        print(f"{label2}: ...{text2[start:end]}...")
    elif len(text1) != len(text2):
        # Length difference at the end
        shorter = text1 if len(text1) < len(text2) else text2
        longer = text2 if len(text1) < len(text2) else text1
        shorter_label = label1 if len(text1) < len(text2) else label2
        longer_label = label2 if len(text1) < len(text2) else label1
        
        print(f"Length difference at the end:")
        print(f"{shorter_label} ends with: ...{shorter[-100:]}")
        print(f"{longer_label} ends with: ...{longer[-100:]}")
        print(f"Extra text in {longer_label}: '{longer[len(shorter):]}'")

def main():
    print("🔍 FINDING EXACT WORD DIFFERENCES")
    print("=" * 60)
    
    # Files to compare
    file1 = "chunked_transcription.txt"
    file2 = "mp3_chunked_transcription.txt"
    label1 = "WAV (4 chunks)"
    label2 = "MP3 (2 chunks)"
    
    # Extract transcription texts
    text1 = extract_transcription_text(file1)
    text2 = extract_transcription_text(file2)
    
    if not text1:
        print(f"❌ Could not extract text from {file1}")
        return
    
    if not text2:
        print(f"❌ Could not extract text from {file2}")
        return
    
    print(f"✅ Extracted transcriptions from both files")
    
    # Find differences
    find_word_differences(text1, text2, label1, label2)

if __name__ == "__main__":
    main()
