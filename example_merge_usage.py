#!/usr/bin/env python3
"""
Example: How to manually merge diarization results
This shows the different ways to use the merge_diarization_chunks.py script
"""

import json
from merge_diarization_chunks import stitch_diarized_chunks, save_merged_transcript

def example_manual_merge():
    """
    Example of manually merging diarization results
    """
    print("📚 EXAMPLE: Manual Diarization Merge")
    print("=" * 50)
    
    # Example 1: Load from your existing results file
    print("\n1️⃣ Loading from existing results file...")
    
    try:
        # Try to load one of your existing result files
        result_files = [
            "mp3_chunked_diarization_results.json",
            "mp4_chunked_diarization_results.json", 
            "mp3_explicit_speakers_diarization_results.json"
        ]
        
        loaded_file = None
        for file in result_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    if 'results' in data and data['results']:
                        loaded_file = file
                        break
            except FileNotFoundError:
                continue
        
        if loaded_file:
            print(f"✅ Found and loaded: {loaded_file}")
            
            # Extract chunk outputs
            chunk_outputs = []
            for result in data['results']:
                if result and 'output' in result:
                    chunk_outputs.append(result['output'])
            
            print(f"📊 Found {len(chunk_outputs)} chunk outputs")
            
            # Merge them
            print("\n2️⃣ Merging chunks...")
            final_text, speaker_map, all_segments = stitch_diarized_chunks(chunk_outputs)
            
            if final_text:
                print("\n3️⃣ Saving merged transcript...")
                save_merged_transcript(final_text, speaker_map, all_segments, f"example_merged_{loaded_file.split('.')[0]}")
                
                print(f"\n🎉 Successfully merged {len(chunk_outputs)} chunks!")
                print(f"📊 Total segments: {len(all_segments)}")
                print(f"👥 Speakers: {list(speaker_map.values())}")
                
                # Show a preview
                print(f"\n📖 PREVIEW (first 500 chars):")
                print("-" * 40)
                print(final_text[:500])
                if len(final_text) > 500:
                    print(f"\n... and {len(final_text) - 500} more characters")
            else:
                print("❌ Failed to merge transcript")
        else:
            print("❌ No valid result files found")
            print("   Make sure you have run a diarization test first")
            
    except Exception as e:
        print(f"❌ Error in example: {e}")
        import traceback
        traceback.print_exc()

def example_custom_merge():
    """
    Example of merging custom diarization data
    """
    print("\n🔧 EXAMPLE: Custom Data Merge")
    print("=" * 40)
    
    # Create sample diarization data (like what your API returns)
    sample_chunks = [
        {
            'diarized_transcript': [
                {
                    'speaker': 'spk0',
                    'start_time': 0.0,
                    'end_time': 45.2,
                    'text': 'Bruno Mars takes his throne back is what I have written.'
                },
                {
                    'speaker': 'spk1', 
                    'start_time': 45.2,
                    'end_time': 89.7,
                    'text': 'Yeah, we are recording this on Thursday evening.'
                }
            ]
        },
        {
            'diarized_transcript': [
                {
                    'speaker': 'spk0',
                    'start_time': 301.0,
                    'end_time': 346.2,
                    'text': 'And I think Chris Brown might drop this year.'
                },
                {
                    'speaker': 'spk1',
                    'start_time': 346.2,
                    'end_time': 390.7,
                    'text': 'I would not be surprised.'
                }
            ]
        }
    ]
    
    print(f"📝 Sample data: {len(sample_chunks)} chunks with diarization")
    
    # Merge the sample data
    final_text, speaker_map, all_segments = stitch_diarized_chunks(sample_chunks)
    
    if final_text:
        print(f"\n✅ Merged sample data successfully!")
        print(f"📊 Total segments: {len(all_segments)}")
        print(f"👥 Speakers: {list(speaker_map.values())}")
        
        print(f"\n📖 MERGED SAMPLE TRANSCRIPT:")
        print("-" * 40)
        print(final_text)
        
        # Save it
        save_merged_transcript(final_text, speaker_map, all_segments, "example_sample_merged")
    else:
        print("❌ Failed to merge sample data")

if __name__ == "__main__":
    print("🎤 DIARIZATION MERGE EXAMPLES")
    print("=" * 50)
    print("This script shows different ways to merge diarization results")
    
    # Run examples
    example_manual_merge()
    example_custom_merge()
    
    print(f"\n🎯 SUMMARY:")
    print("✅ Created merge_diarization_chunks.py - main merging script")
    print("✅ Created example_merge_usage.py - this example file")
    print("✅ Ready to merge your diarization results once RunPod rebuilds!")
    
    print(f"\n📋 USAGE AFTER REBUILD:")
    print("1. Run your diarization test to get new results")
    print("2. Run: python merge_diarization_chunks.py")
    print("3. Or manually: from merge_diarization_chunks import merge_from_json_file")
    print("4. Get unified transcript with consistent speaker labels!")
