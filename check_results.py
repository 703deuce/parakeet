#!/usr/bin/env python3
"""
Check transcription results and save to text file
"""

import requests
import json
import os
import time

def check_job_status(job_id, api_key):
    """Check the status of a RunPod job"""
    
    # RunPod status endpoint
    status_url = f"https://api.runpod.ai/v2/7u304yobo6ytm9/status/{job_id}"
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(status_url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error checking status: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def save_transcription_to_txt(result_data, output_file):
    """Save transcription results to a text file"""
    
    if not result_data or 'output' not in result_data:
        print("No transcription data found")
        return False
    
    output = result_data['output']
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write the main transcription text
        text = output.get('text', '')
        f.write("TRANSCRIPTION:\n")
        f.write("=" * 50 + "\n")
        f.write(text + "\n\n")
        
        # Write metadata
        f.write("METADATA:\n")
        f.write("=" * 50 + "\n")
        f.write(f"Audio Duration: {output.get('audio_duration_seconds', 0):.2f} seconds\n")
        f.write(f"Chunks Processed: {output.get('chunks_processed', 0)}\n")
        f.write(f"Model Used: {output.get('model_used', 'N/A')}\n\n")
        
        # Write timestamps if available
        if output.get('word_timestamps'):
            f.write("WORD TIMESTAMPS:\n")
            f.write("=" * 50 + "\n")
            for word_ts in output['word_timestamps']:
                f.write(f"{word_ts.get('start', 0):.2f}s - {word_ts.get('end', 0):.2f}s: {word_ts.get('word', 'N/A')}\n")
            f.write("\n")
        
        if output.get('segment_timestamps'):
            f.write("SEGMENT TIMESTAMPS:\n")
            f.write("=" * 50 + "\n")
            for seg_ts in output['segment_timestamps']:
                f.write(f"{seg_ts.get('start', 0):.2f}s - {seg_ts.get('end', 0):.2f}s: {seg_ts.get('segment', 'N/A')}\n")
    
    print(f"✅ Transcription saved to: {output_file}")
    return True

def main():
    # Get API key
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        print("Error: RUNPOD_API_KEY environment variable not set")
        return
    
    # Check if we have recent result files
    result_files = ['debug_audio_result.json', 'transcription_result_fixed.json', 'transcription_result.json', 'transcription_with_timestamps.json']
    
    for result_file in result_files:
        if os.path.exists(result_file):
            print(f"Checking {result_file}...")
            
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            if 'id' in data:
                job_id = data['id']
                print(f"Job ID: {job_id}")
                print(f"Current status: {data.get('status', 'Unknown')}")
                
                if data.get('status') == 'IN_QUEUE':
                    print("Job still in queue, checking current status...")
                    current_status = check_job_status(job_id, api_key)
                    
                    if current_status:
                        print(f"Updated status: {current_status.get('status', 'Unknown')}")
                        
                        if current_status.get('status') == 'COMPLETED':
                            # Save updated results
                            with open(result_file, 'w') as f:
                                json.dump(current_status, f, indent=2)
                            
                            # Extract transcription to text file
                            txt_file = result_file.replace('.json', '.txt')
                            if save_transcription_to_txt(current_status, txt_file):
                                print(f"🎉 Transcription completed and saved to {txt_file}")
                            break
                        elif current_status.get('status') == 'FAILED':
                            print(f"❌ Job failed: {current_status.get('error', 'Unknown error')}")
                        else:
                            print(f"⏳ Job still processing... Status: {current_status.get('status')}")
                
                elif data.get('status') == 'COMPLETED':
                    # Already completed, just extract to text
                    txt_file = result_file.replace('.json', '.txt')
                    if save_transcription_to_txt(data, txt_file):
                        print(f"🎉 Transcription already completed, saved to {txt_file}")
                    break
            
            print()

if __name__ == "__main__":
    main()
