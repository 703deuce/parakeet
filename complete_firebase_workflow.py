#!/usr/bin/env python3
"""
Complete Firebase Workflow - Start to Finish with Polling
========================================================

This script performs the COMPLETE workflow:
1. Upload audio file to Firebase Storage
2. Send Firebase URL to RunPod
3. Poll RunPod for job completion (WITH PROPER POLLING!)
4. Extract transcript from completed job
5. Save transcript to text file

This is the END-TO-END solution that goes from raw audio to final transcript file.
"""

import os
import requests
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from test_config import (
    RUNPOD_ENDPOINT, 
    API_KEY, 
    HF_TOKEN,
    NUM_SPEAKERS,
    LARGE_FILE_TIMEOUT
)

# Firebase configuration
FIREBASE_CONFIG = {
    "api_key": "AIzaSyASdf98Soi-LtMowVOQMhQvMWWVEP3KoC8",
    "auth_domain": "aitts-d4c6d.firebaseapp.com",
    "project_id": "aitts-d4c6d",
    "storage_bucket": "aitts-d4c6d.firebasestorage.app",
    "messaging_sender_id": "927299361889",
    "app_id": "1:927299361889:web:13408945d50bda7a2f5e20",
    "measurement_id": "G-P1TK2HHBXR"
}

def verify_firebase_file_available(download_url: str, max_wait_time: int = 600, check_interval: int = 30) -> bool:
    """
    Verify that Firebase file is available for download after upload
    Wait up to max_wait_time seconds (default 10 minutes), checking every check_interval seconds (default 30s)
    
    Based on real-world timings:
    - 11MB file: ~3 minutes
    - 60MB file: could take 10+ minutes
    """
    print(f"🔍 Verifying file availability (max wait: {max_wait_time//60} minutes, check every {check_interval}s)...")
    print("⏳ Large files can take several minutes to upload to Firebase...")
    
    start_time = time.time()
    attempts = 0
    
    while time.time() - start_time < max_wait_time:
        attempts += 1
        elapsed_time = time.time() - start_time
        
        try:
            # Make a HEAD request to check if file is available without downloading
            response = requests.head(download_url, timeout=15)
            
            if response.status_code == 200:
                file_size = response.headers.get('content-length', 'Unknown')
                print(f"✅ File verified available! (attempt {attempts}, elapsed: {elapsed_time:.1f}s, size: {file_size} bytes)")
                return True
            elif response.status_code == 404:
                print(f"⏳ File not yet available (attempt {attempts}, elapsed: {elapsed_time:.1f}s), waiting {check_interval}s...")
                time.sleep(check_interval)
            else:
                print(f"❌ Unexpected status {response.status_code} (attempt {attempts}, elapsed: {elapsed_time:.1f}s)")
                time.sleep(check_interval)
                
        except Exception as e:
            print(f"⏳ Connection error (attempt {attempts}, elapsed: {elapsed_time:.1f}s): {str(e)[:50]}...")
            time.sleep(check_interval)
    
    elapsed_minutes = (time.time() - start_time) / 60
    print(f"❌ File not available after {elapsed_minutes:.1f} minutes and {attempts} attempts")
    print(f"💡 Large files (60MB+) may need even longer upload times")
    return False

def upload_to_firebase(audio_file_path: str) -> tuple[bool, str]:
    """
    Upload audio file to Firebase Storage and return download URL
    """
    try:
        print(f"\n📤 STEP 1: Uploading to Firebase Storage...")
        print(f"📁 File: {audio_file_path}")
        
        if not os.path.exists(audio_file_path):
            print(f"❌ File not found: {audio_file_path}")
            return False, ""
        
        file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
        print(f"📊 File size: {file_size_mb:.1f}MB")
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(audio_file_path).suffix
        firebase_filename = f"transcription_uploads/upload_{timestamp}_{uuid.uuid4().hex[:8]}{file_extension}"
        
        # Firebase Storage REST API endpoint
        bucket_name = FIREBASE_CONFIG["storage_bucket"]
        upload_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o"
        
        print(f"🔗 Upload URL: {upload_url}")
        print(f"📝 Firebase path: {firebase_filename}")
        
        # Prepare the file for upload
        with open(audio_file_path, 'rb') as f:
            files = {'file': (firebase_filename, f, 'application/octet-stream')}
            
            # Upload parameters
            params = {
                'name': firebase_filename,
                'uploadType': 'media'
            }
            
            print("📤 Uploading to Firebase...")
            print("⏳ Large file upload may take several minutes...")
            response = requests.post(upload_url, files=files, params=params, timeout=300)  # 5 minutes for large files
        
        if response.status_code == 200:
            upload_result = response.json()
            # Generate download URL (Firebase provides token in metadata)
            download_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{firebase_filename.replace('/', '%2F')}?alt=media&token={upload_result.get('downloadTokens', '')}"
            print(f"✅ Upload successful! (Size: {os.path.getsize(audio_file_path) / (1024 * 1024):.1f}MB)")
            print(f"🔗 Download URL: {download_url[:100]}...") # Truncate for display
            
            # CRITICAL: Wait for file to be available for download
            if verify_firebase_file_available(download_url):
                return True, download_url
            else:
                print("❌ Firebase file not available after upload")
                return False, ""
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False, ""
            
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return False, ""

def poll_runpod_job(job_id: str, max_wait_minutes: int = 15) -> tuple[bool, dict]:
    """
    Poll RunPod job until completion
    """
    print(f"\n📥 STEP 3: Polling RunPod job for completion...")
    print(f"🆔 Job ID: {job_id}")
    print(f"⏱️ Max wait time: {max_wait_minutes} minutes")
    
    # Extract endpoint ID from the main endpoint for status polling
    # RUNPOD_ENDPOINT format: https://api.runpod.ai/v2/{endpoint_id}/run
    # Status endpoint format: https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}
    endpoint_parts = RUNPOD_ENDPOINT.split('/')
    endpoint_id = endpoint_parts[-2]  # Get the endpoint ID
    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    check_interval = 30  # Check every 30 seconds
    attempt = 0
    
    while time.time() - start_time < max_wait_seconds:
        attempt += 1
        elapsed_time = time.time() - start_time
        elapsed_minutes = elapsed_time / 60
        
        try:
            print(f"🔄 Polling attempt {attempt} (elapsed: {elapsed_minutes:.1f} min)...")
            
            response = requests.get(status_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                status = result.get('status', 'UNKNOWN')
                
                print(f"📊 Job status: {status}")
                
                if status == 'COMPLETED':
                    print("✅ Job completed successfully!")
                    output = result.get('output', {})
                    print(f"🔍 Raw job output: {output}")
                    return True, output
                elif status == 'FAILED':
                    print("❌ Job failed!")
                    error = result.get('error', 'Unknown error')
                    print(f"Error: {error}")
                    return False, {}
                elif status in ['IN_QUEUE', 'IN_PROGRESS']:
                    print(f"⏳ Job {status.lower()}, waiting {check_interval}s...")
                    time.sleep(check_interval)
                else:
                    print(f"❓ Unknown status: {status}")
                    time.sleep(check_interval)
            else:
                print(f"❌ Polling failed: {response.status_code}")
                print(f"Response: {response.text}")
                time.sleep(check_interval)
                
        except Exception as e:
            print(f"❌ Polling error: {str(e)}")
            time.sleep(check_interval)
    
    elapsed_minutes = (time.time() - start_time) / 60
    print(f"❌ Job not completed after {elapsed_minutes:.1f} minutes and {attempt} attempts")
    return False, {}

def send_to_runpod(firebase_url: str) -> tuple[bool, str]:
    """
    Send Firebase URL to RunPod and return job ID
    """
    try:
        print(f"\n📤 STEP 2: Sending to RunPod...")
        print(f"🔗 Firebase URL: {firebase_url[:100]}...")
        
        payload = {
            "input": {
                "audio_url": firebase_url,
                "audio_format": "wav",
                "include_timestamps": True,
                "use_diarization": True,
                "num_speakers": NUM_SPEAKERS,
                "hf_token": HF_TOKEN
            }
        }
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        print("📤 Submitting job to RunPod...")
        response = requests.post(
            RUNPOD_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=60  # Quick timeout for job submission
        )
        
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"🔍 Raw RunPod response: {result}")
            
            # Check if this is an async job or immediate result
            if 'id' in result and 'status' in result:
                # Async job - need to poll
                job_id = result['id']
                status = result['status']
                print(f"🔄 Async job submitted: {job_id}")
                print(f"📊 Initial status: {status}")
                return True, job_id
            else:
                # Immediate result - no polling needed
                print("✅ Immediate result received!")
                # This is already the final result, treat as completed job
                return True, result
                
        else:
            print(f"❌ Job submission failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False, ""
            
    except Exception as e:
        print(f"❌ RunPod submission error: {str(e)}")
        return False, ""

def save_transcript_to_file(result: dict, original_filename: str) -> str:
    """
    Save transcript results to a text file
    """
    print(f"\n📝 STEP 4: Saving transcript to file...")
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(original_filename).stem
    output_filename = f"transcript_{base_name}_{timestamp}.txt"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        # Write header
        f.write("=" * 60 + "\n")
        f.write("PARAKEET ASR + PYANNOTE DIARIZATION TRANSCRIPT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source file: {original_filename}\n")
        f.write(f"Workflow: {result.get('workflow', 'Unknown')}\n")
        f.write(f"Processing method: {result.get('processing_method', 'Unknown')}\n")
        f.write(f"File size: {result.get('file_size_mb', 'Unknown')}MB\n")
        f.write(f"Duration: {result.get('duration_seconds', 'Unknown')}s\n")
        f.write("=" * 60 + "\n\n")
        
        # Write main transcript (check for diarized workflow first)
        main_text = result.get('merged_text') or result.get('text', '')
        if main_text:
            f.write("TRANSCRIPT:\n")
            f.write("-" * 20 + "\n")
            f.write(main_text)
            f.write("\n\n")
        
        # Write diarized transcript if available
        if 'diarized_transcript' in result and result['diarized_transcript']:
            f.write("SPEAKER DIARIZATION:\n")
            f.write("-" * 30 + "\n")
            for segment in result['diarized_transcript']:
                speaker = segment.get('speaker', 'Unknown')
                text = segment.get('text', '')
                start = segment.get('start_time', 0)
                end = segment.get('end_time', 0)
                f.write(f"[{start:.1f}s - {end:.1f}s] {speaker}: {text}\n")
            f.write("\n")
        
        # Write metadata
        f.write("PROCESSING METADATA:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Total characters: {len(main_text)}\n")
        f.write(f"Workflow type: {result.get('workflow', 'Unknown')}\n")
        f.write(f"Processing advantages: {result.get('advantages', [])}\n")
        
        if 'diarized_transcript' in result:
            f.write(f"Speaker segments: {len(result['diarized_transcript'])}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("End of transcript\n")
    
    print(f"✅ Transcript saved to: {output_filename}")
    return output_filename

def complete_workflow(audio_file_path: str) -> bool:
    """
    Complete workflow from audio file to transcript file
    """
    print("🚀 COMPLETE FIREBASE → RUNPOD → TRANSCRIPT WORKFLOW")
    print("=" * 70)
    print(f"📁 Input file: {audio_file_path}")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # Step 1: Upload to Firebase
        upload_success, firebase_url = upload_to_firebase(audio_file_path)
        if not upload_success:
            print("❌ Firebase upload failed")
            return False
        
        # Step 2: Send to RunPod
        submit_success, job_result = send_to_runpod(firebase_url)
        if not submit_success:
            print("❌ RunPod submission failed")
            return False
        
        # Step 3: Handle job result (polling if needed)
        if isinstance(job_result, str):
            # This is a job ID - need to poll
            poll_success, final_result = poll_runpod_job(job_result)
            if not poll_success:
                print("❌ RunPod job polling failed")
                return False
        else:
            # This is already the final result
            final_result = job_result
            print("✅ Immediate result received, no polling needed")
        
        # Step 4: Save transcript
        saved_filename = save_transcript_to_file(final_result, audio_file_path)
        
        # Success summary
        print("\n" + "=" * 70)
        print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"✅ Audio uploaded to Firebase")
        print(f"✅ Processed with RunPod (no base64, no size limits!)")
        print(f"✅ Transcript saved to: {saved_filename}")
        main_text = final_result.get('merged_text') or final_result.get('text', '')
        print(f"📝 Text length: {len(main_text)} characters")
        if final_result.get('use_diarization') and 'diarized_transcript' in final_result:
            print(f"🎤 Diarization segments: {len(final_result['diarized_transcript'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow error: {str(e)}")
        return False

def main():
    """Main function"""
    
    # Check if test audio file exists
    test_file = "test2.wav"
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        print("💡 Please ensure test2.wav is in the current directory")
        return
    
    # Check configuration
    if "your-api-key-here" in API_KEY:
        print("❌ CONFIGURATION ERROR: Please update API_KEY in test_config.py")
        return
    
    if HF_TOKEN == "hf_xxx":
        print("❌ CONFIGURATION ERROR: Please update HF_TOKEN in test_config.py")
        return
    
    print("✅ Configuration verified")
    
    # Run complete workflow
    success = complete_workflow(test_file)
    
    if success:
        print(f"\n🎉 SUCCESS! Complete workflow finished!")
        print("🚀 Your transcript file is ready!")
    else:
        print(f"\n❌ FAILED! Workflow encountered errors")
        print("💡 Check the output above for details")

if __name__ == "__main__":
    main()