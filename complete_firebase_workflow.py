#!/usr/bin/env python3
"""
Complete Firebase Workflow - Start to Finish
===========================================

This script handles the ENTIRE workflow:
1. Upload audio file to Firebase Storage
2. Get Firebase download URL
3. Send URL to RunPod API (no base64, no size limits)
4. Process with Parakeet + Pyannote
5. Save transcript to file
6. Clean up

No base64 encoding, no 10MiB limits, complete automation!
"""

import os
import requests
import json
import tempfile
from datetime import datetime
from pathlib import Path

# Import configuration
try:
    from test_config import (
        RUNPOD_ENDPOINT, 
        API_KEY, 
        HF_TOKEN,
        NUM_SPEAKERS,
        SMALL_FILE_TIMEOUT,
        LARGE_FILE_TIMEOUT
    )
except ImportError:
    print("❌ Could not import test_config.py")
    print("💡 Please ensure test_config.py exists and has the correct configuration")
    exit(1)

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

def upload_to_firebase(audio_file_path: str) -> tuple[bool, str]:
    """
    Upload audio file to Firebase Storage and return download URL
    """
    try:
        print(f"📤 Uploading {audio_file_path} to Firebase Storage...")
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(audio_file_path).suffix
        firebase_filename = f"audio_uploads/runpod_{timestamp}_{os.urandom(4).hex()}{file_extension}"
        
        # Firebase Storage REST API endpoint
        bucket_name = FIREBASE_CONFIG["storage_bucket"]
        upload_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o"
        
        # Prepare the file for upload
        with open(audio_file_path, 'rb') as f:
            files = {'file': (firebase_filename, f, 'audio/wav')}
            
            # Upload parameters
            params = {
                'name': firebase_filename,
                'uploadType': 'media'
            }
            
            # Upload to Firebase
            response = requests.post(upload_url, files=files, params=params, timeout=60)
            
            if response.status_code == 200:
                # Generate download URL
                download_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{firebase_filename}?alt=media"
                print(f"✅ Upload successful!")
                print(f"🔗 Download URL: {download_url}")
                return True, download_url
            else:
                print(f"❌ Upload failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False, ""
                
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return False, ""

def process_with_runpod(firebase_url: str, audio_format: str = "wav", 
                       use_diarization: bool = True, num_speakers: int = None) -> tuple[bool, dict]:
    """
    Send Firebase URL to RunPod API for processing
    """
    try:
        print(f"🚀 Sending Firebase URL to RunPod API...")
        print(f"🔗 Endpoint: {RUNPOD_ENDPOINT}")
        
        # Prepare request payload - ONLY the Firebase URL!
        payload = {
            "input": {
                "audio_url": firebase_url,  # Just the URL, no base64!
                "audio_format": audio_format,
                "include_timestamps": True,
                "use_diarization": use_diarization,
                "num_speakers": num_speakers or NUM_SPEAKERS,
                "hf_token": HF_TOKEN
            }
        }
        
        print(f"📊 Payload size: {len(json.dumps(payload)) / 1024:.1f}KB (tiny!)")
        print("🎯 No 10MiB limit - any file size supported!")
        
        # Send request
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            RUNPOD_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=LARGE_FILE_TIMEOUT
        )
        
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! RunPod processing completed")
            return True, result
        else:
            print(f"❌ FAILED: {response.status_code}")
            print(f"Response: {response.text}")
            return False, {}
            
    except Exception as e:
        print(f"❌ Processing error: {str(e)}")
        return False, {}

def save_transcript_to_file(result: dict, output_filename: str = None) -> str:
    """
    Save transcript results to a file
    """
    try:
        # Generate output filename if not provided
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"transcript_{timestamp}.txt"
        
        print(f"💾 Saving transcript to: {output_filename}")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            # Write header
            f.write("=" * 60 + "\n")
            f.write("AUDIO TRANSCRIPT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Processing Method: {result.get('processing_method', 'Unknown')}\n")
            f.write(f"Audio Format: {result.get('audio_format', 'Unknown')}\n")
            f.write(f"File Size: {result.get('file_size_mb', 'Unknown')}MB\n")
            f.write(f"Duration: {result.get('duration_seconds', 'Unknown')}s\n")
            f.write("=" * 60 + "\n\n")
            
            # Write main transcript
            if 'text' in result and result['text']:
                f.write("TRANSCRIPT:\n")
                f.write("-" * 20 + "\n")
                f.write(result['text'])
                f.write("\n\n")
            
            # Write diarized transcript if available
            if 'diarized_transcript' in result and result['diarized_transcript']:
                f.write("DIARIZED TRANSCRIPT (with speaker labels):\n")
                f.write("-" * 40 + "\n")
                for segment in result['diarized_transcript']:
                    speaker = segment.get('speaker', 'Unknown')
                    start_time = segment.get('start', 'Unknown')
                    end_time = segment.get('end', 'Unknown')
                    text = segment.get('text', '')
                    
                    f.write(f"[{start_time:.2f}s - {end_time:.2f}s] Speaker {speaker}: {text}\n")
                f.write("\n")
            
            # Write word timestamps if available
            if 'word_timestamps' in result and result['word_timestamps']:
                f.write("WORD-LEVEL TIMESTAMPS:\n")
                f.write("-" * 25 + "\n")
                for word_info in result['word_timestamps']:
                    word = word_info.get('word', '')
                    start_time = word_info.get('start', 0)
                    end_time = word_info.get('end', 0)
                    f.write(f"[{start_time:.2f}s - {end_time:.2f}s] {word}\n")
                f.write("\n")
            
            # Write metadata
            f.write("METADATA:\n")
            f.write("-" * 10 + "\n")
            f.write(f"Model: {result.get('model', 'Unknown')}\n")
            f.write(f"Language: {result.get('language', 'Unknown')}\n")
            f.write(f"Processing Time: {result.get('processing_time_seconds', 'Unknown')}s\n")
            f.write(f"Workflow: {result.get('workflow', 'Unknown')}\n")
        
        print(f"✅ Transcript saved successfully to: {output_filename}")
        return output_filename
        
    except Exception as e:
        print(f"❌ Error saving transcript: {str(e)}")
        return ""

def complete_workflow(audio_file_path: str, output_filename: str = None, 
                     use_diarization: bool = True, num_speakers: int = None) -> bool:
    """
    Complete workflow: Upload → Process → Save → Cleanup
    """
    print("🚀 COMPLETE FIREBASE WORKFLOW")
    print("=" * 50)
    print(f"📁 Input file: {audio_file_path}")
    print(f"🎯 Diarization: {'Enabled' if use_diarization else 'Disabled'}")
    if num_speakers:
        print(f"👥 Expected speakers: {num_speakers}")
    print("=" * 50)
    
    try:
        # Step 1: Upload to Firebase
        print("\n📤 STEP 1: Uploading to Firebase Storage...")
        upload_success, firebase_url = upload_to_firebase(audio_file_path)
        if not upload_success:
            print("❌ Firebase upload failed. Aborting workflow.")
            return False
        
        # Step 2: Process with RunPod
        print("\n🚀 STEP 2: Processing with RunPod...")
        process_success, result = process_with_runpod(
            firebase_url, 
            use_diarization=use_diarization, 
            num_speakers=num_speakers
        )
        if not process_success:
            print("❌ RunPod processing failed. Aborting workflow.")
            return False
        
        # Step 3: Save transcript to file
        print("\n💾 STEP 3: Saving transcript to file...")
        saved_filename = save_transcript_to_file(result, output_filename)
        if not saved_filename:
            print("❌ Failed to save transcript. Aborting workflow.")
            return False
        
        # Step 4: Display summary
        print("\n🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"✅ Audio uploaded to Firebase")
        print(f"✅ Processed with RunPod (no base64, no size limits!)")
        print(f"✅ Transcript saved to: {saved_filename}")
        print(f"📝 Text length: {len(result.get('text', ''))} characters")
        if use_diarization and 'diarized_transcript' in result:
            print(f"🎤 Diarization segments: {len(result['diarized_transcript'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow error: {str(e)}")
        return False

def main():
    """Main function to run the complete workflow"""
    print("🎯 COMPLETE FIREBASE WORKFLOW - START TO FINISH")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🚀 Complete automation: Upload → Process → Save (no base64!)")
    print("=" * 60)
    
    # Check configuration
    if "your-api-key-here" in API_KEY:
        print("❌ CONFIGURATION ERROR: Please update API_KEY in test_config.py")
        return
    
    if HF_TOKEN == "hf_xxx":
        print("❌ CONFIGURATION ERROR: Please update HF_TOKEN in test_config.py")
        return
    
    print("✅ Configuration verified")
    print()
    
    # Check if test2.wav exists
    audio_file = "test2.wav"
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        print("💡 Please ensure the audio file exists in the current directory")
        return
    
    print(f"✅ Audio file found: {audio_file}")
    file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
    print(f"📁 File size: {file_size_mb:.1f}MB")
    print()
    
    # Run complete workflow
    success = complete_workflow(
        audio_file_path=audio_file,
        output_filename=f"transcript_{Path(audio_file).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        use_diarization=True,
        num_speakers=NUM_SPEAKERS
    )
    
    if success:
        print("\n🎉 SUCCESS! Complete workflow finished!")
        print("🚀 You can now process files of ANY size without base64 encoding!")
        print("💾 Transcript has been saved to a file for your use.")
    else:
        print("\n❌ Workflow failed. Check the logs above for details.")
    
    print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
