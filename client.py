#!/usr/bin/env python3
"""
Simple Python client for Parakeet Transcription API
"""

import base64
import requests
import json
from typing import Optional, Dict, Any

class ParakeetClient:
    """Client for interacting with Parakeet Transcription API"""
    
    def __init__(self, api_endpoint: str):
        """
        Initialize client with API endpoint
        
        Args:
            api_endpoint: RunPod serverless endpoint URL
        """
        self.api_endpoint = api_endpoint
    
    def transcribe_file(self, 
                       audio_file_path: str, 
                       include_timestamps: bool = False,
                       audio_format: Optional[str] = None,
                       chunk_duration: int = 1200) -> Optional[Dict[str, Any]]:
        """
        Transcribe an audio file
        
        Args:
            audio_file_path: Path to audio file
            include_timestamps: Whether to include timestamps
            audio_format: Audio format (auto-detected if None)
            chunk_duration: Chunk duration in seconds (default 20 minutes)
            
        Returns:
            Transcription result dictionary or None if failed
        """
        
        # Auto-detect format if not provided
        if audio_format is None:
            audio_format = audio_file_path.split('.')[-1].lower()
        
        # Read and encode audio file
        try:
            with open(audio_file_path, 'rb') as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"Error reading audio file: {e}")
            return None
        
        return self.transcribe_base64(audio_data, include_timestamps, audio_format, chunk_duration)
    
    def transcribe_base64(self, 
                         audio_data: str,
                         include_timestamps: bool = False,
                         audio_format: str = "wav",
                         chunk_duration: int = 1200) -> Optional[Dict[str, Any]]:
        """
        Transcribe base64 encoded audio data
        
        Args:
            audio_data: Base64 encoded audio data
            include_timestamps: Whether to include timestamps
            audio_format: Audio format
            chunk_duration: Chunk duration in seconds
            
        Returns:
            Transcription result dictionary or None if failed
        """
        
        payload = {
            "input": {
                "audio_data": audio_data,
                "audio_format": audio_format,
                "include_timestamps": include_timestamps,
                "chunk_duration": chunk_duration
            }
        }
        
        try:
            response = requests.post(self.api_endpoint, json=payload, timeout=600)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
    
    def transcribe_bytes(self,
                        audio_bytes: bytes,
                        include_timestamps: bool = False,
                        audio_format: str = "wav",
                        chunk_duration: int = 1200) -> Optional[Dict[str, Any]]:
        """
        Transcribe audio from bytes
        
        Args:
            audio_bytes: Raw audio bytes
            include_timestamps: Whether to include timestamps
            audio_format: Audio format
            chunk_duration: Chunk duration in seconds
            
        Returns:
            Transcription result dictionary or None if failed
        """
        
        audio_data = base64.b64encode(audio_bytes).decode('utf-8')
        return self.transcribe_base64(audio_data, include_timestamps, audio_format, chunk_duration)

# Example usage
if __name__ == "__main__":
    # Initialize client with your RunPod endpoint
    client = ParakeetClient("https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync")
    
    # Example 1: Transcribe a file without timestamps
    result = client.transcribe_file("audio.wav", include_timestamps=False)
    if result:
        print("Transcription:", result['text'])
    
    # Example 2: Transcribe with timestamps
    result = client.transcribe_file("audio.wav", include_timestamps=True)
    if result:
        print("Transcription:", result['text'])
        print(f"Word timestamps: {len(result.get('word_timestamps', []))} words")
    
    # Example 3: Transcribe base64 data directly
    with open("audio.wav", "rb") as f:
        audio_bytes = f.read()
    
    result = client.transcribe_bytes(audio_bytes, include_timestamps=True)
    if result:
        print("Duration:", result.get('audio_duration_seconds', 0), "seconds")
        print("Chunks processed:", result.get('chunks_processed', 1))
