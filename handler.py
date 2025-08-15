import runpod
import torch
import torchaudio
import numpy as np
import base64
import io
import tempfile
import os
from typing import Dict, Any, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None

def load_model():
    """Load the NVIDIA Parakeet model"""
    global model
    try:
        import nemo.collections.asr as nemo_asr
        logger.info("Loading NVIDIA Parakeet TDT 0.6B v2 model...")
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v2"
        )
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds"""
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        duration = waveform.shape[1] / sample_rate
        return duration
    except Exception as e:
        logger.error(f"Error getting audio duration: {str(e)}")
        return 0

def split_audio(audio_path: str, chunk_duration: int = 1200) -> List[str]:
    """
    Split audio into chunks of specified duration (default 20 minutes = 1200 seconds)
    Returns list of temporary file paths for each chunk
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        total_duration = waveform.shape[1] / sample_rate
        
        if total_duration <= chunk_duration:
            return [audio_path]
        
        chunk_files = []
        chunk_samples = chunk_duration * sample_rate
        
        for i in range(0, waveform.shape[1], chunk_samples):
            chunk_waveform = waveform[:, i:i + chunk_samples]
            
            # Create temporary file for chunk
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            torchaudio.save(temp_file.name, chunk_waveform, sample_rate)
            chunk_files.append(temp_file.name)
            temp_file.close()
        
        logger.info(f"Split audio into {len(chunk_files)} chunks")
        return chunk_files
        
    except Exception as e:
        logger.error(f"Error splitting audio: {str(e)}")
        return [audio_path]

def transcribe_audio_chunk(audio_path: str, include_timestamps: bool = False) -> Dict[str, Any]:
    """Transcribe a single audio chunk"""
    try:
        if include_timestamps:
            output = model.transcribe([audio_path], timestamps=True)
            result = {
                'text': output[0].text,
                'word_timestamps': output[0].timestamp.get('word', []),
                'segment_timestamps': output[0].timestamp.get('segment', []),
                'char_timestamps': output[0].timestamp.get('char', [])
            }
        else:
            output = model.transcribe([audio_path])
            result = {
                'text': output[0].text,
                'word_timestamps': [],
                'segment_timestamps': [],
                'char_timestamps': []
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error transcribing audio chunk: {str(e)}")
        return {
            'text': '',
            'word_timestamps': [],
            'segment_timestamps': [],
            'char_timestamps': [],
            'error': str(e)
        }

def merge_transcription_results(chunk_results: List[Dict[str, Any]], chunk_duration: int = 1200) -> Dict[str, Any]:
    """Merge transcription results from multiple chunks"""
    merged_text = []
    merged_word_timestamps = []
    merged_segment_timestamps = []
    merged_char_timestamps = []
    
    time_offset = 0
    
    for i, chunk_result in enumerate(chunk_results):
        if 'error' in chunk_result:
            continue
            
        merged_text.append(chunk_result['text'])
        
        # Adjust timestamps by adding time offset
        if chunk_result['word_timestamps']:
            for word_ts in chunk_result['word_timestamps']:
                adjusted_word_ts = word_ts.copy()
                adjusted_word_ts['start'] += time_offset
                adjusted_word_ts['end'] += time_offset
                merged_word_timestamps.append(adjusted_word_ts)
        
        if chunk_result['segment_timestamps']:
            for seg_ts in chunk_result['segment_timestamps']:
                adjusted_seg_ts = seg_ts.copy()
                adjusted_seg_ts['start'] += time_offset
                adjusted_seg_ts['end'] += time_offset
                merged_segment_timestamps.append(adjusted_seg_ts)
        
        if chunk_result['char_timestamps']:
            for char_ts in chunk_result['char_timestamps']:
                adjusted_char_ts = char_ts.copy()
                adjusted_char_ts['start'] += time_offset
                adjusted_char_ts['end'] += time_offset
                merged_char_timestamps.append(adjusted_char_ts)
        
        time_offset += chunk_duration
    
    return {
        'text': ' '.join(merged_text),
        'word_timestamps': merged_word_timestamps,
        'segment_timestamps': merged_segment_timestamps,
        'char_timestamps': merged_char_timestamps
    }

def cleanup_temp_files(file_paths: List[str]):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Could not delete temp file {file_path}: {str(e)}")

def handler(job):
    """
    RunPod handler function for audio transcription
    
    Expected input format:
    {
        "input": {
            "audio_data": "base64_encoded_audio_data",  # Base64 encoded audio file
            "audio_format": "wav",  # Optional: audio format (wav, mp3, flac, etc.)
            "include_timestamps": true,  # Optional: include word/segment timestamps
            "chunk_duration": 1200  # Optional: chunk duration in seconds (default 20 minutes)
        }
    }
    """
    try:
        job_input = job["input"]
        
        # Validate required inputs
        if "audio_data" not in job_input:
            return {"error": "Missing required parameter: audio_data"}
        
        # Get parameters
        audio_data = job_input["audio_data"]
        audio_format = job_input.get("audio_format", "wav")
        include_timestamps = job_input.get("include_timestamps", False)
        chunk_duration = job_input.get("chunk_duration", 1200)  # 20 minutes default
        
        logger.info(f"Processing transcription request with timestamps: {include_timestamps}")
        
        # Decode base64 audio data
        try:
            audio_bytes = base64.b64decode(audio_data)
        except Exception as e:
            return {"error": f"Invalid base64 audio data: {str(e)}"}
        
        # Save audio to temporary file
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_format}')
        temp_audio_file.write(audio_bytes)
        temp_audio_file.close()
        
        temp_files_to_cleanup = [temp_audio_file.name]
        
        try:
            # Check audio duration
            duration = audio_duration(temp_audio_file.name)
            logger.info(f"Audio duration: {duration:.2f} seconds")
            
            # Split audio if necessary
            audio_chunks = split_audio(temp_audio_file.name, chunk_duration)
            temp_files_to_cleanup.extend(audio_chunks)
            
            # Transcribe each chunk
            chunk_results = []
            for i, chunk_path in enumerate(audio_chunks):
                logger.info(f"Transcribing chunk {i+1}/{len(audio_chunks)}")
                chunk_result = transcribe_audio_chunk(chunk_path, include_timestamps)
                chunk_results.append(chunk_result)
            
            # Merge results
            if len(chunk_results) == 1:
                final_result = chunk_results[0]
            else:
                final_result = merge_transcription_results(chunk_results, chunk_duration)
            
            # Add metadata
            final_result.update({
                'audio_duration_seconds': duration,
                'chunks_processed': len(audio_chunks),
                'model_used': 'nvidia/parakeet-tdt-0.6b-v2'
            })
            
            return final_result
            
        finally:
            # Clean up temporary files
            cleanup_temp_files(temp_files_to_cleanup)
            
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        return {"error": f"Transcription failed: {str(e)}"}

# Initialize model when the container starts
if __name__ == "__main__":
    logger.info("Initializing model...")
    if load_model():
        logger.info("Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})
    else:
        logger.error("Failed to load model. Exiting.")
        exit(1)
