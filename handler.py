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

def detect_silence_split_points(audio_path: str, target_chunk_duration: int, audio_format: str = "wav"):
    """
    Find optimal split points near target duration that fall on silence
    Uses pydub for silence detection
    """
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent
        
        # Load audio with pydub
        if audio_format.lower() == "mp3":
            audio = AudioSegment.from_mp3(audio_path)
        elif audio_format.lower() == "wav":
            audio = AudioSegment.from_wav(audio_path)
        else:
            audio = AudioSegment.from_file(audio_path)
        
        target_chunk_duration_ms = target_chunk_duration * 1000
        silence_thresh = -40  # dBFS
        min_silence_len = 500  # ms
        
        logger.info(f"Analyzing audio for silence points (target: {target_chunk_duration}s)")
        
        # Detect non-silent segments
        nonsilent_ranges = detect_nonsilent(
            audio, 
            min_silence_len=min_silence_len, 
            silence_thresh=silence_thresh
        )
        
        if not nonsilent_ranges:
            logger.warning("No silence detected, using time-based splitting")
            return list(range(0, len(audio), target_chunk_duration_ms))
        
        logger.info(f"Found {len(nonsilent_ranges)} speech segments")
        
        # Calculate silence points (gaps between speech)
        silence_points = []
        
        # Add start if there's initial silence
        if nonsilent_ranges[0][0] > 0:
            silence_points.append(nonsilent_ranges[0][0] // 2)
        
        # Add points between speech segments
        for i in range(len(nonsilent_ranges) - 1):
            end_of_current = nonsilent_ranges[i][1]
            start_of_next = nonsilent_ranges[i + 1][0]
            
            if start_of_next - end_of_current >= min_silence_len:
                silence_point = end_of_current + (start_of_next - end_of_current) // 2
                silence_points.append(silence_point)
        
        # Add end if there's final silence
        if nonsilent_ranges[-1][1] < len(audio):
            silence_points.append(nonsilent_ranges[-1][1] + (len(audio) - nonsilent_ranges[-1][1]) // 2)
        
        # Find the best split points near our target durations
        split_points = [0]  # Always start at beginning
        current_pos = 0
        
        while current_pos < len(audio):
            target_next_split = current_pos + target_chunk_duration_ms
            
            if target_next_split >= len(audio):
                break
            
            # Find the silence point closest to our target
            best_split = target_next_split
            min_distance = float('inf')
            
            for silence_point in silence_points:
                if silence_point > current_pos:
                    distance = abs(silence_point - target_next_split)
                    max_deviation = target_chunk_duration_ms * 0.3  # Allow 30% deviation
                    
                    if distance < min_distance and distance <= max_deviation:
                        min_distance = distance
                        best_split = silence_point
            
            split_points.append(best_split)
            current_pos = best_split
            
            deviation_seconds = (best_split - target_next_split) / 1000
            logger.info(f"Split point: {best_split/1000:.1f}s (deviation: {deviation_seconds:+.1f}s)")
        
        # Ensure we end at the actual end of audio
        if split_points[-1] < len(audio):
            split_points.append(len(audio))
        
        return split_points
        
    except ImportError:
        logger.warning("pydub not available, falling back to time-based splitting")
        return None
    except Exception as e:
        logger.error(f"Error in silence detection: {str(e)}")
        return None

def smart_split_audio(audio_path: str, audio_format: str, chunk_duration: int = 300) -> List[Tuple[str, float, float]]:
    """
    Smart split audio at silence points with format-specific optimal chunk sizes
    Returns list of (file_path, start_time, end_time) tuples
    """
    try:
        from pydub import AudioSegment
        
        # Determine optimal chunk duration based on format
        if audio_format.lower() == "mp3":
            optimal_chunk_duration = min(chunk_duration, 300)  # 5 minutes max for MP3
            logger.info("Using MP3 format - optimizing for 5-minute chunks")
        else:
            optimal_chunk_duration = min(chunk_duration, 180)  # 3 minutes max for WAV
            logger.info("Using WAV format - optimizing for 3-minute chunks")
        
        # Load audio
        if audio_format.lower() == "mp3":
            audio = AudioSegment.from_mp3(audio_path)
        elif audio_format.lower() == "wav":
            audio = AudioSegment.from_wav(audio_path)
        else:
            audio = AudioSegment.from_file(audio_path)
        
        total_duration = len(audio) / 1000
        logger.info(f"Audio duration: {total_duration:.1f}s, target chunks: {optimal_chunk_duration}s")
        
        # If audio is shorter than chunk duration, return as single chunk
        if total_duration <= optimal_chunk_duration:
            return [(audio_path, 0.0, total_duration)]
        
        # Get smart split points
        split_points = detect_silence_split_points(audio_path, optimal_chunk_duration, audio_format)
        
        if not split_points:
            # Fallback to time-based splitting
            split_points = list(range(0, len(audio), optimal_chunk_duration * 1000))
            if split_points[-1] < len(audio):
                split_points.append(len(audio))
        
        logger.info(f"Creating {len(split_points)-1} smart chunks")
        
        chunk_files = []
        
        for i in range(len(split_points) - 1):
            start_ms = split_points[i]
            end_ms = split_points[i + 1]
            
            chunk = audio[start_ms:end_ms]
            
            # Create temporary file for chunk
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_format}')
            
            # Export with optimal settings
            if audio_format.lower() == "mp3":
                chunk.export(temp_file.name, format="mp3", parameters=["-ar", "16000", "-ac", "1"])
            else:
                chunk.export(temp_file.name, format="wav", parameters=["-ar", "16000", "-ac", "1"])
            
            temp_file.close()
            
            start_time = start_ms / 1000
            end_time = end_ms / 1000
            
            chunk_files.append((temp_file.name, start_time, end_time))
            
            logger.info(f"Chunk {i+1}: {start_time:.1f}s-{end_time:.1f}s ({end_time-start_time:.1f}s)")
        
        return chunk_files
        
    except ImportError:
        logger.warning("pydub not available, falling back to basic splitting")
        return basic_split_audio(audio_path, chunk_duration)
    except Exception as e:
        logger.error(f"Error in smart splitting: {str(e)}")
        return basic_split_audio(audio_path, chunk_duration)

def basic_split_audio(audio_path: str, chunk_duration: int = 300) -> List[Tuple[str, float, float]]:
    """
    Fallback: Basic time-based audio splitting
    Returns list of (file_path, start_time, end_time) tuples
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        total_duration = waveform.shape[1] / sample_rate
        
        if total_duration <= chunk_duration:
            return [(audio_path, 0.0, total_duration)]
        
        chunk_files = []
        chunk_samples = int(chunk_duration * sample_rate)
        
        for i in range(0, waveform.shape[1], chunk_samples):
            chunk_waveform = waveform[:, i:i + chunk_samples]
            
            # Create temporary file for chunk
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            torchaudio.save(temp_file.name, chunk_waveform, sample_rate)
            temp_file.close()
            
            start_time = i / sample_rate
            end_time = min((i + chunk_samples) / sample_rate, total_duration)
            
            chunk_files.append((temp_file.name, start_time, end_time))
        
        logger.info(f"Split audio into {len(chunk_files)} basic chunks")
        return chunk_files
        
    except Exception as e:
        logger.error(f"Error in basic splitting: {str(e)}")
        return [(audio_path, 0.0, 0.0)]

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

def merge_smart_transcription_results(chunk_results: List[Dict[str, Any]], chunk_times: List[Tuple[float, float]]) -> Dict[str, Any]:
    """Merge transcription results from smart chunks with accurate timing"""
    merged_text = []
    merged_word_timestamps = []
    merged_segment_timestamps = []
    merged_char_timestamps = []
    
    for i, (chunk_result, (start_time, end_time)) in enumerate(zip(chunk_results, chunk_times)):
        if 'error' in chunk_result:
            continue
            
        merged_text.append(chunk_result['text'])
        
        # Adjust timestamps by adding actual chunk start time
        if chunk_result['word_timestamps']:
            for word_ts in chunk_result['word_timestamps']:
                adjusted_word_ts = word_ts.copy()
                adjusted_word_ts['start'] += start_time
                adjusted_word_ts['end'] += start_time
                merged_word_timestamps.append(adjusted_word_ts)
        
        if chunk_result['segment_timestamps']:
            for seg_ts in chunk_result['segment_timestamps']:
                adjusted_seg_ts = seg_ts.copy()
                adjusted_seg_ts['start'] += start_time
                adjusted_seg_ts['end'] += start_time
                merged_segment_timestamps.append(adjusted_seg_ts)
        
        if chunk_result['char_timestamps']:
            for char_ts in chunk_result['char_timestamps']:
                adjusted_char_ts = char_ts.copy()
                adjusted_char_ts['start'] += start_time
                adjusted_char_ts['end'] += start_time
                merged_char_timestamps.append(adjusted_char_ts)
    
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
    RunPod handler function for audio transcription with smart silence-based chunking
    
    Expected input format:
    {
        "input": {
            "audio_data": "base64_encoded_audio_data",  # Base64 encoded audio file
            "audio_format": "wav",  # Optional: audio format (wav, mp3, flac, etc.)
            "include_timestamps": true,  # Optional: include word/segment timestamps
            "chunk_duration": 300  # Optional: target chunk duration in seconds (default 5 minutes)
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
        chunk_duration = job_input.get("chunk_duration", 300)  # 5 minutes default
        
        logger.info(f"Processing transcription request: format={audio_format}, timestamps={include_timestamps}, chunk_duration={chunk_duration}s")
        
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
            # Smart split audio at silence points
            chunk_info = smart_split_audio(temp_audio_file.name, audio_format, chunk_duration)
            
            # Extract file paths and timing info
            chunk_files = [info[0] for info in chunk_info]
            chunk_times = [(info[1], info[2]) for info in chunk_info]
            
            temp_files_to_cleanup.extend(chunk_files)
            
            # Calculate total duration
            total_duration = chunk_times[-1][1] if chunk_times else 0
            
            # Transcribe each chunk
            chunk_results = []
            for i, chunk_path in enumerate(chunk_files):
                start_time, end_time = chunk_times[i]
                logger.info(f"Transcribing chunk {i+1}/{len(chunk_files)} ({start_time:.1f}s-{end_time:.1f}s)")
                chunk_result = transcribe_audio_chunk(chunk_path, include_timestamps)
                chunk_results.append(chunk_result)
            
            # Merge results with smart timing
            if len(chunk_results) == 1:
                final_result = chunk_results[0]
            else:
                final_result = merge_smart_transcription_results(chunk_results, chunk_times)
            
            # Add metadata
            final_result.update({
                'audio_duration_seconds': total_duration,
                'chunks_processed': len(chunk_files),
                'model_used': 'nvidia/parakeet-tdt-0.6b-v2',
                'chunking_method': 'smart_silence_based',
                'chunk_boundaries': [{'start': start, 'end': end} for start, end in chunk_times]
            })
            
            logger.info(f"Transcription completed: {len(final_result.get('word_timestamps', []))} words, {len(chunk_files)} chunks")
            
            return final_result
            
        finally:
            # Clean up temporary files
            cleanup_temp_files(temp_files_to_cleanup)
            
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        return {"error": f"Transcription failed: {str(e)}"}

# Initialize model when the container starts
if __name__ == "__main__":
    logger.info("Initializing NVIDIA Parakeet TDT 0.6B v2 model...")
    if load_model():
        logger.info("Starting RunPod serverless handler with smart silence-based chunking...")
        runpod.serverless.start({"handler": handler})
    else:
        logger.error("Failed to load model. Exiting.")
        exit(1)