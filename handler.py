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

# Global model variables
model = None
diarization_model = None

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

def load_diarization_model():
    """
    Load NeMo diarization models on-demand
    Note: We use ClusteringDiarizer which loads models automatically
    """
    logger.info("Diarization models will be loaded on-demand by ClusteringDiarizer")
    return True

def perform_speaker_diarization(audio_path: str, num_speakers: int = None) -> List[Dict[str, Any]]:
    """
    Perform speaker diarization on audio file using NeMo ClusteringDiarizer
    Returns list of segments with speaker labels and timestamps
    """
    try:
        logger.info(f"Performing speaker diarization on audio: {audio_path}")
        
        # Create a simple manifest for diarization (must be a list!)
        manifest_data = [{
            "audio_filepath": audio_path,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": num_speakers,
            "rttm_filepath": None,
            "uem_filepath": None
        }]
        
        # Create temporary manifest file
        manifest_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        import json
        for item in manifest_data:
            json.dump(item, manifest_file)
            manifest_file.write('\n')
        manifest_file.close()
        
        # Configure diarization with proper NeMo pipeline - FORCE SPEAKER CLUSTERING
        from omegaconf import OmegaConf
        
        # Force speaker clustering by setting both min and max speakers
        min_speakers = int(num_speakers) if num_speakers else 1
        max_speakers = int(num_speakers) if num_speakers else 8
        
        diar_cfg = OmegaConf.create({
            'diarizer': {
                'manifest_filepath': manifest_file.name,
                'out_dir': tempfile.gettempdir(),
                'oracle_num_speakers': num_speakers is not None,  # Use oracle if num_speakers provided
                'min_num_speakers': min_speakers,  # CRITICAL: Force minimum speakers
                'max_num_speakers': max_speakers,  # CRITICAL: Force maximum speakers
                'oracle_vad': False,  # CRITICAL: Add missing oracle_vad key
                'speaker_embeddings': {
                    'model_path': 'nvidia/speakerverification_en_titanet_large',
                    'parameters': {
                        'window_length_in_sec': 0.63,
                        'shift_length_in_sec': 0.01,
                        'multiscale_weights': [1, 1, 1, 1, 1],
                        'save_embeddings': False
                    }
                },
                'clustering': {
                    'parameters': {
                        'oracle_num_speakers': num_speakers is not None,
                        'min_num_speakers': min_speakers,  # CRITICAL: Force clustering to find min speakers
                        'max_num_speakers': max_speakers,  # CRITICAL: Force clustering to find max speakers
                        'enhanced_count_thres': 40,
                        'max_rp_threshold': 0.25,
                        'sparse_search_volume': 30
                    }
                },
                'vad': {
                    'model_path': 'nvidia/vad_multilingual_marblenet',
                    'parameters': {
                        'window_length_in_sec': 0.63,
                        'shift_length_in_sec': 0.01,
                        'smoothing': "median",
                        'overlap': 0.875,
                        'onset': 0.8,
                        'offset': 0.6,
                        'pad_onset': 0.05,
                        'pad_offset': -0.1,
                        'min_duration_on': 0.2,
                        'min_duration_off': 0.2
                    }
                }
            }
        })
        
        logger.info(f"DIARIZATION CONFIG: min_speakers={min_speakers}, max_speakers={max_speakers}, oracle={num_speakers is not None}")
        
        # Run diarization with ClusteringDiarizer
        from nemo.collections.asr.models import ClusteringDiarizer
        logger.info(f"Initializing ClusteringDiarizer with config: {diar_cfg}")
        diarizer = ClusteringDiarizer(cfg=diar_cfg)
        
        logger.info("Starting diarization process...")
        diarizer.diarize()
        
        # Parse results from RTTM file
        # NeMo might create RTTM with different naming conventions
        possible_rttm_files = [
            os.path.join(tempfile.gettempdir(), f"{os.path.basename(audio_path)}.rttm"),
            os.path.join(tempfile.gettempdir(), f"{os.path.basename(audio_path).split('.')[0]}.rttm"),
            os.path.join(tempfile.gettempdir(), "pred_rttms", f"{os.path.basename(audio_path)}.rttm"),
            os.path.join(tempfile.gettempdir(), "pred_rttms", f"{os.path.basename(audio_path).split('.')[0]}.rttm")
        ]
        
        segments = []
        rttm_file = None
        
        # Find the actual RTTM file
        for rttm_path in possible_rttm_files:
            if os.path.exists(rttm_path):
                rttm_file = rttm_path
                logger.info(f"Found RTTM file: {rttm_file}")
                break
        
        if rttm_file:
            # DEBUG: Log the entire RTTM file before parsing
            logger.info("=== RTTM FILE CONTENTS START ===")
            with open(rttm_file, 'r') as f:
                rttm_content = f.read()
                logger.info(rttm_content)
            logger.info("=== RTTM FILE CONTENTS END ===")
            
            # Now parse the RTTM file
            with open(rttm_file, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    parts = line.split()
                    if len(parts) >= 8:
                        start_time = float(parts[3])
                        duration = float(parts[4])
                        end_time = start_time + duration
                        speaker_id = parts[7]
                        
                        logger.info(f"RTTM line {line_num}: speaker={speaker_id}, start={start_time:.2f}s, end={end_time:.2f}s")
                        
                        segments.append({
                            'start': start_time,
                            'end': end_time,
                            'speaker': speaker_id,
                            'duration': duration
                        })
        else:
            logger.warning(f"RTTM file not found. Checked paths: {possible_rttm_files}")
            # Check what files are in the output directory
            out_dir = tempfile.gettempdir()
            logger.info(f"Files in output directory {out_dir}:")
            try:
                for file in os.listdir(out_dir):
                    if 'rttm' in file.lower() or os.path.basename(audio_path) in file:
                        logger.info(f"  Found: {file}")
                
                # Also check for pred_rttms subdirectory
                pred_rttm_dir = os.path.join(out_dir, "pred_rttms")
                if os.path.exists(pred_rttm_dir):
                    logger.info(f"Files in pred_rttms directory:")
                    for file in os.listdir(pred_rttm_dir):
                        logger.info(f"  Found: {file}")
            except Exception as e:
                logger.error(f"Error listing output directory: {e}")
        
        # Cleanup temporary files
        try:
            os.unlink(manifest_file.name)
            if os.path.exists(rttm_file):
                os.unlink(rttm_file)
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {e}")
        
        logger.info(f"Diarization completed: {len(segments)} segments found")
        if segments:
            speakers_found = set(seg['speaker'] for seg in segments)
            logger.info(f"Speakers detected: {speakers_found}")
        
        return segments
        
    except Exception as e:
        logger.error(f"Error in speaker diarization: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []

def extract_audio_segment(audio_path: str, start_time: float, end_time: float) -> str:
    """Extract audio segment from start to end time"""
    try:
        from pydub import AudioSegment
        
        # Load audio
        if audio_path.lower().endswith('.mp3'):
            audio = AudioSegment.from_mp3(audio_path)
        else:
            audio = AudioSegment.from_wav(audio_path)
        
        # Extract segment (convert seconds to milliseconds)
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        segment = audio[start_ms:end_ms]
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        segment.export(temp_file.name, format="wav", parameters=["-ar", "16000", "-ac", "1"])
        temp_file.close()
        
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Error extracting audio segment: {str(e)}")
        return None

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
    RunPod handler function for audio transcription with smart silence-based chunking and optional speaker diarization
    
    Expected input format:
    {
        "input": {
            "audio_data": "base64_encoded_audio_data",  # Base64 encoded audio file
            "audio_format": "wav",  # Optional: audio format (wav, mp3, flac, etc.)
            "include_timestamps": true,  # Optional: include word/segment timestamps
            "chunk_duration": 300,  # Optional: target chunk duration in seconds (default 5 minutes)
            "use_diarization": false,  # Optional: enable speaker diarization (default false)
            "num_speakers": null  # Optional: expected number of speakers (null for auto-detection)
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
        use_diarization = job_input.get("use_diarization", False)
        num_speakers = job_input.get("num_speakers", None)
        
        logger.info(f"Processing transcription request: format={audio_format}, timestamps={include_timestamps}, chunk_duration={chunk_duration}s, diarization={use_diarization}")
        
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
            # Use the existing smart silence-based chunking for ALL modes
            logger.info("Starting smart silence-based chunking...")
            
            # Determine optimal chunk duration based on format and RunPod 10MiB limit
            if audio_format == 'mp3':
                optimal_chunk_duration = 300  # 5 minutes for MP3
                logger.info("Using 5-minute chunks for MP3 for optimal efficiency.")
            elif audio_format == 'wav':
                optimal_chunk_duration = 180  # 3 minutes for WAV  
                logger.info("Using 3-minute chunks for WAV for optimal efficiency.")
            else:
                optimal_chunk_duration = chunk_duration
                logger.info(f"Using default chunk duration of {optimal_chunk_duration}s for {audio_format}.")
            
            # Smart split audio at silence points (same for both modes)
            chunk_info = smart_split_audio(temp_audio_file.name, audio_format, optimal_chunk_duration)
            
            # Extract file paths and timing info
            chunk_files = [info[0] for info in chunk_info]
            chunk_times = [(info[1], info[2]) for info in chunk_info]
            
            temp_files_to_cleanup.extend(chunk_files)
            
            # Calculate total duration
            total_duration = chunk_times[-1][1] if chunk_times else 0
            
            if use_diarization:
                # DIARIZATION MODE: Process each chunk with diarization + transcription
                logger.info(f"Processing {len(chunk_files)} chunks with diarization...")
                
                diarized_results = []
                
                for i, chunk_path in enumerate(chunk_files):
                    chunk_start_time, chunk_end_time = chunk_times[i]
                    logger.info(f"Processing chunk {i+1}/{len(chunk_files)} with diarization ({chunk_start_time:.1f}s-{chunk_end_time:.1f}s)")
                    
                    # Perform diarization on this chunk
                    chunk_diarized_segments = perform_speaker_diarization(chunk_path, num_speakers)
                    
                    if chunk_diarized_segments:
                        # Process each speaker segment within this chunk
                        for segment in chunk_diarized_segments:
                            segment_start = segment['start'] + chunk_start_time  # Adjust to absolute time
                            segment_end = segment['end'] + chunk_start_time
                            speaker_id = segment['speaker']
                            
                            # Extract the speaker segment from the chunk
                            segment_file = extract_audio_segment(chunk_path, segment['start'], segment['end'])
                            if segment_file:
                                temp_files_to_cleanup.append(segment_file)
                                
                                # Transcribe this speaker segment
                                segment_result = transcribe_audio_chunk(segment_file, include_timestamps)
                                
                                # Adjust timestamps to absolute time
                                if segment_result.get('word_timestamps'):
                                    for word_ts in segment_result['word_timestamps']:
                                        word_ts['start'] += segment_start
                                        word_ts['end'] += segment_start
                                
                                if segment_result.get('segment_timestamps'):
                                    for seg_ts in segment_result['segment_timestamps']:
                                        seg_ts['start'] += segment_start
                                        seg_ts['end'] += segment_start
                                
                                # Add to results
                                diarized_results.append({
                                    'speaker': speaker_id,
                                    'start_time': segment_start,
                                    'end_time': segment_end,
                                    'duration': segment_end - segment_start,
                                    'text': segment_result.get('text', ''),
                                    'word_timestamps': segment_result.get('word_timestamps', []),
                                    'segment_timestamps': segment_result.get('segment_timestamps', []),
                                    'source_chunk': i + 1
                                })
                    else:
                        # Fallback: transcribe chunk normally if no diarization found
                        logger.warning(f"No speakers detected in chunk {i+1}, transcribing normally")
                        chunk_result = transcribe_audio_chunk(chunk_path, include_timestamps)
                        
                        # Adjust timestamps to absolute time
                        if chunk_result.get('word_timestamps'):
                            for word_ts in chunk_result['word_timestamps']:
                                word_ts['start'] += chunk_start_time
                                word_ts['end'] += chunk_start_time
                        
                        if chunk_result.get('segment_timestamps'):
                            for seg_ts in chunk_result['segment_timestamps']:
                                seg_ts['start'] += chunk_start_time
                                seg_ts['end'] += chunk_start_time
                        
                        diarized_results.append({
                            'speaker': 'UNKNOWN',
                            'start_time': chunk_start_time,
                            'end_time': chunk_end_time,
                            'duration': chunk_end_time - chunk_start_time,
                            'text': chunk_result.get('text', ''),
                            'word_timestamps': chunk_result.get('word_timestamps', []),
                            'segment_timestamps': chunk_result.get('segment_timestamps', []),
                            'source_chunk': i + 1
                        })
                
                # Format diarized output
                final_result = {
                    'diarized_transcript': diarized_results,
                    'audio_duration_seconds': total_duration,
                    'chunks_processed': len(chunk_files),
                    'segments_processed': len(diarized_results),
                    'speakers_detected': len(set(seg['speaker'] for seg in diarized_results if seg['speaker'] != 'UNKNOWN')),
                    'model_used': 'nvidia/parakeet-tdt-0.6b-v2',
                    'diarization_model': 'nvidia/speakerverification_en_titanet_large',
                    'processing_method': 'chunk_based_diarization_with_transcription',
                    'chunking_method': 'smart_silence_based',
                    'chunk_boundaries': [{'start': start, 'end': end} for start, end in chunk_times]
                }
                
                # Also provide merged text for convenience
                merged_text = ' '.join([result['text'] for result in diarized_results if result['text']])
                final_result['merged_text'] = merged_text
                
                logger.info(f"Chunk-based diarization completed: {len(diarized_results)} segments, {final_result['speakers_detected']} speakers across {len(chunk_files)} chunks")
                
                return final_result
                
            else:
                # REGULAR TRANSCRIPTION MODE: Process chunks normally
                logger.info(f"Processing {len(chunk_files)} chunks with regular transcription...")
                
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
                    'processing_method': 'regular_transcription',
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
        logger.info("Parakeet model loaded successfully")
        
        # Optionally pre-load diarization model (comment out to load on-demand)
        # logger.info("Pre-loading NeMo Speaker Diarization model...")
        # load_diarization_model()
        
        logger.info("Starting RunPod serverless handler with smart chunking and optional diarization...")
        logger.info("FIXED: Now using proper NeMo ClusteringDiarizer pipeline for real speaker labels (spk0, spk1, etc.)")
        logger.info("CRITICAL FIX: Added min_num_speakers and max_num_speakers to force speaker clustering!")
        logger.info("FIXED: Added missing oracle_vad key to NeMo diarization config!")
        runpod.serverless.start({"handler": handler})
    else:
        logger.error("Failed to load Parakeet model. Exiting.")
        exit(1)