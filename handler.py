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
        logger.info("Loading NVIDIA Parakeet TDT 0.6B v3 model...")
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v3"
        )
        logger.info("Model loaded successfully")
        
        # Optimize for long audio processing (up to 3 hours with local attention)
        optimize_for_long_audio()
        
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def optimize_for_long_audio():
    """
    Enable local attention for processing very long audio (up to 3 hours)
    Based on Parakeet v3 documentation: supports up to 24 minutes with full attention,
    up to 3 hours with local attention using rel_pos_local_attn
    """
    global model
    try:
        if model:
            logger.info("Configuring Parakeet v3 for long audio processing...")
            model.change_attention_model(
                self_attention_model="rel_pos_local_attn", 
                att_context_size=[256, 256]
            )
            logger.info("✅ Enabled local attention for long audio (up to 3 hours support)")
        else:
            logger.warning("Model not loaded, cannot configure long audio optimization")
    except Exception as e:
        logger.warning(f"Failed to configure long audio optimization: {str(e)}")
        logger.info("Continuing with default attention model")

def configure_streaming_mode(chunk_size_sec=2.0, left_context_sec=10.0, right_context_sec=2.0):
    """
    Configure model for streaming mode processing
    Based on Parakeet v3 streaming capabilities
    
    Args:
        chunk_size_sec: Size of each processing chunk in seconds (default 2.0)
        left_context_sec: Left context window in seconds (default 10.0) 
        right_context_sec: Right context window in seconds (default 2.0)
    """
    global model
    try:
        if model:
            logger.info(f"Configuring streaming mode: chunk={chunk_size_sec}s, left_ctx={left_context_sec}s, right_ctx={right_context_sec}s")
            
            # Configure for streaming inference
            model.change_attention_model(
                self_attention_model="rel_pos_local_attn",
                att_context_size=[int(left_context_sec * 50), int(right_context_sec * 50)]  # Convert to frames (50fps)
            )
            
            logger.info("✅ Streaming mode configured successfully")
            return {
                'chunk_size_sec': chunk_size_sec,
                'left_context_sec': left_context_sec, 
                'right_context_sec': right_context_sec,
                'mode': 'streaming_optimized'
            }
        else:
            logger.warning("Model not loaded, cannot configure streaming mode")
            return None
    except Exception as e:
        logger.error(f"Failed to configure streaming mode: {str(e)}")
        return None

def load_diarization_model(hf_token=None):
    """
    Load pyannote.audio diarization pipeline
    """
    global diarization_model
    try:
        from pyannote.audio import Pipeline
        import torch
        logger.info("Loading pyannote.audio speaker diarization pipeline...")
        
        # Try to load with HuggingFace token if provided
        if hf_token:
            logger.info("Using provided HuggingFace token for pyannote access")
            diarization_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", 
                use_auth_token=hf_token
            )
        else:
            logger.error("HuggingFace token is required for pyannote.audio models")
            logger.error("Please provide hf_token parameter in your request")
            logger.error("You can get a token at https://hf.co/settings/tokens")
            return False
            
        # Move pipeline to GPU if available
        if torch.cuda.is_available():
            logger.info("Moving pyannote pipeline to GPU")
            diarization_model.to(torch.device("cuda"))
        else:
            logger.warning("CUDA not available, using CPU for diarization")
            
        logger.info("Pyannote diarization pipeline loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading pyannote diarization pipeline: {str(e)}")
        logger.error("Make sure you have:")
        logger.error("1. Accepted pyannote/segmentation-3.0 user conditions")
        logger.error("2. Accepted pyannote/speaker-diarization-3.1 user conditions") 
        logger.error("3. Created a valid HuggingFace access token")
        return False

def perform_speaker_diarization(audio_path: str, num_speakers: int = None) -> List[Dict[str, Any]]:
    """
    Perform speaker diarization on audio file using pyannote.audio
    Returns list of segments with speaker labels and timestamps
    """
    try:
        logger.info(f"Performing pyannote.audio speaker diarization on: {audio_path}")
        
        # Run pyannote diarization
        logger.info("Running pyannote diarization pipeline...")
        diarization = diarization_model(audio_path)
        
        # Convert pyannote output to our format
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker,
                'duration': turn.end - turn.start
            })
            logger.info(f"Speaker segment: {speaker} ({turn.start:.2f}s-{turn.end:.2f}s)")
        
        logger.info(f"Pyannote diarization completed: {len(segments)} segments found")
        if segments:
            speakers_found = set(seg['speaker'] for seg in segments)
            logger.info(f"Speakers detected: {speakers_found}")
        
        return segments
        
    except Exception as e:
        logger.error(f"Error in pyannote speaker diarization: {str(e)}")
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
    """Transcribe a single audio chunk with Parakeet v3"""
    try:
        logger.info(f"🎯 Transcribing chunk: {audio_path} (timestamps={include_timestamps})")
        
        if include_timestamps:
            output = model.transcribe([audio_path], timestamps=True)
        else:
            output = model.transcribe([audio_path])
        
        # 🔍 LOG RAW RESPONSE STRUCTURE FOR DEBUGGING
        logger.info(f"📊 Raw Parakeet v3 output type: {type(output)}")
        logger.info(f"📊 Raw output length: {len(output) if hasattr(output, '__len__') else 'N/A'}")
        
        if len(output) > 0:
            first_result = output[0]
            logger.info(f"📊 First result type: {type(first_result)}")
            logger.info(f"📊 First result keys/attributes: {dir(first_result) if hasattr(first_result, '__dict__') else 'No __dict__'}")
            
            # Log first result as dict if possible
            if hasattr(first_result, '__dict__'):
                logger.info(f"📊 First result __dict__: {first_result.__dict__}")
            elif hasattr(first_result, 'keys'):
                logger.info(f"📊 First result keys: {list(first_result.keys())}")
                logger.info(f"📊 First result content: {dict(first_result)}")
            else:
                logger.info(f"📊 First result direct: {first_result}")
        
        # 🔧 SAFE KEY ACCESS - Try multiple ways to get text and timestamps
        first_result = output[0]
        
        # Try to get text content
        text_content = ""
        try:
            # Method 1: attribute access (.text)
            if hasattr(first_result, 'text'):
                text_content = first_result.text
                logger.info("✅ Got text via .text attribute")
            # Method 2: dictionary access (['text'])
            elif hasattr(first_result, '__getitem__') and 'text' in first_result:
                text_content = first_result['text']
                logger.info("✅ Got text via ['text'] key")
            # Method 3: check other possible text keys
            elif hasattr(first_result, '__getitem__'):
                possible_text_keys = ['transcript', 'transcription', 'result', 'output']
                for key in possible_text_keys:
                    if key in first_result:
                        text_content = first_result[key]
                        logger.info(f"✅ Got text via ['{key}'] key")
                        break
            else:
                logger.warning("❌ Could not find text content in transcription result")
        except Exception as text_error:
            logger.error(f"❌ Error extracting text: {text_error}")
        
        # Try to get timestamps if requested
        word_timestamps = []
        segment_timestamps = []
        char_timestamps = []
        
        if include_timestamps:
            try:
                # Method 1: attribute access (.timestamp)
                if hasattr(first_result, 'timestamp'):
                    timestamp_data = first_result.timestamp
                    logger.info("✅ Got timestamps via .timestamp attribute")
                    if hasattr(timestamp_data, 'get'):
                        word_timestamps = timestamp_data.get('word', [])
                        segment_timestamps = timestamp_data.get('segment', [])
                        char_timestamps = timestamp_data.get('char', [])
                # Method 2: dictionary access
                elif hasattr(first_result, '__getitem__') and 'timestamp' in first_result:
                    timestamp_data = first_result['timestamp']
                    logger.info("✅ Got timestamps via ['timestamp'] key")
                    word_timestamps = timestamp_data.get('word', [])
                    segment_timestamps = timestamp_data.get('segment', [])
                    char_timestamps = timestamp_data.get('char', [])
                # Method 3: direct timestamp keys
                elif hasattr(first_result, '__getitem__'):
                    word_timestamps = first_result.get('word_timestamps', [])
                    segment_timestamps = first_result.get('segment_timestamps', [])
                    char_timestamps = first_result.get('char_timestamps', [])
                    logger.info("✅ Got timestamps via direct keys")
                else:
                    logger.warning("❌ Could not find timestamp data in transcription result")
            except Exception as timestamp_error:
                logger.error(f"❌ Error extracting timestamps: {timestamp_error}")
        
        result = {
            'text': text_content,
            'word_timestamps': word_timestamps,
            'segment_timestamps': segment_timestamps,
            'char_timestamps': char_timestamps
        }
        
        logger.info(f"✅ Transcription successful: {len(text_content)} chars, {len(word_timestamps)} words, {len(segment_timestamps)} segments")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error transcribing audio chunk: {str(e)}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
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
            "num_speakers": null,  # Optional: expected number of speakers (null for auto-detection)
            "hf_token": "hf_xxx",  # Optional: HuggingFace token for pyannote.audio access
            "streaming_mode": false,  # Optional: enable streaming mode for real-time processing
            "streaming_chunk_sec": 2.0,  # Optional: streaming chunk size in seconds (default 2.0)
            "streaming_left_context_sec": 10.0,  # Optional: left context for streaming (default 10.0)
            "streaming_right_context_sec": 2.0  # Optional: right context for streaming (default 2.0)
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
        hf_token = job_input.get("hf_token", None)
        
        # Streaming mode parameters (Parakeet v3 feature)
        streaming_mode = job_input.get("streaming_mode", False)
        streaming_chunk_sec = job_input.get("streaming_chunk_sec", 2.0)
        streaming_left_context_sec = job_input.get("streaming_left_context_sec", 10.0)
        streaming_right_context_sec = job_input.get("streaming_right_context_sec", 2.0)
        
        logger.info(f"Processing transcription request: format={audio_format}, timestamps={include_timestamps}, chunk_duration={chunk_duration}s, diarization={use_diarization}, streaming={streaming_mode}")
        
        # Configure streaming mode if requested
        streaming_config = None
        if streaming_mode:
            logger.info("🚀 Configuring Parakeet v3 streaming mode...")
            streaming_config = configure_streaming_mode(
                chunk_size_sec=streaming_chunk_sec,
                left_context_sec=streaming_left_context_sec,
                right_context_sec=streaming_right_context_sec
            )
            if streaming_config:
                logger.info(f"✅ Streaming mode active: {streaming_config}")
            else:
                logger.warning("⚠️ Failed to configure streaming mode, continuing with standard processing")
        
        # If diarization is requested and HF token is provided, reload the diarization model
        if use_diarization and hf_token and diarization_model is None:
            logger.info("Diarization requested with HF token - loading pyannote model...")
            if not load_diarization_model(hf_token):
                return {"error": "Failed to load diarization model with provided HF token"}
        elif use_diarization and diarization_model is None:
            logger.info("Diarization requested - attempting to load pyannote model without token...")
            if not load_diarization_model():
                return {"error": "Failed to load diarization model. You may need to provide a HuggingFace token (hf_token parameter)"}
        
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
                # NEW WORKFLOW: Diarization first, then transcription, then combine
                logger.info(f"Processing {len(chunk_files)} chunks with NEW diarization workflow...")
                logger.info("Step 1: Diarization on whole chunk, Step 2: Transcription on whole chunk, Step 3: Match timestamps")
                
                diarized_results = []
                
                for i, chunk_path in enumerate(chunk_files):
                    chunk_start_time, chunk_end_time = chunk_times[i]
                    logger.info(f"Processing chunk {i+1}/{len(chunk_files)} ({chunk_start_time:.1f}s-{chunk_end_time:.1f}s)")
                    
                    # STEP 1: Run diarization on the whole chunk
                    logger.info(f"  Step 1: Diarization on chunk {i+1}")
                    chunk_diarized_segments = perform_speaker_diarization(chunk_path, num_speakers)
                    
                    # STEP 2: Run transcription on the whole chunk (SAME AS WORKING VERSION)
                    logger.info(f"  Step 2: Transcription on chunk {i+1}")
                    chunk_result = transcribe_audio_chunk(chunk_path, include_timestamps=True)
                    
                    # STEP 3: Match timestamps to assign speakers
                    logger.info(f"  Step 3: Matching timestamps for chunk {i+1}")
                    
                    if chunk_diarized_segments and chunk_result.get('text'):
                        # We have both diarization and transcription - now match them
                        logger.info(f"    Found {len(chunk_diarized_segments)} speaker segments and transcription text")
                        
                        # Use segment-level timestamps for better matching, fallback to word-level
                        transcript_segments = chunk_result.get('segment_timestamps', [])
                        
                        # If no segment timestamps or empty, try to use word timestamps
                        if not transcript_segments and chunk_result.get('word_timestamps'):
                            logger.info(f"    No segment timestamps, using word-level timestamps for matching")
                            word_timestamps = chunk_result['word_timestamps']
                            
                            # Group words into segments (combine words with small gaps)
                            transcript_segments = []
                            if word_timestamps:
                                current_segment = {
                                    'start': word_timestamps[0]['start'],
                                    'end': word_timestamps[0]['end'],
                                    'text': word_timestamps[0]['word']
                                }
                                
                                for word in word_timestamps[1:]:
                                    # If gap is less than 1 second, combine with current segment
                                    if word['start'] - current_segment['end'] < 1.0:
                                        current_segment['end'] = word['end']
                                        current_segment['text'] += ' ' + word['word']
                                    else:
                                        # Start new segment
                                        transcript_segments.append(current_segment)
                                        current_segment = {
                                            'start': word['start'],
                                            'end': word['end'],
                                            'text': word['word']
                                        }
                                
                                # Don't forget the last segment
                                transcript_segments.append(current_segment)
                        
                        if transcript_segments:
                            logger.info(f"    Matching {len(transcript_segments)} transcript segments to speaker segments")
                            
                            for seg in transcript_segments:
                                # Safe key access with error handling
                                try:
                                    seg_start = seg.get('start', 0)
                                    seg_end = seg.get('end', 0) 
                                    seg_text = seg.get('text', '')
                                except Exception as seg_error:
                                    logger.error(f"    ❌ Error accessing segment keys: {seg_error}, segment: {seg}")
                                    continue
                                
                                # Find which speaker segment this transcript segment overlaps with
                                assigned_speaker = 'UNKNOWN'
                                max_overlap = 0
                                
                                for spk_seg in chunk_diarized_segments:
                                    spk_start = spk_seg['start']
                                    spk_end = spk_seg['end']
                                    
                                    # Calculate overlap
                                    overlap_start = max(seg_start, spk_start)
                                    overlap_end = min(seg_end, spk_end)
                                    overlap = max(0, overlap_end - overlap_start)
                                    
                                    if overlap > max_overlap:
                                        max_overlap = overlap
                                        assigned_speaker = spk_seg['speaker']
                                
                                # Only assign speaker if there's meaningful overlap
                                if max_overlap > 0.1:  # At least 100ms overlap
                                    diarized_results.append({
                                        'speaker': assigned_speaker,
                                        'start_time': seg_start + chunk_start_time,
                                        'end_time': seg_end + chunk_start_time,
                                        'duration': seg_end - seg_start,
                                        'text': seg_text,
                                        'word_timestamps': [],  # Could be added later
                                        'segment_timestamps': [seg],
                                        'source_chunk': i + 1,
                                        'overlap_duration': max_overlap
                                    })
                                else:
                                    # No good overlap, mark as unknown
                                    diarized_results.append({
                                        'speaker': 'UNKNOWN',
                                        'start_time': seg_start + chunk_start_time,
                                        'end_time': seg_end + chunk_start_time,
                                        'duration': seg_end - seg_start,
                                        'text': seg_text,
                                        'word_timestamps': [],
                                        'segment_timestamps': [seg],
                                        'source_chunk': i + 1,
                                        'overlap_duration': 0
                                    })
                            
                            logger.info(f"    ✅ Matched {len(transcript_segments)} segments for chunk {i+1}")
                        
                        else:
                            logger.warning(f"    ❌ No usable timestamps found (segment: {len(chunk_result.get('segment_timestamps', []))}, word: {len(chunk_result.get('word_timestamps', []))})")
                            # Fallback: use the whole chunk text with first speaker
                            first_speaker = chunk_diarized_segments[0]['speaker'] if chunk_diarized_segments else 'UNKNOWN'
                            diarized_results.append({
                                'speaker': first_speaker,
                                'start_time': chunk_start_time,
                                'end_time': chunk_end_time,
                                'duration': chunk_end_time - chunk_start_time,
                                'text': chunk_result.get('text', ''),
                                'word_timestamps': chunk_result.get('word_timestamps', []),
                                'segment_timestamps': chunk_result.get('segment_timestamps', []),
                                'source_chunk': i + 1,
                                'fallback_reason': 'no_usable_timestamps'
                            })
                            logger.info(f"    ⚠️  Using whole chunk with speaker {first_speaker} (no timestamps)")
                    
                    else:
                        # Fallback: transcribe chunk normally if diarization or transcription failed
                        logger.warning(f"No speakers detected or transcription failed in chunk {i+1}, transcribing normally")
                        
                        if not chunk_result.get('text'):
                            # Retry transcription without timestamps
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
                    'model_used': 'nvidia/parakeet-tdt-0.6b-v3',
                    'diarization_model': 'pyannote/speaker-diarization-3.1',
                    'processing_method': 'diarization_then_transcription_then_match',
                    'chunking_method': 'smart_silence_based',
                    'streaming_config': streaming_config,
                    'long_audio_optimization': 'local_attention_enabled',
                    'chunk_boundaries': [{'start': start, 'end': end} for start, end in chunk_times]
                }
                
                # Also provide merged text for convenience
                merged_text = ' '.join([result['text'] for result in diarized_results if result['text']])
                final_result['merged_text'] = merged_text
                
                logger.info(f"NEW diarization workflow completed: {len(diarized_results)} segments, {final_result['speakers_detected']} speakers across {len(chunk_files)} chunks")
                
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
                    'model_used': 'nvidia/parakeet-tdt-0.6b-v3',
                    'chunking_method': 'smart_silence_based',
                    'processing_method': 'regular_transcription',
                    'streaming_config': streaming_config,
                    'long_audio_optimization': 'local_attention_enabled',
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
    logger.info("Initializing NVIDIA Parakeet TDT 0.6B v3 model...")
    if load_model():
        logger.info("Parakeet model loaded successfully")
        
        # Diarization model will be loaded on-demand when needed (with HF token if provided)
        logger.info("Pyannote diarization model will be loaded on-demand when diarization is requested")
        
        logger.info("Starting RunPod serverless handler with enhanced Parakeet v3 capabilities...")
        logger.info("🚀 FEATURES: Smart chunking, pyannote diarization, long audio support (3+ hours), streaming mode")
        logger.info("🎯 MODEL: NVIDIA Parakeet TDT 0.6B v3 with 25 language support and local attention optimization")
        runpod.serverless.start({"handler": handler})
    else:
        logger.error("Failed to load Parakeet model. Exiting.")
        exit(1)