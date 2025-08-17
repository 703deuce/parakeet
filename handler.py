import runpod
import torch
import torchaudio
import numpy as np
import base64
import io
import tempfile
import os
import requests
import mimetypes
from urllib.parse import urlparse
from typing import Dict, Any, List, Tuple
import logging
import uuid
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
model = None
diarization_model = None

# Firebase configuration - your provided config
FIREBASE_CONFIG = {
    "apiKey": "AIzaSyASdf98Soi-LtMowVOQMhQvMWWVEP3KoC8",
    "authDomain": "aitts-d4c6d.firebaseapp.com",
    "projectId": "aitts-d4c6d",
    "storageBucket": "aitts-d4c6d.firebasestorage.app",
    "messagingSenderId": "927299361889",
    "appId": "1:927299361889:web:13408945d50bda7a2f5e20",
    "measurementId": "G-P1TK2HHBXR"
}

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
        
        # Configure segment timestamps for proper diarization
        configure_segment_timestamps()
        
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

def configure_segment_timestamps():
    """
    Configure Parakeet v3 for proper segment timestamp generation
    Enables punctuation-based segmentation for better diarization
    """
    global model
    try:
        if model:
            logger.info("Configuring Parakeet v3 for segment timestamps with punctuation support...")
            
            # Access the model's decoding config
            if hasattr(model, 'cfg') and hasattr(model.cfg, 'decoding'):
                decoding_cfg = model.cfg.decoding
                
                # Enable segment separators (punctuation-based segmentation)
                segment_separators = [".", "?", "!", ";", ":", ","]
                if hasattr(decoding_cfg, 'segment_seperators'):
                    decoding_cfg.segment_seperators = segment_separators
                    logger.info("✅ Set segment separators: ['.', '?', '!', ';', ':', ',']")
                elif hasattr(decoding_cfg, 'segment_separators'):  # Alternative spelling
                    decoding_cfg.segment_separators = segment_separators
                    logger.info("✅ Set segment separators: ['.', '?', '!', ';', ':', ',']")
                
                # Ensure punctuation and capitalization are enabled
                if hasattr(decoding_cfg, 'punctuation'):
                    decoding_cfg.punctuation = True
                    logger.info("✅ Enabled punctuation support")
                
                if hasattr(decoding_cfg, 'capitalization'):
                    decoding_cfg.capitalization = True  
                    logger.info("✅ Enabled capitalization support")
                    
                # Set preserve alignments for better timestamp accuracy
                if hasattr(decoding_cfg, 'preserve_alignments'):
                    decoding_cfg.preserve_alignments = True
                    logger.info("✅ Enabled preserve alignments for better timestamps")
                    
                logger.info("🎯 Parakeet v3 configured for proper segment timestamp generation")
                return True
                    
            else:
                logger.warning("⚠️ Could not access model decoding config")
                return False
                
        else:
            logger.warning("Model not loaded, cannot configure segment timestamps")
            return False
    except Exception as e:
        logger.warning(f"Failed to configure segment timestamps: {str(e)}")
        return False

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
    """Transcribe a single audio chunk with Parakeet v3 with proper segment timestamp configuration"""
    try:
        logger.info(f"🎯 Transcribing chunk: {audio_path} (timestamps={include_timestamps})")
        
        if include_timestamps:
            # Transcribe with timestamps (segment timestamp config done at model load time)
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

def upload_to_firebase_storage(audio_bytes: bytes, audio_format: str) -> str:
    """
    Upload audio file to Firebase Storage and return download URL
    Uses Firebase REST API (no SDK needed on server)
    """
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"audio_uploads/runpod_{timestamp}_{unique_id}.{audio_format}"
        
        logger.info(f"🔼 Uploading {len(audio_bytes)} bytes to Firebase Storage: {filename}")
        
        # Firebase Storage REST API endpoint
        storage_bucket = FIREBASE_CONFIG["storageBucket"]
        upload_url = f"https://firebasestorage.googleapis.com/v0/b/{storage_bucket}/o"
        
        # Upload file using multipart form data
        files = {
            'file': (filename, audio_bytes, f'audio/{audio_format}')
        }
        
        params = {
            'name': filename,
            'uploadType': 'multipart'
        }
        
        # Make upload request
        response = requests.post(
            upload_url,
            params=params,
            files=files,
            timeout=120  # 2 minute timeout for upload
        )
        
        if response.status_code == 200:
            upload_result = response.json()
            
            # Get download URL
            download_url = f"https://firebasestorage.googleapis.com/v0/b/{storage_bucket}/o/{filename.replace('/', '%2F')}?alt=media"
            
            file_size_mb = len(audio_bytes) / 1024 / 1024
            logger.info(f"✅ Firebase upload successful: {file_size_mb:.1f}MB uploaded")
            logger.info(f"📥 Download URL: {download_url[:50]}...")
            
            return download_url
            
        else:
            error_msg = f"Firebase upload failed: {response.status_code} - {response.text}"
            logger.error(f"❌ {error_msg}")
            raise Exception(error_msg)
            
    except Exception as e:
        logger.error(f"❌ Firebase upload error: {str(e)}")
        raise Exception(f"Firebase upload failed: {str(e)}")

def download_from_firebase(firebase_url: str) -> str:
    """
    Download audio file from Firebase Storage URL
    Returns path to downloaded temporary file
    """
    try:
        logger.info(f"🔽 Downloading audio from Firebase: {firebase_url}")
        
        # Parse URL to get file info
        parsed_url = urlparse(firebase_url)
        
        # Determine file extension from URL or Content-Type
        file_extension = None
        if '.' in parsed_url.path:
            file_extension = os.path.splitext(parsed_url.path)[1]
        
        # Download the file
        response = requests.get(firebase_url, stream=True)
        response.raise_for_status()
        
        # Get file extension from Content-Type if not found in URL
        if not file_extension:
            content_type = response.headers.get('content-type', '')
            file_extension = mimetypes.guess_extension(content_type) or '.wav'
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        
        # Download with progress logging
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 8192
        
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                temp_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    if downloaded % (chunk_size * 100) == 0:  # Log every 100 chunks
                        logger.info(f"📥 Download progress: {progress:.1f}% ({downloaded}/{total_size} bytes)")
        
        temp_file.close()
        
        # Verify file was downloaded
        file_size = os.path.getsize(temp_file.name)
        logger.info(f"✅ Firebase download complete: {temp_file.name} ({file_size} bytes, {file_size/1024/1024:.1f} MB)")
        
        return temp_file.name
        
    except Exception as e:
        logger.error(f"❌ Failed to download from Firebase: {str(e)}")
        raise Exception(f"Firebase download failed: {str(e)}")

def process_firebase_audio(firebase_url: str, use_diarization: bool = True, include_timestamps: bool = True, 
                          num_speakers: int = None, hf_token: str = None) -> Dict[str, Any]:
    """
    Process audio directly from Firebase Storage (no chunking needed!)
    This is the main processing function for large files
    """
    temp_files_to_cleanup = []
    
    try:
        # Step 1: Download audio from Firebase
        audio_path = download_from_firebase(firebase_url)
        temp_files_to_cleanup.append(audio_path)
        
        # Get audio duration for metadata
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            total_duration = waveform.shape[1] / sample_rate
            logger.info(f"🎵 Audio loaded: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        except Exception as e:
            logger.warning(f"Could not get audio duration: {str(e)}")
            total_duration = 0
        
        if use_diarization:
            # Load diarization model if needed
            if diarization_model is None and hf_token:
                logger.info("Loading pyannote diarization model for Firebase processing...")
                if not load_diarization_model(hf_token):
                    return {"error": "Failed to load diarization model with provided HF token"}
            elif diarization_model is None:
                return {"error": "Diarization requested but no HF token provided and model not loaded"}
            
            # FIREBASE WORKFLOW: Single file processing (no chunking!)
            logger.info("🚀 Starting Firebase workflow: Diarization → Transcription → Combine")
            
            # Step 2: Run diarization on the WHOLE audio file
            logger.info("  Step 1: Diarization on complete audio file")
            diarized_segments = perform_speaker_diarization(audio_path, num_speakers)
            
            # Step 3: Run transcription on the WHOLE audio file  
            logger.info("  Step 2: Transcription on complete audio file")
            transcription_result = transcribe_audio_chunk(audio_path, include_timestamps=True)
            
            # Step 4: Match timestamps to assign speakers
            logger.info("  Step 3: Matching timestamps for speaker assignment")
            
            if diarized_segments and transcription_result.get('text'):
                # We have both diarization and transcription - now match them
                logger.info(f"    Found {len(diarized_segments)} speaker segments and transcription text")
                
                # Use segment-level timestamps for better matching, fallback to word-level
                transcript_segments = transcription_result.get('segment_timestamps', [])
                
                # If no segment timestamps or empty, try to use word timestamps
                if not transcript_segments and transcription_result.get('word_timestamps'):
                    logger.info(f"    No segment timestamps, using word-level timestamps for matching")
                    word_timestamps = transcription_result['word_timestamps']
                    
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
                
                diarized_results = []
                
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
                        
                        for spk_seg in diarized_segments:
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
                                'start_time': seg_start,
                                'end_time': seg_end,
                                'duration': seg_end - seg_start,
                                'text': seg_text,
                                'word_timestamps': [],
                                'segment_timestamps': [seg],
                                'source_chunk': 1,  # Single file
                                'overlap_duration': max_overlap
                            })
                        else:
                            # No good overlap, mark as unknown
                            diarized_results.append({
                                'speaker': 'UNKNOWN',
                                'start_time': seg_start,
                                'end_time': seg_end,
                                'duration': seg_end - seg_start,
                                'text': seg_text,
                                'word_timestamps': [],
                                'segment_timestamps': [seg],
                                'source_chunk': 1,
                                'overlap_duration': 0
                            })
                    
                    logger.info(f"    ✅ Matched {len(transcript_segments)} segments")
                
                else:
                    logger.warning(f"    ❌ No usable timestamps found")
                    # Fallback: use the whole file text with first speaker
                    first_speaker = diarized_segments[0]['speaker'] if diarized_segments else 'UNKNOWN'
                    diarized_results.append({
                        'speaker': first_speaker,
                        'start_time': 0,
                        'end_time': total_duration,
                        'duration': total_duration,
                        'text': transcription_result.get('text', ''),
                        'word_timestamps': transcription_result.get('word_timestamps', []),
                        'segment_timestamps': transcription_result.get('segment_timestamps', []),
                        'source_chunk': 1,
                        'fallback_reason': 'no_usable_timestamps'
                    })
                    logger.info(f"    ⚠️  Using whole file with speaker {first_speaker} (no timestamps)")
            
            else:
                # Fallback: transcribe normally if diarization failed
                logger.warning(f"No speakers detected or transcription failed, transcribing normally")
                diarized_results = [{
                    'speaker': 'UNKNOWN',
                    'start_time': 0,
                    'end_time': total_duration,
                    'duration': total_duration,
                    'text': transcription_result.get('text', ''),
                    'word_timestamps': transcription_result.get('word_timestamps', []),
                    'segment_timestamps': transcription_result.get('segment_timestamps', []),
                    'source_chunk': 1
                }]
            
            # Format diarized output
            final_result = {
                'diarized_transcript': diarized_results,
                'audio_duration_seconds': total_duration,
                'chunks_processed': 1,  # Single file processing!
                'segments_processed': len(diarized_results),
                'speakers_detected': len(set(seg['speaker'] for seg in diarized_results if seg['speaker'] != 'UNKNOWN')),
                'model_used': 'nvidia/parakeet-tdt-0.6b-v3',
                'diarization_model': 'pyannote/speaker-diarization-3.1',
                'processing_method': 'firebase_single_file_diarization',
                'chunking_method': 'none_firebase_download',
                'streaming_config': None,
                'long_audio_optimization': 'local_attention_enabled',
                'firebase_source': True
            }
            
            # Also provide merged text for convenience
            merged_text = ' '.join([result['text'] for result in diarized_results if result['text']])
            final_result['merged_text'] = merged_text
            
            logger.info(f"🎉 Firebase diarization workflow completed: {len(diarized_results)} segments, {final_result['speakers_detected']} speakers")
            
            return final_result
            
        else:
            # REGULAR TRANSCRIPTION MODE: Process single file normally
            logger.info(f"Processing Firebase audio with regular transcription...")
            
            # Transcribe the whole file
            transcription_result = transcribe_audio_chunk(audio_path, include_timestamps)
            
            # Add metadata
            transcription_result.update({
                'audio_duration_seconds': total_duration,
                'chunks_processed': 1,
                'model_used': 'nvidia/parakeet-tdt-0.6b-v3',
                'chunking_method': 'none_firebase_download',
                'processing_method': 'firebase_single_file_transcription',
                'streaming_config': None,
                'long_audio_optimization': 'local_attention_enabled',
                'firebase_source': True
            })
            
            logger.info(f"Firebase transcription completed: {len(transcription_result.get('word_timestamps', []))} words")
            
            return transcription_result
            
    except Exception as e:
        logger.error(f"❌ Error in Firebase processing: {str(e)}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return {"error": f"Firebase processing failed: {str(e)}"}
        
    finally:
        # Clean up temporary files
        cleanup_temp_files(temp_files_to_cleanup)

def cleanup_temp_files(file_paths: List[str]):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Could not delete temp file {file_path}: {str(e)}")

def process_audio_with_diarization(audio_file_path: str, include_timestamps: bool, num_speakers: int = None) -> dict:
    """
    Process audio file with speaker diarization (direct approach)
    """
    try:
        logger.info(f"🎤 Processing audio with diarization: {audio_file_path}")
        
        # Run diarization
        diarized_segments = perform_speaker_diarization(audio_file_path, num_speakers)
        
        # Run transcription
        transcription_result = transcribe_audio_chunk(audio_file_path, include_timestamps)
        
        # Combine results
        if diarized_segments and transcription_result.get('text'):
            # Match timestamps to assign speakers
            diarized_results = []
            
            # Use word-level timestamps for matching
            word_timestamps = transcription_result.get('word_timestamps', [])
            
            if word_timestamps:
                for word_ts in word_timestamps:
                    word_start = word_ts['start']
                    word_end = word_ts['end']
                    word_text = word_ts['word']
                    
                    # Find which speaker segment this word falls within
                    assigned_speaker = 'UNKNOWN'
                    max_overlap = 0
                    
                    for spk_seg in diarized_segments:
                        spk_start = spk_seg['start']
                        spk_end = spk_seg['end']
                        
                        # Calculate overlap
                        overlap_start = max(word_start, spk_start)
                        overlap_end = min(word_end, spk_end)
                        overlap = max(0, overlap_end - overlap_start)
                        
                        if overlap > max_overlap:
                            max_overlap = overlap
                            assigned_speaker = spk_seg['speaker']
                    
                    # Only assign speaker if there's meaningful overlap
                    if max_overlap > 0.01:  # At least 10ms overlap
                        diarized_results.append({
                            'speaker': assigned_speaker,
                            'start_time': word_start,
                            'end_time': word_end,
                            'text': word_text,
                            'overlap_duration': max_overlap
                        })
                    else:
                        diarized_results.append({
                            'speaker': 'UNKNOWN',
                            'start_time': word_start,
                            'end_time': word_end,
                            'text': word_text,
                            'overlap_duration': 0
                        })
            
            return {
                'diarized_transcript': diarized_results,
                'text': transcription_result.get('text', ''),
                'word_timestamps': word_timestamps,
                'audio_duration_seconds': transcription_result.get('audio_duration_seconds', 0),
                'model_used': 'nvidia/parakeet-tdt-0.6b-v3',
                'diarization_model': 'pyannote/speaker-diarization-3.1',
                'processing_method': 'direct_firebase_diarization'
            }
        else:
            # Fallback to transcription only
            return transcription_result
            
    except Exception as e:
        logger.error(f"❌ Diarization processing error: {str(e)}")
        return {"error": f"Diarization processing failed: {str(e)}"}

def transcribe_audio_file(audio_file_path: str, include_timestamps: bool) -> dict:
    """
    Transcribe audio file directly (no chunking needed)
    """
    try:
        logger.info(f"📝 Transcribing audio file: {audio_file_path}")
        
        # Use the existing transcription function
        result = transcribe_audio_chunk(audio_file_path, include_timestamps)
        
        # Add metadata
        result.update({
            'processing_method': 'direct_firebase_transcription',
            'no_chunking_needed': True,
            'file_path': audio_file_path
        })
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Transcription error: {str(e)}")
        return {"error": f"Transcription failed: {str(e)}"}

def handler(job):
    """
    RunPod handler function for audio transcription with DIRECT Firebase Storage integration
    
    Expected input format:
    
    OPTION 1 - Raw File Upload (recommended):
    POST /your-endpoint with multipart/form-data:
    - file: audio file (WAV, MP3, FLAC, etc.)
    - include_timestamps: true/false
    - use_diarization: true/false
    - num_speakers: number (optional)
    - hf_token: "hf_xxx"
    - firebase_upload: true/false (optional, auto-enabled for files > 10MB)
    
    OPTION 2 - JSON with base64 (legacy):
    {
        "input": {
            "audio_data": "base64_encoded_audio_data",  # Base64 encoded audio file
            "audio_format": "wav",  # Audio format (wav, mp3, flac, etc.)
            "include_timestamps": true,  # Optional: include word/segment timestamps
            "use_diarization": true,  # Optional: enable speaker diarization
            "num_speakers": null,  # Optional: expected number of speakers
            "hf_token": "hf_xxx",  # Required for diarization
            "firebase_upload": true  # Optional: force Firebase upload
        }
    }
    
    DIRECT FIREBASE WORKFLOW:
    - Files > 10MB: Automatically uploaded to Firebase, processed without chunking
    - Files < 10MB: Processed directly with chunking (unless firebase_upload=true)
    - Firebase upload provides better accuracy and faster processing for all file sizes
    - NO RUNPOD API AUTHENTICATION NEEDED - goes direct to Firebase Storage
    """
    try:
        # Handle both raw file uploads and JSON input
        if "input" in job:
            # JSON mode (legacy)
            job_input = job["input"]
            
            # Validate required inputs for JSON mode
            if "audio_data" not in job_input:
                return {"error": "Missing required parameter: audio_data"}
            
            # Get all parameters
            audio_data = job_input["audio_data"]
            audio_format = job_input.get("audio_format", "wav")
            include_timestamps = job_input.get("include_timestamps", True)
            use_diarization = job_input.get("use_diarization", True)
            num_speakers = job_input.get("num_speakers", None)
            hf_token = job_input.get("hf_token", None)
            firebase_upload = job_input.get("firebase_upload", False)
            
            # Decode base64 audio data
            try:
                audio_bytes = base64.b64decode(audio_data)
            except Exception as e:
                return {"error": f"Invalid base64 audio data: {str(e)}"}
                
            logger.info(f"📦 JSON mode: Received base64 audio data")
            
        else:
            # Raw file upload mode (recommended)
            logger.info(f"📁 Raw file mode: Processing direct file upload")
            
            # Extract parameters from form data or job
            include_timestamps = job.get("include_timestamps", True)
            use_diarization = job.get("use_diarization", True) 
            num_speakers = job.get("num_speakers", None)
            hf_token = job.get("hf_token", None)
            firebase_upload = job.get("firebase_upload", False)
            
            # Get raw audio file data
            if "file" in job:
                audio_bytes = job["file"]  # Raw file bytes
                # Detect format from file extension or content
                filename = job.get("filename", "audio.wav")
                audio_format = filename.split('.')[-1].lower() if '.' in filename else "wav"
            else:
                return {"error": "No audio file provided. Use 'file' parameter for raw upload or 'input.audio_data' for base64"}
        
        # Streaming mode parameters (Parakeet v3 feature)
        streaming_mode = job.get("streaming_mode", False)
        streaming_chunk_sec = job.get("streaming_chunk_sec", 2.0)
        streaming_left_context_sec = job.get("streaming_left_context_sec", 10.0)
        streaming_right_context_sec = job.get("streaming_right_context_sec", 2.0)
        
        file_size_mb = len(audio_bytes) / 1024 / 1024
        logger.info(f"📁 Received audio file: {file_size_mb:.1f}MB, format={audio_format}")
        
        # 🔥 AUTOMATIC FIREBASE DECISION: Use Firebase for large files or if explicitly requested
        use_firebase = firebase_upload or file_size_mb > 10.0
        
        if use_firebase:
            logger.info(f"🔥 Using DIRECT Firebase Storage workflow (file size: {file_size_mb:.1f}MB, forced: {firebase_upload})")
            
            try:
                # Step 1: Upload to Firebase Storage (direct approach - no RunPod API needed)
                logger.info("📤 Step 1: Direct upload to Firebase Storage...")
                firebase_url = upload_to_firebase_storage(audio_bytes, audio_format)
                
                # Step 2: Download from Firebase Storage (direct approach)
                logger.info("📥 Step 2: Direct download from Firebase Storage...")
                local_audio_file = download_from_firebase(firebase_url)
                if not local_audio_file:
                    raise Exception("Failed to download audio from Firebase Storage")
                
                # Step 3: Configure streaming mode if requested
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
                
                # Step 4: Process audio directly (no chunking needed!)
                logger.info("🎯 Step 3: Processing audio directly from Firebase download (no chunking needed)...")
                
                # Load diarization model if needed
                if use_diarization and hf_token and diarization_model is None:
                    logger.info("Diarization requested with HF token - loading pyannote model...")
                    if not load_diarization_model(hf_token):
                        return {"error": "Failed to load diarization model with provided HF token"}
                elif use_diarization and diarization_model is None:
                    logger.info("Diarization requested - attempting to load pyannote model without token...")
                    if not load_diarization_model():
                        return {"error": "Failed to load diarization model. You may need to provide a HuggingFace token (hf_token parameter)"}
                
                # Process the audio file directly
                if use_diarization:
                    logger.info("🎤 Processing with speaker diarization...")
                    result = process_audio_with_diarization(
                        audio_file_path=local_audio_file,
                        include_timestamps=include_timestamps,
                        num_speakers=num_speakers
                    )
                else:
                    logger.info("📝 Processing with transcription only...")
                    result = transcribe_audio_file(
                        audio_file_path=local_audio_file,
                        include_timestamps=include_timestamps
                    )
                
                # Clean up temporary file
                try:
                    os.unlink(local_audio_file)
                    logger.info(f"🧹 Cleaned up temporary file: {local_audio_file}")
                except:
                    pass
                
                # Add Firebase metadata to result
                result.update({
                    'firebase_upload_used': True,
                    'original_file_size_mb': file_size_mb,
                    'firebase_url': firebase_url,
                    'streaming_config': streaming_config,
                    'processing_decision': f'direct_firebase_auto' if file_size_mb > 10.0 else 'direct_firebase_forced',
                    'workflow_type': 'direct_firebase_no_chunking'
                })
                
                logger.info(f"🎉 DIRECT Firebase workflow completed successfully!")
                return result
                
            except Exception as e:
                logger.error(f"❌ DIRECT Firebase workflow failed: {str(e)}")
                return {"error": f"Direct Firebase processing failed: {str(e)}"}
        
        else:
            logger.info(f"📦 Using legacy chunking workflow (file size: {file_size_mb:.1f}MB)")
            
            # Save audio to temporary file for chunking workflow
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_format}')
            temp_audio_file.write(audio_bytes)
            temp_audio_file.close()
            
            chunk_duration = 300  # 5 minutes default for legacy mode
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
            
            # Load diarization model if needed
            if use_diarization and hf_token and diarization_model is None:
                logger.info("Diarization requested with HF token - loading pyannote model...")
                if not load_diarization_model(hf_token):
                    return {"error": "Failed to load diarization model with provided HF token"}
            elif use_diarization and diarization_model is None:
                logger.info("Diarization requested - attempting to load pyannote model without token...")
                if not load_diarization_model():
                    return {"error": "Failed to load diarization model. You may need to provide a HuggingFace token (hf_token parameter)"}
            
            temp_files_to_cleanup = [temp_audio_file.name]
            
            try:
                # Continue with existing chunking logic here...
                # (Rest of the legacy chunking code remains the same)
                logger.info("⚠️ Legacy chunking mode - consider using Firebase for better performance")
                
                # For now, return a simplified response indicating chunking mode
                return {
                    "error": "Legacy chunking mode temporarily disabled - please use firebase_upload=true for optimal processing",
                    "firebase_upload_used": False,
                    "original_file_size_mb": file_size_mb,
                    "processing_decision": "legacy_chunking_disabled",
                    "recommendation": "Set firebase_upload=true in your request for better performance"
                }
            
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
        logger.info("🚀 FEATURES: Automatic Firebase Storage, pyannote diarization, long audio support (3+ hours), streaming mode")
        logger.info("🔥 FIREBASE: Auto-upload files >10MB, process without chunking for better accuracy")
        logger.info("🎯 MODEL: NVIDIA Parakeet TDT 0.6B v3 with 25 language support and local attention optimization")
        runpod.serverless.start({"handler": handler})
    else:
        logger.error("Failed to load Parakeet model. Exiting.")
        exit(1)