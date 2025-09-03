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
import gc
from urllib.parse import urlparse
from typing import Dict, Any, List, Tuple
import logging
import uuid
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
model = None
diarization_model = None

def clear_gpu_memory():
    """Clear GPU memory and run garbage collection"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("🧹 GPU memory cache cleared")
        
        # Run Python garbage collection
        gc.collect()
        logger.info("🧹 Python garbage collection completed")
        
        # Log memory usage if available
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            logger.info(f"💾 GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            
    except Exception as e:
        logger.warning(f"⚠️ Memory cleanup error: {str(e)}")

def ensure_cuda_available():
    """Check and log CUDA availability"""
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            logger.info(f"🚀 CUDA available: {device_count} device(s), using: {device_name}")
        else:
            logger.warning("⚠️ CUDA not available, using CPU")
        return cuda_available
    except Exception as e:
        logger.error(f"❌ CUDA check error: {str(e)}")
        return False

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
    """Load the NVIDIA Parakeet model with caching"""
    global model
    try:
        # Clear memory before loading
        clear_gpu_memory()
        
        # Check CUDA availability
        ensure_cuda_available()
        
        # Set up cache directory for persistent storage
        cache_dir = "/runpod-volume/cache"
        parakeet_cache_dir = os.path.join(cache_dir, "parakeet-tdt-0.6b-v3")
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(parakeet_cache_dir, exist_ok=True)
        
        import nemo.collections.asr as nemo_asr
        
        # Check if model is already cached
        cached_model_path = os.path.join(parakeet_cache_dir, "parakeet-tdt-0.6b-v3.nemo")
        
        if os.path.exists(cached_model_path):
            logger.info(f"📦 Loading cached Parakeet model from: {cached_model_path}")
            try:
                model = nemo_asr.models.ASRModel.restore_from(cached_model_path)
                logger.info("✅ Cached model loaded successfully")
            except Exception as cache_error:
                logger.warning(f"⚠️ Failed to load cached model: {cache_error}")
                logger.info("🔄 Downloading fresh model...")
                model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name="nvidia/parakeet-tdt-0.6b-v3"
                )
                # Save the model to cache for next time
                try:
                    model.save_to(cached_model_path)
                    logger.info(f"💾 Model cached to: {cached_model_path}")
                except Exception as save_error:
                    logger.warning(f"⚠️ Failed to cache model: {save_error}")
        else:
            logger.info("🔄 Downloading NVIDIA Parakeet TDT 0.6B v3 model (first time)...")
            model = nemo_asr.models.ASRModel.from_pretrained(
                model_name="nvidia/parakeet-tdt-0.6b-v3"
            )
            # Save the model to cache for next time
            try:
                model.save_to(cached_model_path)
                logger.info(f"💾 Model cached to: {cached_model_path}")
            except Exception as save_error:
                logger.warning(f"⚠️ Failed to cache model: {save_error}")
        
        logger.info("Model loaded successfully")
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("🚀 Model moved to GPU")
        
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
            
            # Debug: Log available model attributes
            logger.info(f"🔍 Model type: {type(model)}")
            logger.info(f"🔍 Model has cfg: {hasattr(model, 'cfg')}")
            if hasattr(model, 'cfg'):
                logger.info(f"🔍 Model cfg type: {type(model.cfg)}")
                logger.info(f"🔍 Model cfg has decoding: {hasattr(model.cfg, 'decoding')}")
            
            # Access the model's decoding config
            if hasattr(model, 'cfg') and hasattr(model.cfg, 'decoding'):
                decoding_cfg = model.cfg.decoding
                logger.info(f"🔍 Decoding config type: {type(decoding_cfg)}")
                logger.info(f"🔍 Available decoding config attributes: {[attr for attr in dir(decoding_cfg) if not attr.startswith('_')]}")
                
                # Enable segment separators (punctuation-based segmentation)
                segment_separators = [".", "?", "!", ";", ":", ","]
                separator_set = False
                
                # Try different possible attribute names for segment separators
                for attr_name in ['segment_seperators', 'segment_separators', 'segment_seps', 'separators']:
                    if hasattr(decoding_cfg, attr_name):
                        setattr(decoding_cfg, attr_name, segment_separators)
                        logger.info(f"✅ Set segment separators via '{attr_name}': {segment_separators}")
                        separator_set = True
                        break
                
                if not separator_set:
                    logger.warning("⚠️ Could not find segment separator attribute in decoding config")
                
                # Ensure punctuation and capitalization are enabled
                punct_enabled = False
                for attr_name in ['punctuation', 'punct', 'add_punctuation']:
                    if hasattr(decoding_cfg, attr_name):
                        setattr(decoding_cfg, attr_name, True)
                        logger.info(f"✅ Enabled punctuation via '{attr_name}'")
                        punct_enabled = True
                        break
                
                if not punct_enabled:
                    logger.warning("⚠️ Could not find punctuation attribute in decoding config")
                
                cap_enabled = False
                for attr_name in ['capitalization', 'caps', 'add_capitalization']:
                    if hasattr(decoding_cfg, attr_name):
                        setattr(decoding_cfg, attr_name, True)  
                        logger.info(f"✅ Enabled capitalization via '{attr_name}'")
                        cap_enabled = True
                        break
                
                if not cap_enabled:
                    logger.warning("⚠️ Could not find capitalization attribute in decoding config")
                    
                # Set preserve alignments for better timestamp accuracy
                align_enabled = False
                for attr_name in ['preserve_alignments', 'preserve_alignment', 'alignments']:
                    if hasattr(decoding_cfg, attr_name):
                        setattr(decoding_cfg, attr_name, True)
                        logger.info(f"✅ Enabled preserve alignments via '{attr_name}'")
                        align_enabled = True
                        break
                
                if not align_enabled:
                    logger.warning("⚠️ Could not find preserve alignments attribute in decoding config")
                
                # Additional config for timestamps  
                for attr_name in ['timestamps', 'return_timestamps', 'word_timestamps']:
                    if hasattr(decoding_cfg, attr_name):
                        setattr(decoding_cfg, attr_name, True)
                        logger.info(f"✅ Enabled timestamps via '{attr_name}'")
                        break
                
                logger.info("🎯 Parakeet v3 segment timestamp configuration completed")
                return separator_set or punct_enabled or cap_enabled
                    
            else:
                logger.warning("⚠️ Could not access model decoding config")
                # Try alternative configuration methods
                if hasattr(model, 'decoder') and hasattr(model.decoder, 'cfg'):
                    logger.info("🔍 Trying decoder.cfg instead...")
                    decoding_cfg = model.decoder.cfg
                    # Apply same configuration logic here if needed
                
                return False
                
        else:
            logger.warning("Model not loaded, cannot configure segment timestamps")
            return False
    except Exception as e:
        logger.error(f"Failed to configure segment timestamps: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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
    Load pyannote.audio diarization pipeline with caching
    """
    global diarization_model
    try:
        # Clear memory before loading diarization model
        clear_gpu_memory()
        
        # Set up cache directory for persistent storage
        cache_dir = "/runpod-volume/cache"
        pyannote_cache_dir = os.path.join(cache_dir, "pyannote-speaker-diarization-3.1")
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(pyannote_cache_dir, exist_ok=True)
        
        from pyannote.audio import Pipeline
        import torch
        
        # Check if model is already cached
        cached_config_path = os.path.join(pyannote_cache_dir, "config.yaml")
        
        if os.path.exists(cached_config_path):
            logger.info(f"📦 Loading cached pyannote model directly from disk: {pyannote_cache_dir}")
            try:
                # Load directly from local cache directory - no internet or token needed!
                diarization_model = Pipeline.from_pretrained(pyannote_cache_dir)
                logger.info("✅ Cached pyannote model loaded successfully from local files (no token needed)")
            except Exception as cache_error:
                logger.warning(f"⚠️ Failed to load cached pyannote model: {cache_error}")
                if not hf_token:
                    logger.error("HuggingFace token is required for pyannote.audio models")
                    logger.error("Please provide hf_token parameter in your request")
                    logger.error("You can get a token at https://hf.co/settings/tokens")
                    return False
                logger.info("🔄 Downloading fresh pyannote model...")
                diarization_model = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", 
                    use_auth_token=hf_token,
                    cache_dir=pyannote_cache_dir
                )
                logger.info(f"💾 Pyannote model downloaded and cached to: {pyannote_cache_dir}")
        else:
            logger.info("🔄 Downloading pyannote speaker diarization model (first time)...")
            # Try to load with HuggingFace token if provided
            if hf_token:
                logger.info("Using provided HuggingFace token for pyannote access")
                # Set environment variables for caching
                os.environ['PYANNOTE_CACHE'] = pyannote_cache_dir
                os.environ['HF_HOME'] = pyannote_cache_dir
                
                diarization_model = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", 
                    use_auth_token=hf_token,
                    cache_dir=pyannote_cache_dir
                )
                logger.info(f"💾 Pyannote model downloaded and cached to: {pyannote_cache_dir}")
            else:
                logger.error("HuggingFace token is required for pyannote.audio models")
                logger.error("Please provide hf_token parameter in your request")
                logger.error("You can get a token at https://hf.co/settings/tokens")
                return False
            
        # Move pipeline to GPU if available
        if torch.cuda.is_available():
            logger.info("🚀 Moving pyannote pipeline to GPU")
            diarization_model.to(torch.device("cuda"))
        else:
            logger.warning("⚠️ CUDA not available, using CPU for diarization")
        
        # Clear memory after loading
        clear_gpu_memory()
            
        logger.info("Pyannote diarization pipeline loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading pyannote diarization pipeline: {str(e)}")
        logger.error("Make sure you have:")
        logger.error("1. Accepted pyannote/segmentation-3.0 user conditions")
        logger.error("2. Accepted pyannote/speaker-diarization-3.1 user conditions") 
        logger.error("3. Created a valid HuggingFace access token")
        return False

def analyze_audio_quality(audio_path: str) -> Dict[str, Any]:
    """
    Analyze audio quality and characteristics to help with diarization
    """
    try:
        import librosa
        import numpy as np
        
        # Load audio for analysis
        y, sr = librosa.load(audio_path, sr=16000)
        duration = len(y) / sr
        
        # Calculate audio statistics
        rms_energy = np.sqrt(np.mean(y**2))
        max_amplitude = np.max(np.abs(y))
        silence_threshold = 0.01
        non_silent_samples = np.sum(np.abs(y) > silence_threshold)
        speech_ratio = non_silent_samples / len(y)
        
        analysis = {
            'duration': duration,
            'sample_rate': sr,
            'rms_energy': float(rms_energy),
            'max_amplitude': float(max_amplitude),
            'speech_ratio': float(speech_ratio),
            'likely_has_speech': rms_energy > 0.001 and speech_ratio > 0.1,
            'is_too_quiet': max_amplitude < 0.01,
            'is_too_short': duration < 2.0
        }
        
        logger.info(f"🔍 Audio Analysis: duration={duration:.1f}s, energy={rms_energy:.4f}, speech_ratio={speech_ratio:.2f}")
        return analysis
        
    except Exception as e:
        logger.warning(f"Could not analyze audio quality: {e}")
        return {'duration': 0, 'likely_has_speech': True, 'is_too_quiet': False, 'is_too_short': False}

def perform_speaker_diarization(audio_path: str, num_speakers: int = None) -> List[Dict[str, Any]]:
    """
    Perform speaker diarization on audio file using pyannote.audio
    Returns list of segments with speaker labels and timestamps
    """
    try:
        logger.info(f"Performing pyannote.audio speaker diarization on: {audio_path}")
        
        # DEBUG: Verify file details
        if os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
            logger.info(f"🔍 DIARIZATION DEBUG - File: {audio_path}")
            logger.info(f"🔍 DIARIZATION DEBUG - Size: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
            
            # Check actual audio duration with multiple methods
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(audio_path)
                duration_pydub = len(audio) / 1000.0
                logger.info(f"🔍 DIARIZATION DEBUG - Duration (pydub): {duration_pydub:.2f}s")
                
                import torchaudio
                waveform, sample_rate = torchaudio.load(audio_path)
                duration_torch = waveform.shape[1] / sample_rate
                logger.info(f"🔍 DIARIZATION DEBUG - Duration (torchaudio): {duration_torch:.2f}s")
                
                if duration_pydub < 10:
                    logger.error(f"❌ DIARIZATION ERROR - Audio too short: {duration_pydub}s - this might be the wrong file!")
                    
            except Exception as e:
                logger.error(f"🔍 DIARIZATION DEBUG - Duration check failed: {e}")
        else:
            logger.error(f"❌ DIARIZATION ERROR - File does not exist: {audio_path}")
            return []
        
        # First analyze audio quality
        audio_analysis = analyze_audio_quality(audio_path)
        
        # Check if audio is suitable for diarization
        if not audio_analysis.get('likely_has_speech', True):
            logger.warning("⚠️ Audio doesn't appear to contain speech - skipping diarization")
            return []
            
        if audio_analysis.get('is_too_quiet', False):
            logger.warning("⚠️ Audio appears to be too quiet - may affect diarization quality")
            
        if audio_analysis.get('is_too_short', False):
            logger.warning("⚠️ Audio is very short (<2s) - diarization may be unreliable")
        
        # Adjust diarization parameters based on audio characteristics
        pipeline_params = {}
        
        # For short audio, use more lenient thresholds
        if audio_analysis.get('duration', 0) < 10:
            logger.info("🔧 Using relaxed thresholds for short audio")
            pipeline_params = {
                "segmentation": {
                    "min_duration_off": 0.1,  # Reduced from default 0.5826
                    "threshold": 0.4,         # Reduced from default 0.4697
                },
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": 1,    # Allow single-segment clusters
                    "threshold": 0.6,         # Reduced from default 0.7153
                }
            }
        
        # Ensure audio is mono for diarization
        mono_audio_path = ensure_mono_audio(audio_path)
        temp_files_to_cleanup = []
        
        if mono_audio_path != audio_path:
            temp_files_to_cleanup.append(mono_audio_path)
            logger.info(f"🔄 Using converted mono audio for diarization: {mono_audio_path}")
        else:
            logger.info("✅ Using original mono audio for diarization")
        
        try:
            # Run pyannote diarization with adjusted parameters
            logger.info("Running pyannote diarization pipeline...")
            if pipeline_params:
                # Apply the custom parameters
                logger.info(f"🔧 Applying custom diarization parameters: {pipeline_params}")
                diarization = diarization_model(mono_audio_path, **pipeline_params)
            else:
                diarization = diarization_model(mono_audio_path)
        finally:
            # Clean up temporary mono file if created
            for temp_file in temp_files_to_cleanup:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                        logger.info(f"🧹 Cleaned up temporary mono file: {temp_file}")
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Could not clean up temporary file {temp_file}: {cleanup_error}")
        
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
        else:
            logger.warning("⚠️ No speaker segments detected - trying fallback strategies...")
            
            # FALLBACK 1: Try with much more relaxed parameters
            try:
                logger.info("🔄 Fallback 1: Trying with very relaxed clustering thresholds...")
                fallback_params = {
                    "segmentation": {
                        "min_duration_off": 0.05,  # Very short pauses
                        "threshold": 0.3,          # Very low threshold
                    },
                    "clustering": {
                        "method": "centroid",
                        "min_cluster_size": 1,
                        "threshold": 0.4,          # Much lower clustering threshold
                    }
                }
                
                diarization_fallback = diarization_model(audio_path, **fallback_params)
                segments = []
                for turn, _, speaker in diarization_fallback.itertracks(yield_label=True):
                    segments.append({
                        'start': turn.start,
                        'end': turn.end,
                        'speaker': speaker,
                        'duration': turn.end - turn.start
                    })
                
                if segments:
                    logger.info(f"✅ Fallback 1 successful: {len(segments)} segments found")
                else:
                    logger.warning("❌ Fallback 1 failed")
                    
            except Exception as e:
                logger.warning(f"Fallback 1 error: {str(e)}")
            
            # FALLBACK 2: Create a single speaker segment if still no results
            if not segments:
                logger.info("🔄 Fallback 2: Creating single speaker segment for entire audio...")
                try:
                    # Get audio duration
                    import librosa
                    y, sr = librosa.load(audio_path, sr=None)
                    duration = len(y) / sr
                    
                    segments = [{
                        'start': 0.0,
                        'end': duration,
                        'speaker': 'SPEAKER_00',
                        'duration': duration
                    }]
                    logger.info(f"✅ Fallback 2: Created single speaker segment (0.0s - {duration:.1f}s)")
                    
                except Exception as e:
                    logger.error(f"Fallback 2 error: {str(e)}")
                    segments = []
        
        final_count = len(segments)
        logger.info(f"🎯 Final diarization result: {final_count} segments")
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
        # Ensure audio is mono for processing
        mono_audio_path = ensure_mono_audio(audio_path)
        temp_files_to_cleanup = []
        
        if mono_audio_path != audio_path:
            temp_files_to_cleanup.append(mono_audio_path)
        
        waveform, sample_rate = torchaudio.load(mono_audio_path)
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
    finally:
        # Clean up temporary mono file if created
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.info(f"🧹 Cleaned up temporary mono file: {temp_file}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Could not clean up temporary file {temp_file}: {cleanup_error}")

def transcribe_audio_file_direct(audio_path: str, include_timestamps: bool = False) -> Dict[str, Any]:
    """Transcribe entire audio file directly with Parakeet v3 (NO CHUNKING - processes whole file at once)"""
    try:
        logger.info(f"🎯 Transcribing ENTIRE FILE directly: {audio_path} (timestamps={include_timestamps}) - NO CHUNKING!")
        
        # Log detailed audio file info for debugging
        if os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
            logger.info(f"🎵 Audio file exists: {audio_path} ({file_size} bytes)")
            
            # Check audio with pydub for debugging
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(audio_path)
                duration_ms = len(audio)
                channels = audio.channels
                frame_rate = audio.frame_rate
                logger.info(f"🎵 Audio info: {duration_ms}ms, {channels}ch, {frame_rate}Hz")
                
                # Check if audio is mostly silent
                if duration_ms > 0:
                    samples = audio.get_array_of_samples()
                    if len(samples) > 0:
                        max_amplitude = max(abs(sample) for sample in samples)
                        logger.info(f"🔊 Max amplitude: {max_amplitude}")
                        if max_amplitude < 100:  # Very quiet threshold
                            logger.warning("⚠️ Audio appears to be very quiet or silent")
                    else:
                        logger.warning("⚠️ No audio samples found")
                else:
                    logger.warning("⚠️ Audio duration is 0ms")
                
                # Log channel information for debugging
                if channels == 1:
                    logger.info("✅ Audio is mono - compatible with Parakeet model")
                elif channels == 2:
                    logger.info("⚠️ Audio is stereo - will be converted to mono for Parakeet model")
                else:
                    logger.warning(f"⚠️ Audio has {channels} channels - may cause issues with Parakeet model")
                    
            except Exception as audio_debug_error:
                logger.warning(f"⚠️ Audio debug failed: {str(audio_debug_error)}")
        else:
            logger.error(f"❌ Audio file does not exist: {audio_path}")
            return {"error": f"Audio file not found: {audio_path}"}
        
        # Check if model is loaded
        if model is None:
            logger.error("❌ Parakeet model is not loaded!")
            return {"error": "Model not loaded"}
        
        # Ensure audio is mono before transcription
        mono_audio_path = ensure_mono_audio(audio_path)
        temp_files_to_cleanup = []
        
        if mono_audio_path != audio_path:
            temp_files_to_cleanup.append(mono_audio_path)
            logger.info(f"🔄 Using converted mono audio: {mono_audio_path}")
        else:
            logger.info("✅ Using original mono audio")
        
        logger.info("🚀 Starting Parakeet transcription...")
        
        try:
            if include_timestamps:
                # Transcribe with timestamps (segment timestamp config done at model load time)
                output = model.transcribe([mono_audio_path], timestamps=True)
            else:
                output = model.transcribe([mono_audio_path])
        finally:
            # Clean up temporary mono file if created
            for temp_file in temp_files_to_cleanup:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                        logger.info(f"🧹 Cleaned up temporary mono file: {temp_file}")
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Could not clean up temporary file {temp_file}: {cleanup_error}")
        
        logger.info(f"🔍 Raw Parakeet output type: {type(output)}")
        logger.info(f"🔍 Raw Parakeet output length: {len(output) if hasattr(output, '__len__') else 'N/A'}")
        
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
        
        # DEBUG: Add call stack info to track which flow called this
        import traceback
        stack_info = traceback.extract_stack()
        caller_line = stack_info[-2].lineno if len(stack_info) >= 2 else "unknown"
        logger.info(f"📍 Download called from line: {caller_line}")
        
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
        
        # DEBUG: Check audio duration immediately after download
        try:
            import librosa
            y, sr = librosa.load(temp_file.name, sr=None)
            duration = len(y) / sr
            logger.info(f"📊 Downloaded audio duration: {duration:.1f}s ({duration/60:.1f} minutes)")
        except Exception as e:
            logger.warning(f"⚠️ Could not analyze downloaded audio duration: {str(e)}")
        
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
            transcription_result = transcribe_audio_file_direct(audio_path, include_timestamps=True)
            
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
            transcription_result = transcribe_audio_file_direct(audio_path, include_timestamps)
            
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
        transcription_result = transcribe_audio_file_direct(audio_file_path, include_timestamps)
        
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
        
        # Use the direct transcription function (NO CHUNKING!)
        result = transcribe_audio_file_direct(audio_file_path, include_timestamps)
        
        # Add metadata
        result.update({
            'processing_method': 'direct_file_transcription_no_chunking',
            'whole_file_processed': True,
            'no_chunking_used': True,
            'file_path': audio_file_path,
            'advantage': 'Entire file processed at once for maximum accuracy'
        })
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Transcription error: {str(e)}")
        return {"error": f"Transcription failed: {str(e)}"}

def process_downloaded_audio(audio_file_path: str, include_timestamps: bool, use_diarization: bool, 
                           num_speakers: int = None, hf_token: str = None, audio_format: str = "wav") -> dict:
    """
    Process downloaded audio file with transcription and optional diarization
    This is the main processing function for Firebase URL workflow
    """
    try:
        logger.info(f"🎯 Processing downloaded audio file: {audio_file_path}")
        
        # Get audio duration for metadata
        try:
            waveform, sample_rate = torchaudio.load(audio_file_path)
            total_duration = waveform.shape[1] / sample_rate
            logger.info(f"🎵 Audio loaded: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        except Exception as e:
            logger.warning(f"Could not get audio duration: {str(e)}")
            total_duration = 0
        
        # Check if audio is long enough to benefit from chunking (30+ minutes)
        use_chunking = total_duration > 1800  # 30 minutes
        if use_chunking:
            logger.info(f"🔪 Long audio detected ({total_duration/60:.1f} minutes) - using chunked processing")
            return process_long_audio_with_chunking(
                audio_file_path, include_timestamps, use_diarization, 
                num_speakers, hf_token, audio_format, total_duration
            )
        
        if use_diarization:
            # Load diarization model if needed
            if diarization_model is None and hf_token:
                logger.info("🎤 Loading pyannote diarization model...")
                if not load_diarization_model(hf_token):
                    return {"error": "Failed to load diarization model with provided HF token"}
            elif diarization_model is None:
                return {"error": "Diarization requested but no HF token provided"}
            
            # Run diarization on the complete audio file
            logger.info("🎤 Running speaker diarization on complete audio file...")
            diarized_segments = perform_speaker_diarization(audio_file_path, num_speakers)
            
            # Run transcription on the complete audio file
            logger.info("📝 Running transcription on complete audio file...")
            transcription_result = transcribe_audio_file_direct(audio_file_path, include_timestamps=True)
            
            # Match timestamps to assign speakers
            logger.info("🔗 Matching timestamps for speaker assignment...")
            
            if diarized_segments and transcription_result.get('text'):
                # Use segment-level timestamps for matching (much better than word-level)
                segment_timestamps = transcription_result.get('segment_timestamps', [])
                
                diarized_results = []
                
                if segment_timestamps:
                    logger.info(f"📊 Using {len(segment_timestamps)} segment timestamps for speaker assignment")
                    
                    # Get the full transcribed text
                    full_text = transcription_result.get('text', '')
                    word_timestamps = transcription_result.get('word_timestamps', [])
                    
                    for segment_ts in segment_timestamps:
                        segment_start = segment_ts['start']
                        segment_end = segment_ts['end']
                        
                        # Extract text for this segment by finding words within the time range
                        segment_text = ""
                        if word_timestamps:
                            segment_words = []
                            for word_ts in word_timestamps:
                                word_start = word_ts['start']
                                word_end = word_ts['end']
                                # If word overlaps with segment timeframe
                                if (word_start >= segment_start and word_start <= segment_end) or \
                                   (word_end >= segment_start and word_end <= segment_end) or \
                                   (word_start <= segment_start and word_end >= segment_end):
                                    segment_words.append(word_ts['word'])
                            segment_text = ' '.join(segment_words)
                        
                        # Fallback: if no words found, use a placeholder
                        if not segment_text.strip():
                            segment_text = f"[Segment {segment_start:.1f}-{segment_end:.1f}s]"
                        
                        # Find which speaker segment this transcription segment overlaps with most
                        assigned_speaker = 'UNKNOWN'
                        max_overlap = 0
                        
                        for spk_seg in diarized_segments:
                            spk_start = spk_seg['start']
                            spk_end = spk_seg['end']
                            
                            # Calculate overlap between transcription segment and speaker segment
                            overlap_start = max(segment_start, spk_start)
                            overlap_end = min(segment_end, spk_end)
                            overlap = max(0, overlap_end - overlap_start)
                            
                            if overlap > max_overlap:
                                max_overlap = overlap
                                assigned_speaker = spk_seg['speaker']
                        
                        # Only assign speaker if there's meaningful overlap
                        if max_overlap > 0.01:  # At least 10ms overlap
                            diarized_results.append({
                                'speaker': assigned_speaker,
                                'start_time': segment_start,
                                'end_time': segment_end,
                                'text': segment_text,
                                'overlap_duration': max_overlap
                            })
                        else:
                            diarized_results.append({
                                'speaker': 'UNKNOWN',
                                'start_time': segment_start,
                                'end_time': segment_end,
                                'text': segment_text,
                                'overlap_duration': 0
                            })
                            
                    logger.info(f"✅ Assigned speakers to {len(diarized_results)} segments")
                
                else:
                    logger.error("❌ No segment timestamps available - Parakeet v3 should always produce segment timestamps")
                    # Create a single segment with the full transcript as fallback
                    first_speaker = diarized_segments[0]['speaker'] if diarized_segments else 'UNKNOWN'
                    diarized_results.append({
                        'speaker': first_speaker,
                        'start_time': 0,
                        'end_time': total_duration,
                        'text': transcription_result.get('text', ''),
                        'overlap_duration': total_duration,
                        'fallback_reason': 'no_segment_timestamps'
                    })
                
                # Format diarized output
                final_result = {
                    'diarized_transcript': diarized_results,
                    'audio_duration_seconds': total_duration,
                    'chunks_processed': 1,  # Single file processing
                    'segments_processed': len(diarized_results),
                    'speakers_detected': len(set(seg['speaker'] for seg in diarized_results if seg['speaker'] != 'UNKNOWN')),
                    'model_used': 'nvidia/parakeet-tdt-0.6b-v3',
                    'diarization_model': 'pyannote/speaker-diarization-3.1',
                    'processing_method': 'firebase_url_diarization',
                    'chunking_method': 'none_direct_download',
                    'long_audio_optimization': 'local_attention_enabled'
                }
                
                # Also provide merged text for convenience
                merged_text = ' '.join([result['text'] for result in diarized_results if result['text']])
                final_result['merged_text'] = merged_text
                
                logger.info(f"🎉 Diarization completed: {len(diarized_results)} segments, {final_result['speakers_detected']} speakers")
                return final_result
                
            else:
                # Fallback: transcribe normally if diarization failed
                logger.warning("Diarization failed, falling back to transcription only")
                return process_downloaded_audio_transcription_only(audio_file_path, include_timestamps, total_duration)
        
        else:
            # Transcription only mode
            logger.info("📝 Processing with transcription only...")
            return process_downloaded_audio_transcription_only(audio_file_path, include_timestamps, total_duration)
            
    except Exception as e:
        logger.error(f"❌ Error processing downloaded audio: {str(e)}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return {"error": f"Audio processing failed: {str(e)}"}

def process_downloaded_audio_transcription_only(audio_file_path: str, include_timestamps: bool, total_duration: float) -> dict:
    """
    Process downloaded audio file with transcription only (no diarization)
    """
    try:
        logger.info(f"📝 Transcribing downloaded audio file: {audio_file_path}")
        
        # Transcribe the whole file directly (NO CHUNKING!)
        transcription_result = transcribe_audio_file_direct(audio_file_path, include_timestamps)
        
        # Add metadata
        transcription_result.update({
            'audio_duration_seconds': total_duration,
            'whole_file_processed': True,
            'no_chunking_used': True,
            'model_used': 'nvidia/parakeet-tdt-0.6b-v3',
            'processing_method': 'direct_firebase_transcription_no_chunking',
            'long_audio_optimization': 'local_attention_enabled',
            'advantage': 'Entire file processed at once for better accuracy'
        })
        
        logger.info(f"✅ Transcription completed: {len(transcription_result.get('word_timestamps', []))} words")
        return transcription_result
        
    except Exception as e:
        logger.error(f"❌ Transcription error: {str(e)}")
        return {"error": f"Transcription failed: {str(e)}"}

def ensure_mono_audio(audio_path: str) -> str:
    """
    Convert stereo audio to mono if needed
    Returns path to mono audio file (original if already mono)
    """
    try:
        from pydub import AudioSegment
        
        # Load audio and check channels
        audio = AudioSegment.from_file(audio_path)
        
        if audio.channels == 1:
            logger.info("✅ Audio is already mono - no conversion needed")
            return audio_path
        elif audio.channels == 2:
            logger.info(f"🔄 Converting stereo audio ({audio.channels} channels) to mono...")
            
            # Convert to mono
            mono_audio = audio.set_channels(1)
            
            # Create temporary mono file
            temp_mono_path = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_path)[1])
            temp_mono_path.close()
            
            # Export mono audio
            mono_audio.export(temp_mono_path.name, format=os.path.splitext(audio_path)[1][1:])
            
            logger.info(f"✅ Converted to mono: {temp_mono_path.name}")
            return temp_mono_path.name
        else:
            logger.warning(f"⚠️ Unexpected audio format: {audio.channels} channels - attempting to convert to mono")
            # Try to convert to mono anyway
            mono_audio = audio.set_channels(1)
            temp_mono_path = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_path)[1])
            temp_mono_path.close()
            mono_audio.export(temp_mono_path.name, format=os.path.splitext(audio_path)[1][1:])
            return temp_mono_path.name
            
    except Exception as e:
        logger.warning(f"⚠️ Could not convert audio to mono: {str(e)} - using original file")
        return audio_path

def split_audio_into_chunks(audio_path: str, chunk_duration: int = 900, overlap_duration: int = 30) -> List[Dict[str, Any]]:
    """
    Split audio file into chunks for long transcription processing.
    
    Args:
        audio_path: Path to the audio file
        chunk_duration: Duration of each chunk in seconds (default: 900 = 15 minutes)
        overlap_duration: Overlap between chunks in seconds (default: 30)
        
    Returns:
        List of chunk information dictionaries
    """
    try:
        import soundfile as sf
        import numpy as np
        import librosa
        
        logger.info(f"🔪 Splitting audio into {chunk_duration//60}-minute chunks with {overlap_duration}s overlap...")
        
        # Load audio file
        audio_data, sample_rate = sf.read(audio_path, always_2d=True)
        total_samples = audio_data.shape[0]
        total_duration = total_samples / sample_rate
        
        logger.info(f"📊 Audio info: {total_duration/60:.1f} minutes, {sample_rate}Hz, {audio_data.shape[1]} channels")
        
        # Find silence boundaries for better splitting
        silence_boundaries = find_silence_boundaries(audio_data, sample_rate)
        
        chunks = []
        chunk_index = 0
        current_start = 0
        
        while current_start < total_samples:
            # Calculate target end point
            target_end_sample = current_start + int(chunk_duration * sample_rate)
            
            if target_end_sample >= total_samples:
                # Last chunk - use the end of the file
                actual_end_sample = total_samples
            else:
                # Find optimal split point near silence
                actual_end_sample = find_optimal_split_point(
                    target_end_sample, silence_boundaries, sample_rate, chunk_duration
                )
            
            # Add overlap for context
            if actual_end_sample < total_samples:
                chunk_end_sample = min(actual_end_sample + int(overlap_duration * sample_rate), total_samples)
            else:
                chunk_end_sample = actual_end_sample
            
            # Extract chunk data
            chunk_data = audio_data[current_start:chunk_end_sample]
            
            # Create temporary chunk file
            temp_chunk_path = tempfile.NamedTemporaryFile(
                suffix='.wav', 
                delete=False,
                prefix=f'chunk_{chunk_index:03d}_'
            ).name
            
            # Write chunk to file
            sf.write(temp_chunk_path, chunk_data, sample_rate)
            
            # Calculate times
            start_time = current_start / sample_rate
            end_time = chunk_end_sample / sample_rate
            actual_duration = (chunk_end_sample - current_start) / sample_rate
            
            chunk_info = {
                "id": f"chunk_{chunk_index}",
                "index": chunk_index,
                "start_time": start_time,
                "end_time": end_time,
                "duration": actual_duration,
                "file_path": temp_chunk_path,
                "start_sample": current_start,
                "end_sample": chunk_end_sample,
                "sample_rate": sample_rate,
                "channels": audio_data.shape[1],
                "status": "ready"
            }
            chunks.append(chunk_info)
            
            logger.info(f"✅ Created chunk {chunk_index}: {start_time/60:.1f}min - {end_time/60:.1f}min ({actual_duration/60:.1f}min)")
            
            # Move to next chunk start (without overlap)
            current_start = actual_end_sample
            chunk_index += 1
            
            # Safety check
            if chunk_index > 100:
                raise Exception("Too many chunks generated - possible infinite loop")
        
        logger.info(f"🎯 Successfully created {len(chunks)} chunks for processing")
        return chunks
        
    except Exception as e:
        logger.error(f"❌ Failed to split audio into chunks: {e}")
        raise

def find_silence_boundaries(audio_data: np.ndarray, sample_rate: int, top_db: int = 25) -> List[int]:
    """
    Find silence boundaries in the audio using librosa.
    """
    try:
        # Convert to mono for silence detection
        if len(audio_data.shape) > 1:
            mono_audio = np.mean(audio_data, axis=1)
        else:
            mono_audio = audio_data
        
        # Find non-silent regions
        non_silent_intervals = librosa.effects.split(mono_audio, top_db=top_db)
        
        # Extract silence boundaries
        silence_boundaries = [0]  # Start of file
        
        for i in range(len(non_silent_intervals) - 1):
            silence_start = non_silent_intervals[i][1]
            silence_end = non_silent_intervals[i + 1][0]
            silence_middle = (silence_start + silence_end) // 2
            silence_boundaries.append(silence_middle)
        
        silence_boundaries.append(len(mono_audio))  # End of file
        
        logger.info(f"🔍 Found {len(silence_boundaries)} silence boundaries")
        return sorted(set(silence_boundaries))
        
    except Exception as e:
        logger.warning(f"⚠️ Silence detection failed: {e}, using time-based splitting")
        return [0, len(audio_data)]

def find_optimal_split_point(target_sample: int, silence_boundaries: List[int], 
                           sample_rate: int, chunk_duration: int) -> int:
    """
    Find the optimal split point near the target sample using silence boundaries.
    """
    max_deviation = 60  # Allow ±1 minute deviation
    max_deviation_samples = max_deviation * sample_rate
    
    # Find silence boundaries within acceptable range
    valid_boundaries = []
    for boundary in silence_boundaries:
        if abs(boundary - target_sample) <= max_deviation_samples:
            valid_boundaries.append(boundary)
    
    if valid_boundaries:
        # Choose the boundary closest to target
        optimal_point = min(valid_boundaries, key=lambda x: abs(x - target_sample))
        deviation_sec = abs(optimal_point - target_sample) / sample_rate
        logger.info(f"🎯 Found silence boundary {deviation_sec:.1f}s from target")
        return optimal_point
    else:
        logger.warning(f"⚠️ No silence boundary within ±{max_deviation}s, using target point")
        return target_sample

def merge_chunk_results(chunk_results: List[Dict[str, Any]], overlap_duration: int = 30) -> Dict[str, Any]:
    """
    Merge transcription results from multiple chunks into a single result.
    
    Args:
        chunk_results: List of transcription results from each chunk
        overlap_duration: Overlap duration in seconds to handle
        
    Returns:
        Merged transcription result
    """
    try:
        logger.info(f"🔗 Merging {len(chunk_results)} chunk results...")
        
        if not chunk_results:
            return {"error": "No chunk results to merge"}
        
        if len(chunk_results) == 1:
            return chunk_results[0]
        
        # Initialize merged result
        merged_result = {
            "transcript": "",
            "diarized_transcript": [],
            "word_timestamps": [],
            "segment_timestamps": [],
            "char_timestamps": [],
            "metadata": {
                "chunks_processed": len(chunk_results),
                "processing_method": "chunked_transcription"
            }
        }
        
        current_time_offset = 0
        
        for i, chunk_result in enumerate(chunk_results):
            logger.info(f"🔗 Processing chunk {i+1}/{len(chunk_results)}")
            
            # Extract chunk data
            chunk_transcript = chunk_result.get("transcript", "")
            chunk_diarized = chunk_result.get("diarized_transcript", [])
            chunk_word_timestamps = chunk_result.get("word_timestamps", [])
            chunk_segment_timestamps = chunk_result.get("segment_timestamps", [])
            chunk_char_timestamps = chunk_result.get("char_timestamps", [])
            
            if i == 0:
                # First chunk - use entire content
                merged_result["transcript"] = chunk_transcript
                merged_result["diarized_transcript"] = chunk_diarized
                merged_result["word_timestamps"] = chunk_word_timestamps
                merged_result["segment_timestamps"] = chunk_segment_timestamps
                merged_result["char_timestamps"] = chunk_char_timestamps
                
                # Set time offset for next chunk
                if chunk_segment_timestamps:
                    current_time_offset = max(ts.get("end", ts.get("end_time", 0)) for ts in chunk_segment_timestamps)
                else:
                    current_time_offset = 0
                    
            else:
                # Subsequent chunks - adjust timestamps and merge
                
                # Adjust diarized transcript timestamps
                for segment in chunk_diarized:
                    adjusted_segment = segment.copy()
                    if "start_time" in adjusted_segment:
                        adjusted_segment["start_time"] += current_time_offset - overlap_duration
                    if "end_time" in adjusted_segment:
                        adjusted_segment["end_time"] += current_time_offset - overlap_duration
                    merged_result["diarized_transcript"].append(adjusted_segment)
                
                # Adjust word timestamps
                for word_ts in chunk_word_timestamps:
                    adjusted_word = word_ts.copy()
                    if "start" in adjusted_word:
                        adjusted_word["start"] += current_time_offset - overlap_duration
                    if "end" in adjusted_word:
                        adjusted_word["end"] += current_time_offset - overlap_duration
                    merged_result["word_timestamps"].append(adjusted_word)
                
                # Adjust segment timestamps
                for seg_ts in chunk_segment_timestamps:
                    adjusted_seg = seg_ts.copy()
                    if "start" in adjusted_seg:
                        adjusted_seg["start"] += current_time_offset - overlap_duration
                    if "end" in adjusted_seg:
                        adjusted_seg["end"] += current_time_offset - overlap_duration
                    merged_result["segment_timestamps"].append(adjusted_seg)
                
                # Adjust char timestamps
                for char_ts in chunk_char_timestamps:
                    adjusted_char = char_ts.copy()
                    if "start" in adjusted_char:
                        adjusted_char["start"] += current_time_offset - overlap_duration
                    if "end" in adjusted_char:
                        adjusted_char["end"] += current_time_offset - overlap_duration
                    merged_result["char_timestamps"].append(adjusted_char)
                
                # Merge transcript text
                if merged_result["transcript"] and chunk_transcript:
                    merged_result["transcript"] += " " + chunk_transcript
                elif chunk_transcript:
                    merged_result["transcript"] = chunk_transcript
                
                # Update time offset for next chunk
                if chunk_segment_timestamps:
                    chunk_end_time = max(ts.get("end", ts.get("end_time", 0)) for ts in chunk_segment_timestamps)
                    current_time_offset += chunk_end_time - overlap_duration
                else:
                    current_time_offset += 900  # Default 15 minutes if no timestamps
        
        # Clean up merged result
        merged_result["transcript"] = merged_result["transcript"].strip()
        
        # Calculate final statistics
        total_duration = current_time_offset
        word_count = len(merged_result["transcript"].split())
        
        merged_result["metadata"].update({
            "total_duration": total_duration,
            "word_count": word_count,
            "speaker_count": len(set(seg.get("speaker", "") for seg in merged_result["diarized_transcript"])),
            "total_segments": len(merged_result["diarized_transcript"]),
            "total_words": len(merged_result["word_timestamps"]),
            "total_characters": len(merged_result["transcript"])
        })
        
        logger.info(f"✅ Successfully merged {len(chunk_results)} chunks")
        logger.info(f"📊 Final result: {word_count} words, {total_duration/60:.1f} minutes")
        
        return merged_result
        
    except Exception as e:
        logger.error(f"❌ Failed to merge chunk results: {e}")
        return {"error": f"Failed to merge chunk results: {e}"}

def transcribe_long_audio(audio_path: str, include_timestamps: bool = True, 
                         chunk_duration: int = 900, overlap_duration: int = 30) -> Dict[str, Any]:
    """
    Transcribe long audio files by splitting into chunks and merging results.
    
    Args:
        audio_path: Path to the audio file
        include_timestamps: Whether to include timestamps
        chunk_duration: Duration of each chunk in seconds (default: 900 = 15 minutes)
        overlap_duration: Overlap between chunks in seconds (default: 30)
        
    Returns:
        Complete transcription result
    """
    try:
        logger.info(f"🎯 Starting long audio transcription: {audio_path}")
        
        # Ensure audio is mono
        mono_audio_path = ensure_mono_audio(audio_path)
        temp_files_to_cleanup = []
        
        if mono_audio_path != audio_path:
            temp_files_to_cleanup.append(mono_audio_path)
        
        try:
            # Split audio into chunks
            chunks = split_audio_into_chunks(mono_audio_path, chunk_duration, overlap_duration)
            
            if len(chunks) == 1:
                # Single chunk - process directly
                logger.info("📝 Single chunk detected - processing directly")
                result = transcribe_audio_file_direct(mono_audio_path, include_timestamps)
                return result
            
            # Process each chunk
            chunk_results = []
            for i, chunk in enumerate(chunks):
                logger.info(f"🎤 Transcribing chunk {i+1}/{len(chunks)}: {chunk['file_path']}")
                
                try:
                    # Transcribe chunk
                    chunk_result = transcribe_audio_file_direct(chunk['file_path'], include_timestamps)
                    
                    if chunk_result.get("error"):
                        logger.error(f"❌ Chunk {i+1} failed: {chunk_result['error']}")
                        continue
                    
                    chunk_results.append(chunk_result)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to transcribe chunk {i+1}: {e}")
                    continue
                
                finally:
                    # Clean up chunk file
                    try:
                        if os.path.exists(chunk['file_path']):
                            os.unlink(chunk['file_path'])
                    except Exception as cleanup_error:
                        logger.warning(f"⚠️ Could not clean up chunk file: {cleanup_error}")
            
            if not chunk_results:
                return {"error": "All chunks failed to transcribe"}
            
            # Merge results
            logger.info(f"🔗 Merging {len(chunk_results)} successful chunks...")
            merged_result = merge_chunk_results(chunk_results, overlap_duration)
            
            return merged_result
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files_to_cleanup:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                        logger.info(f"🧹 Cleaned up temporary file: {temp_file}")
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Could not clean up temporary file {temp_file}: {cleanup_error}")
        
    except Exception as e:
        logger.error(f"❌ Long audio transcription failed: {e}")
        return {"error": f"Long audio transcription failed: {e}"}

def process_long_audio_with_chunking(audio_file_path: str, include_timestamps: bool, use_diarization: bool,
                                   num_speakers: int = None, hf_token: str = None, audio_format: str = "wav",
                                   total_duration: float = 0) -> dict:
    """
    Process long audio files using chunking for better memory management and accuracy.
    
    Args:
        audio_file_path: Path to the audio file
        include_timestamps: Whether to include timestamps
        use_diarization: Whether to use speaker diarization
        num_speakers: Expected number of speakers
        hf_token: Hugging Face token for diarization
        audio_format: Audio format
        total_duration: Total duration of the audio file
        
    Returns:
        Complete transcription result with diarization
    """
    try:
        logger.info(f"🔪 Processing long audio with chunking: {total_duration/60:.1f} minutes")
        
        # Load diarization model if needed
        if use_diarization and diarization_model is None and hf_token:
            logger.info("🎤 Loading pyannote diarization model for chunked processing...")
            if not load_diarization_model(hf_token):
                return {"error": "Failed to load diarization model with provided HF token"}
        elif use_diarization and diarization_model is None:
            return {"error": "Diarization requested but no HF token provided"}
        
        # Use the transcribe_long_audio function for chunked transcription
        logger.info("🎤 Starting chunked transcription...")
        transcription_result = transcribe_long_audio(
            audio_file_path, 
            include_timestamps=include_timestamps,
            chunk_duration=900,  # 15 minutes
            overlap_duration=30  # 30 seconds overlap
        )
        
        if transcription_result.get("error"):
            return transcription_result
        
        if not use_diarization:
            # Just return transcription without diarization
            return {
                **transcription_result,
                "workflow": "chunked_transcription_only",
                "total_duration": total_duration,
                "processing_method": "chunked_no_diarization"
            }
        
        # Run diarization on the complete audio file
        logger.info("🎤 Running speaker diarization on complete audio file...")
        diarized_segments = perform_speaker_diarization(audio_file_path, num_speakers)
        
        if not diarized_segments:
            logger.warning("⚠️ No diarized segments found, returning transcription only")
            return {
                **transcription_result,
                "workflow": "chunked_transcription_fallback",
                "total_duration": total_duration,
                "processing_method": "chunked_diarization_failed"
            }
        
        # Match timestamps to assign speakers
        logger.info("🔗 Matching timestamps for speaker assignment...")
        
        if transcription_result.get('text'):
            # Use segment-level timestamps for matching
            segment_timestamps = transcription_result.get('segment_timestamps', [])
            
            diarized_results = []
            
            if segment_timestamps:
                logger.info(f"📊 Using {len(segment_timestamps)} segment timestamps for speaker assignment")
                
                # Get the full transcribed text
                full_text = transcription_result.get('text', '')
                word_timestamps = transcription_result.get('word_timestamps', [])
                
                # Match each segment timestamp to a speaker
                for segment in segment_timestamps:
                    segment_start = segment.get('start', segment.get('start_time', 0))
                    segment_end = segment.get('end', segment.get('end_time', 0))
                    segment_text = segment.get('text', '')
                    
                    # Find the best matching speaker for this time segment
                    best_speaker = find_best_speaker_for_time_segment(
                        diarized_segments, segment_start, segment_end
                    )
                    
                    if best_speaker:
                        diarized_results.append({
                            "speaker": best_speaker,
                            "start_time": segment_start,
                            "end_time": segment_end,
                            "text": segment_text
                        })
                    else:
                        # Fallback to first speaker if no match found
                        diarized_results.append({
                            "speaker": "Speaker_00",
                            "start_time": segment_start,
                            "end_time": segment_end,
                            "text": segment_text
                        })
                
                logger.info(f"✅ Successfully assigned speakers to {len(diarized_results)} segments")
                
            else:
                # Fallback: assign entire transcript to first speaker
                logger.warning("⚠️ No segment timestamps available, assigning entire transcript to first speaker")
                diarized_results = [{
                    "speaker": "Speaker_00",
                    "start_time": 0,
                    "end_time": total_duration,
                    "text": transcription_result.get('text', '')
                }]
            
            # Calculate statistics
            unique_speakers = set(seg["speaker"] for seg in diarized_results)
            word_count = len(transcription_result.get('text', '').split())
            
            result = {
                "transcript": transcription_result.get('text', ''),
                "diarized_transcript": diarized_results,
                "word_timestamps": transcription_result.get('word_timestamps', []),
                "segment_timestamps": transcription_result.get('segment_timestamps', []),
                "char_timestamps": transcription_result.get('char_timestamps', []),
                "metadata": {
                    **transcription_result.get('metadata', {}),
                    "total_duration": total_duration,
                    "speaker_count": len(unique_speakers),
                    "word_count": word_count,
                    "diarized_segments": len(diarized_results),
                    "processing_method": "chunked_with_diarization",
                    "chunks_processed": transcription_result.get('metadata', {}).get('chunks_processed', 1)
                },
                "workflow": "chunked_transcription_with_diarization"
            }
            
            logger.info(f"🎉 Chunked processing completed successfully!")
            logger.info(f"📊 Final stats: {word_count} words, {len(unique_speakers)} speakers, {len(diarized_results)} segments")
            
            return result
            
        else:
            return {"error": "No transcription text available for speaker assignment"}
            
    except Exception as e:
        logger.error(f"❌ Chunked processing failed: {str(e)}")
        return {"error": f"Chunked processing failed: {str(e)}"}

def handler(job):
    """
    RunPod handler function for audio transcription with DIRECT Firebase Storage integration
    
    Expected input format:
    
    OPTION 1 - Direct File URL Processing (RECOMMENDED for large files):
    Send a file URL to RunPod, which downloads and processes it directly:
    {
        "input": {
            "audio_url": "https://example.com/audio.wav",  # Direct URL to audio file
            "audio_format": "wav",  # Audio format (wav, mp3, flac, etc.)
            "include_timestamps": true,  # Optional: include word/segment timestamps
            "use_diarization": true,  # Optional: enable speaker diarization
            "num_speakers": null,  # Optional: expected number of speakers
            "hf_token": "hf_xxx",  # Required for diarization
        }
    }
    
    OPTION 2 - Raw File Upload (for smaller files):
    POST /your-endpoint with multipart/form-data:
    - file: audio file (WAV, MP3, FLAC, etc.)
    - include_timestamps: true/false
    - use_diarization: true/false
    - num_speakers: number (optional)
    - hf_token: "hf_xxx"
    - firebase_upload: true/false (optional, auto-enabled for files > 10MB)
    
    OPTION 3 - JSON with base64 (legacy, limited to 10MiB):
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
        # Handle different input modes
        if "input" in job:
            # JSON mode - check for audio_url first (RECOMMENDED)
            job_input = job["input"]
            
            # OPTION 1: Direct Firebase URL (RECOMMENDED - no size limits!)
            if "audio_url" in job_input:
                audio_url = job_input["audio_url"]
                audio_format = job_input.get("audio_format", "wav")
                include_timestamps = job_input.get("include_timestamps", True)
                use_diarization = job_input.get("use_diarization", True)
                num_speakers = job_input.get("num_speakers", None)
                hf_token = job_input.get("hf_token", None)
                
                logger.info(f"🌐 URL mode: Processing audio from Firebase URL")
                logger.info(f"🔗 URL: {audio_url[:50]}...")
                
                # Download file from Firebase URL
                try:
                    local_audio_file = download_from_firebase(audio_url)
                    if not local_audio_file:
                        return {"error": "Failed to download audio from Firebase URL"}
                    
                    # Get file size for logging
                    file_size = os.path.getsize(local_audio_file)
                    file_size_mb = file_size / 1024 / 1024
                    logger.info(f"📁 Downloaded: {local_audio_file} ({file_size_mb:.1f}MB)")
                    
                    # Process the downloaded file directly
                    result = process_downloaded_audio(
                        audio_file_path=local_audio_file,
                        include_timestamps=include_timestamps,
                        use_diarization=use_diarization,
                        num_speakers=num_speakers,
                        hf_token=hf_token,
                        audio_format=audio_format
                    )
                    
                    # Clean up downloaded file
                    try:
                        os.unlink(local_audio_file)
                        logger.info(f"🧹 Cleaned up downloaded file: {local_audio_file}")
                    except:
                        pass
                    
                    # Add metadata
                    result.update({
                        'workflow': 'firebase_url_direct_download',
                        'audio_url': audio_url,
                        'file_size_mb': file_size_mb,
                        'processing_method': 'url_download_no_base64'
                    })
                    
                    logger.info(f"🎉 Firebase URL workflow completed successfully!")
                    
                    # Clear memory after processing
                    clear_gpu_memory()
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"❌ Firebase URL download failed: {str(e)}")
                    # Clear memory on error
                    clear_gpu_memory()
                    return {"error": f"Failed to download from Firebase URL: {str(e)}"}
            
            # OPTION 2: Legacy base64 mode (limited to 10MiB)
            elif "audio_data" in job_input:
                logger.info(f"📦 JSON mode: Received base64 audio data (limited to 10MiB)")
                
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
                return {"error": "Missing required parameter: audio_url or audio_data"}
            
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
                
                # Step 2: Download from Firebase Storage with retry logic
                logger.info("📥 Step 2: Direct download from Firebase Storage...")
                max_download_retries = 10  # Increased for large files
                download_retry_count = 0
                local_audio_file = None
                expected_size = len(audio_bytes)
                
                while download_retry_count < max_download_retries and not local_audio_file:
                    try:
                        local_audio_file = download_from_firebase(firebase_url)
                        if local_audio_file:
                            # Verify the downloaded file is complete
                            actual_size = os.path.getsize(local_audio_file)
                            logger.info(f"📊 Downloaded file size: {actual_size} bytes (expected: {expected_size})")
                            
                            if actual_size == expected_size:
                                logger.info(f"✅ Download successful and complete on attempt {download_retry_count + 1}")
                                break
                            elif actual_size > 0:
                                logger.warning(f"⚠️ Partial download: {actual_size}/{expected_size} bytes")
                                # Delete partial file and retry
                                try:
                                    os.unlink(local_audio_file)
                                except:
                                    pass
                                local_audio_file = None
                            else:
                                logger.warning(f"⚠️ Empty file downloaded")
                                local_audio_file = None
                                
                    except Exception as e:
                        download_retry_count += 1
                        logger.warning(f"❌ Download attempt {download_retry_count} failed: {str(e)}")
                        
                        if download_retry_count < max_download_retries:
                            wait_time = 30  # Simple 30-second wait for large files
                            logger.info(f"⏳ Waiting {wait_time}s before retry {download_retry_count + 1}/{max_download_retries}...")
                            logger.info(f"💡 Large files may take 7+ minutes to upload/download completely")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"❌ All {max_download_retries} download attempts failed")
                
                if not local_audio_file:
                    raise Exception(f"Failed to download complete audio file from Firebase Storage after {max_download_retries} attempts (up to {max_download_retries * 30 / 60:.1f} minutes)")
                
                logger.info(f"🎉 File completely downloaded and verified: {local_audio_file}")
                
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
                
                # Clear memory after processing
                clear_gpu_memory()
                
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
    # Clear memory and check CUDA at startup
    clear_gpu_memory()
    ensure_cuda_available()
    
    logger.info("Initializing NVIDIA Parakeet TDT 0.6B v3 model...")
    if load_model():
        logger.info("Parakeet model loaded successfully")
        
        # Diarization model will be loaded on-demand when needed (with HF token if provided)
        logger.info("Pyannote diarization model will be loaded on-demand when diarization is requested")
        
        logger.info("Starting RunPod serverless handler with enhanced Parakeet v3 capabilities...")
        logger.info("🚀 FEATURES: Firebase URL workflow, pyannote diarization, long audio support (3+ hours), streaming mode")
        logger.info("🌐 FIREBASE URL: Send Firebase URL only - no size limits, no base64, direct processing")
        logger.info("🎯 MODEL: NVIDIA Parakeet TDT 0.6B v3 with 25 language support and local attention optimization")
        runpod.serverless.start({"handler": handler})
    else:
        logger.error("Failed to load Parakeet model. Exiting.")
        exit(1)