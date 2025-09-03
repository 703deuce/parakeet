# Configuration file for test_complete_workflow.py
# Update these values with your actual credentials

# RunPod Configuration
RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run"  # Replace with your endpoint
API_KEY = "YOUR_API_KEY"  # Replace with your RunPod API key

# HuggingFace Configuration  
HF_TOKEN = "YOUR_HF_TOKEN"  # Replace with your HuggingFace token

# Audio Processing Configuration
AUDIO_FILE = "test.mp3"  # Audio file to test with
SPEAKER_THRESHOLD = 0.35  # Aggressive speaker merging threshold
SINGLE_SPEAKER_MODE = True  # Assume single speaker for testing
MAX_WAIT_MINUTES = 30  # Maximum time to wait for RunPod job completion

# Firebase Configuration (if using actual Firebase)
FIREBASE_PROJECT_ID = "your-project-id"
FIREBASE_BUCKET_NAME = "your-bucket-name"
FIREBASE_CREDENTIALS_PATH = "path/to/serviceAccountKey.json"
