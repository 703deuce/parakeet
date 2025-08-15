#!/usr/bin/env python3
"""
Simple connection test for RunPod endpoint
"""

import requests
import os
import json

def test_connection():
    """Test basic connection to RunPod endpoint"""
    
    # Get API key from environment
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        print("Error: RUNPOD_API_KEY environment variable not set")
        return
    
    endpoint_url = "https://api.runpod.ai/v2/7u304yobo6ytm9/run"
    
    print(f"Testing connection to: {endpoint_url}")
    print(f"API Key: {api_key[:10]}...")
    
    # Test with minimal payload
    payload = {
        "input": {
            "test": "connection_test"
        }
    }
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    try:
        print("\nSending test request...")
        response = requests.post(endpoint_url, json=payload, headers=headers, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Body: {response.text}")
        
        if response.status_code == 200:
            print("\n✅ Connection successful!")
        else:
            print(f"\n❌ Connection failed with status {response.status_code}")
            
    except requests.exceptions.Timeout:
        print("\n⏰ Request timed out (30 seconds)")
    except requests.exceptions.ConnectionError as e:
        print(f"\n🔌 Connection error: {e}")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request error: {e}")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")

if __name__ == "__main__":
    test_connection()
