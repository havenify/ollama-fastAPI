#!/usr/bin/env python3
"""
Test script for the audio transcription API
Usage: python test_transcription.py <audio_file_path>
"""

import requests
import sys
import json

API_BASE_URL = "http://localhost:8288"

def test_health_check():
    """Test if the Whisper service is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/transcribe/health")
        print("Health Check Response:")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_supported_languages():
    """Test getting supported languages"""
    try:
        response = requests.get(f"{API_BASE_URL}/transcribe/languages")
        if response.status_code == 200:
            data = response.json()
            print(f"Supported languages count: {len(data['supported_languages'])}")
            print(f"First 10 languages: {data['supported_languages'][:10]}")
        else:
            print(f"Failed to get languages: {response.status_code}")
    except Exception as e:
        print(f"Error getting languages: {e}")

def transcribe_audio(audio_file_path, language=None, task="transcribe", word_timestamps=False):
    """Test audio transcription"""
    try:
        with open(audio_file_path, 'rb') as audio_file:
            files = {'audio': audio_file}
            data = {
                'task': task,
                'word_timestamps': str(word_timestamps).lower()
            }
            if language:
                data['language'] = language
            
            print(f"Transcribing {audio_file_path}...")
            response = requests.post(f"{API_BASE_URL}/transcribe", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print("\nTranscription Result:")
                print("-" * 50)
                print(f"Text: {result['transcription']['text']}")
                print(f"Language: {result['transcription']['language']} (confidence: {result['transcription']['language_probability']:.2f})")
                print(f"Duration: {result['transcription']['duration']:.2f} seconds")
                print(f"Number of segments: {len(result['transcription']['segments'])}")
                
                if word_timestamps and result['transcription']['segments']:
                    print("\nFirst segment with word timestamps:")
                    first_segment = result['transcription']['segments'][0]
                    if 'words' in first_segment:
                        for word in first_segment['words'][:5]:  # Show first 5 words
                            print(f"  {word['word']} ({word['start']:.2f}s - {word['end']:.2f}s)")
                
                return True
            else:
                print(f"Transcription failed: {response.status_code}")
                print(response.json())
                return False
                
    except FileNotFoundError:
        print(f"Audio file not found: {audio_file_path}")
        return False
    except Exception as e:
        print(f"Error during transcription: {e}")
        return False

def main():
    print("Testing Audio Transcription API")
    print("=" * 40)
    
    # Test health check
    print("\n1. Testing health check...")
    if not test_health_check():
        print("❌ Health check failed. Make sure the server is running.")
        return
    print("✅ Health check passed")
    
    # Test supported languages
    print("\n2. Testing supported languages...")
    test_supported_languages()
    
    # Test transcription if audio file provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        print(f"\n3. Testing transcription with {audio_file}...")
        
        # Basic transcription
        print("\n3a. Basic transcription:")
        if transcribe_audio(audio_file):
            print("✅ Basic transcription successful")
        
        # Transcription with word timestamps
        print("\n3b. Transcription with word timestamps:")
        if transcribe_audio(audio_file, word_timestamps=True):
            print("✅ Word timestamps transcription successful")
        
        # Translation to English (if not English audio)
        print("\n3c. Translation to English:")
        if transcribe_audio(audio_file, task="translate"):
            print("✅ Translation successful")
    else:
        print("\n3. Skipping transcription test (no audio file provided)")
        print("Usage: python test_transcription.py <audio_file_path>")

if __name__ == "__main__":
    main()
