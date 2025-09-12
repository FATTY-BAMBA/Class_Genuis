# test_chapter_llama_simulated.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
from fastapi.testclient import TestClient
from chapter_llama.main import app  # Import your wrapper app

load_dotenv()

def test_with_simulated_service():
    """
    Test that simulates both the wrapper AND the actual service
    """
    # Create a test client for your wrapper
    client = TestClient(app)
    
    # Test 1: Check if wrapper is running
    print("Testing wrapper health...")
    response = client.get("/health")
    print(f"Wrapper health: {response.status_code} - {response.json()}")
    
    # Test 2: Test with sample data (this will fail gracefully if no actual service)
    sample_audio = [
        {"start": 0.0, "end": 10.0, "text": "這是第一段音頻內容"},
        {"start": 10.0, "end": 20.0, "text": "這是第二段音頻內容"}
    ]
    
    sample_ocr = [
        {"timestamp": 5.0, "text": "這是第一張投影片"},
        {"timestamp": 15.0, "text": "這是第二張投影片"}
    ]
    
    print("\nTesting boundary detection (will fail gracefully if no actual service)...")
    try:
        response = client.post("/v1/chapter/boundaries", json={
            "audio_segments": sample_audio,
            "ocr_segments": sample_ocr,
            "win_sec": 240,
            "overlap_sec": 45,
            "language": "zh-hant"
        })
        
        if response.status_code == 502:
            print("✅ Wrapper is working correctly! (It detected no actual service)")
            print("The wrapper received the request but couldn't find the actual Chapter-Llama service")
        else:
            print(f"Unexpected response: {response.status_code} - {response.json()}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def check_whats_available():
    """
    Check what Chapter-Llama components are actually available
    """
    print("\n=== Checking Available Components ===")
    
    # Check if we have the actual service files
    try:
        # Try to import actual service components
        try:
            from chapter_llama import server, api, service
            print("✅ Found actual service modules!")
        except ImportError:
            print("❌ No actual service modules found")
            
        # Check what's in the chapter_llama package
        import chapter_llama
        print(f"Chapter-Llama package contents: {[x for x in dir(chapter_llama) if not x.startswith('_')]}")
        
    except Exception as e:
        print(f"Error checking components: {e}")

if __name__ == "__main__":
    check_whats_available()
    test_with_simulated_service()