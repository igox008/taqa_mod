#!/usr/bin/env python3
"""
Quick Test Script for Training Pipeline
=======================================

This script performs a quick test of the training pipeline functionality.
Run this to verify your training system is working correctly.
"""

import requests
import json
import sys

# Configuration
API_URL = "http://localhost:5000"
HEADERS = {"Content-Type": "application/json"}

def test_training_pipeline():
    """Quick test of the training pipeline"""
    
    print("🧪 Testing Training Pipeline")
    print("=" * 30)
    
    # Test data with one training sample
    test_data = [{
        "anomaly_id": "TEST_PIPELINE_001",
        "description": "Test anomaly for pipeline validation - pump showing irregular pressure fluctuations during startup sequence",
        "equipment_name": "Test Pump Unit", 
        "equipment_id": "test-uuid-12345",
        "availability_score": 3,
        "fiability_score": 2,
        "process_safety_score": 4
    }]
    
    try:
        # Test training endpoint
        print("📤 Testing training endpoint...")
        response = requests.post(f"{API_URL}/train", headers=HEADERS, json=test_data, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Training endpoint works!")
            print(f"   Records added: {result['statistics']['total_records_added']}")
            print(f"   Training time: {result['statistics']['training_time_seconds']}s")
        else:
            print(f"❌ Training failed: {response.status_code}")
            print(f"   Error: {response.json()}")
            return False
            
        # Test status endpoint
        print("\n📊 Testing status endpoint...")
        response = requests.get(f"{API_URL}/train/status")
        
        if response.status_code == 200:
            status = response.json()
            print("✅ Status endpoint works!")
            print(f"   Total sessions: {status['training_sessions_completed']}")
        else:
            print(f"❌ Status check failed: {response.status_code}")
            
        # Test reload endpoint
        print("\n🔄 Testing model reload...")
        response = requests.post(f"{API_URL}/train/reload")
        
        if response.status_code == 200:
            print("✅ Model reload works!")
        else:
            print(f"❌ Model reload failed: {response.status_code}")
            
        print("\n🎉 All tests passed! Training pipeline is operational.")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the Flask app is running.")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    if test_training_pipeline():
        sys.exit(0)
    else:
        sys.exit(1) 