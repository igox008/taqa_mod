#!/usr/bin/env python3
"""
Test script for the DigitalOcean deployed API
Update the SERVER_IP with your actual DigitalOcean droplet IP
"""

import requests
import json

# ⚠️ REPLACE WITH YOUR DIGITALOCEAN DROPLET IP
SERVER_IP = "YOUR_DROPLET_IP_HERE"  # e.g., "164.90.123.45"
API_BASE_URL = f"http://{SERVER_IP}:5000"

def test_deployed_api():
    """Test the deployed API on DigitalOcean"""
    print(f"🧪 Testing DigitalOcean Deployed API: {API_BASE_URL}")
    print("=" * 60)
    
    # Test health check
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print(f"   Make sure your server is running and accessible at {API_BASE_URL}")
        return False
    
    # Test single prediction
    print("\n🔍 Testing single prediction...")
    single_anomaly = {
        "anomaly_id": "DEPLOY-TEST-001",
        "description": "Fuite importante d'huile au niveau du palier avec vibrations anormales",
        "equipment_name": "POMPE FUEL PRINCIPALE N°1", 
        "equipment_id": "98b82203-7170-45bf-879e-f47ba6e12c86"
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=single_anomaly)
        if response.status_code == 200:
            result = response.json()
            print("✅ Single prediction successful!")
            print(f"   Anomaly ID: {result['anomaly_id']}")
            print(f"   Overall Score: {result['overall_score']}")
            print(f"   Availability: {result['predictions']['availability']['score']}")
            print(f"   Reliability: {result['predictions']['reliability']['score']}")
            print(f"   Process Safety: {result['predictions']['process_safety']['score']}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction request failed: {e}")
        return False
    
    # Test batch prediction
    print("\n🔄 Testing batch prediction...")
    batch_anomalies = [
        {
            "anomaly_id": "BATCH-001",
            "description": "Maintenance préventive normale",
            "equipment_name": "Equipment 1",
            "equipment_id": "uuid-1"
        },
        {
            "anomaly_id": "BATCH-002", 
            "description": "Fuite importante vapeur toxique",
            "equipment_name": "Equipment 2",
            "equipment_id": "uuid-2"
        }
    ]
    
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=batch_anomalies)
        if response.status_code == 200:
            result = response.json()
            batch_info = result['batch_info']
            print("✅ Batch prediction successful!")
            print(f"   Total Anomalies: {batch_info['total_anomalies']}")
            print(f"   Successful: {batch_info['successful_predictions']}")
            print(f"   Processing Time: {batch_info['processing_time_seconds']}s")
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Batch request failed: {e}")
    
    print("\n🎉 DigitalOcean deployment test completed!")
    print(f"\n📡 Your API is live at: {API_BASE_URL}")
    print(f"📋 Available endpoints:")
    print(f"   GET  {API_BASE_URL}/health")
    print(f"   GET  {API_BASE_URL}/models/info")
    print(f"   POST {API_BASE_URL}/predict")
    
    return True

if __name__ == "__main__":
    if SERVER_IP == "YOUR_DROPLET_IP_HERE":
        print("❌ Please update SERVER_IP with your DigitalOcean droplet IP address!")
        print("   Edit this file and replace 'YOUR_DROPLET_IP_HERE' with your actual IP")
        exit(1)
    
    test_deployed_api() 