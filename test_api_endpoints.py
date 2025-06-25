#!/usr/bin/env python3
"""
Test script for TAQA Anomaly Priority API
Demonstrates how to use the API endpoints
"""

import requests
import json
import time

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change this to your deployed URL
# API_BASE_URL = "https://your-app.ondigitalocean.app"

def test_single_anomaly():
    """Test single anomaly priority calculation"""
    print("🧪 Testing Single Anomaly Calculation...")
    
    url = f"{API_BASE_URL}/api/v1/calculate_priority"
    
    # Test data
    test_cases = [
        {
            "description": "Pompe alimentaire principale fait du bruit anormal et vibrations importantes",
            "equipment": "POMPE ALIMENTAIRE PRINCIPALE",
            "department": "34MC"
        },
        {
            "description": "Panne électrique sur alternateur unité 2 - arrêt immédiat requis",
            "equipment": "ALTERNATEUR UNITE 2", 
            "department": "34EL"
        },
        {
            "description": "Éclairage bureau maintenance défaillant",
            "equipment": "ECLAIRAGE",
            "department": "34MM"
        },
        {
            "description": "Fuite importante sur chaudière principale - danger sécurité",
            "equipment": "CHAUDIERE PRINCIPALE",
            "department": "34MC"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Equipment: {test_case['equipment']}")
        print(f"Department: {test_case['department']}")
        print(f"Description: {test_case['description']}")
        
        try:
            response = requests.post(
                url,
                json=test_case,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Priority Score: {result['priority_score']}")
                print(f"📊 Priority Label: {result['priority_label']}")
                print(f"🎯 Confidence: {result['confidence']}")
                print(f"🔍 Method: {result['method']}")
                print(f"💡 Explanation: {result['explanation']}")
                print(f"⚡ Processing Time: {result['processing_time_ms']}ms")
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection Error: {e}")

def test_batch_anomalies():
    """Test batch anomaly processing"""
    print("\n🧪 Testing Batch Anomaly Calculation...")
    
    url = f"{API_BASE_URL}/api/v1/batch_calculate"
    
    batch_data = {
        "anomalies": [
            {
                "id": "ANOM_001",
                "description": "Transformateur en surchauffe",
                "equipment": "TRANSFORMATEUR",
                "department": "34EL"
            },
            {
                "id": "ANOM_002", 
                "description": "Ventilateur tirage forcé fait du bruit",
                "equipment": "VENTILATEUR TIRAGE FORCE",
                "department": "34MC"
            },
            {
                "id": "ANOM_003",
                "description": "Câble d'alimentation endommagé",
                "equipment": "CABLE ALIMENTATION",
                "department": "34EL"
            },
            {
                "id": "ANOM_004",
                "description": "Maintenance préventive programmée",
                "equipment": "POMPE",
                "department": "34MM"
            }
        ]
    }
    
    try:
        response = requests.post(
            url,
            json=batch_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch Status: {result['status']}")
            print(f"📊 Total Processed: {result['total_processed']}")
            print(f"✅ Successful: {result['successful']}")
            print(f"❌ Failed: {result['failed']}")
            print(f"⚡ Total Processing Time: {result['processing_time_ms']}ms")
            
            print("\n📋 Individual Results:")
            for r in result['results']:
                print(f"  {r['id']}: {r['priority_score']} ({r['priority_label']}) - {r['method']}")
                
            if result['errors']:
                print("\n❌ Errors:")
                for error in result['errors']:
                    print(f"  {error['id']}: {error['error']}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")

def test_api_info():
    """Test API information endpoint"""
    print("\n🧪 Testing API Info...")
    
    url = f"{API_BASE_URL}/api/v1/info"
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            info = response.json()
            print(f"✅ API Name: {info['api_name']}")
            print(f"📊 Version: {info['version']}")
            print(f"🎯 Status: {info['status']}")
            print("\n📋 Available Endpoints:")
            for endpoint, details in info['endpoints'].items():
                print(f"  {endpoint} ({details['method']}): {details['description']}")
            
            print(f"\n🏭 Supported Departments: {len(info['supported_departments'])}")
            print(f"🎯 Known Equipment Accuracy: {info['accuracy']['known_equipment']}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")

def test_health_check():
    """Test health check endpoint"""
    print("\n🧪 Testing Health Check...")
    
    url = f"{API_BASE_URL}/health"
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Status: {health['status']}")
            print(f"🤖 Model: {health['model']}")
            print(f"🎯 Lookup Accuracy: {health['lookup_accuracy']}")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection Error: {e}")

def main():
    """Run all API tests"""
    print("🚀 TAQA Anomaly Priority API Test Suite")
    print("=" * 50)
    
    # Test health first
    test_health_check()
    
    # Test API info
    test_api_info()
    
    # Test single anomaly calculation
    test_single_anomaly()
    
    # Test batch processing
    test_batch_anomalies()
    
    print("\n" + "=" * 50)
    print("✅ API Test Suite Completed!")
    print("\n💡 To integrate with your system:")
    print("1. Use /api/v1/calculate_priority for single anomalies")
    print("2. Use /api/v1/batch_calculate for multiple anomalies")
    print("3. Check /api/v1/info for full API documentation")

if __name__ == "__main__":
    main() 