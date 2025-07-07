#!/usr/bin/env python3
"""
Test script for deployed Equipment Prediction API
"""
import requests
import json

# Replace with your actual server IP
SERVER_IP = "YOUR_SERVER_IP"  # e.g., "134.122.123.45"
API_URL = f"http://{SERVER_IP}"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_models_info():
    """Test the models info endpoint"""
    print("\n📋 Testing models info...")
    try:
        response = requests.get(f"{API_URL}/models/info", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Models info failed: {e}")
        return False

def test_prediction():
    """Test the prediction endpoint"""
    print("\n🎯 Testing prediction...")
    
    # Example prediction request
    test_data = {
        "anomaly_id": "TEST-API-001",
        "description": "Fuite importante d'huile avec vibration anormale et température élevée",
        "equipment_name": "POMPE FUEL PRINCIPALE N°1",
        "equipment_id": "98b82203-7170-45bf-879e-f47ba6e12c86"
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30  # Prediction might take longer
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"Anomaly ID: {result['anomaly_id']}")
            print(f"Equipment: {result['equipment_name']}")
            print("\n📊 Predictions:")
            for pred_type, pred_data in result['predictions'].items():
                print(f"  {pred_type.title()}: {pred_data['score']}")
            print(f"\n🎯 Overall Score: {result['overall_score']}")
            print(f"📈 Total Sum: {result['total_sum']}")
            print(f"⚠️  Risk Level: {result['risk_assessment']['overall_risk_level']}")
        else:
            print(f"❌ Prediction failed: {response.text}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Prediction request failed: {e}")
        return False

def test_missing_fields():
    """Test validation with missing fields"""
    print("\n🚫 Testing validation (missing fields)...")
    
    incomplete_data = {
        "anomaly_id": "TEST-VALIDATION-001",
        "description": "Test anomaly"
        # Missing equipment_name and equipment_id
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=incomplete_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # Should return 400 for missing fields
        return response.status_code == 400
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

def main():
    """Run all API tests"""
    print("🚀 Testing Equipment Prediction API")
    print(f"🌐 Server: {API_URL}")
    print("=" * 50)
    
    if SERVER_IP == "YOUR_SERVER_IP":
        print("❌ Please update SERVER_IP in this script with your actual server IP")
        return
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Models Info", test_models_info),
        ("Prediction", test_prediction),
        ("Validation", test_missing_fields)
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the API server and try again.")

if __name__ == "__main__":
    main() 