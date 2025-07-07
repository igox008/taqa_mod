#!/usr/bin/env python3
"""
Example script demonstrating how to use the batch API functionality
for processing multiple anomalies at once.
"""

import requests
import json
import time

# API Configuration
API_BASE_URL = "http://localhost:5000"

def test_single_anomaly():
    """Test single anomaly prediction"""
    print("🔍 Testing single anomaly prediction...")
    
    single_anomaly = {
        "anomaly_id": "SINGLE-TEST-001",
        "description": "Fuite importante d'huile au niveau du palier avec vibrations anormales",
        "equipment_name": "POMPE FUEL PRINCIPALE N°1", 
        "equipment_id": "98b82203-7170-45bf-879e-f47ba6e12c86"
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=single_anomaly)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Single prediction successful!")
            print(f"   Anomaly ID: {result['anomaly_id']}")
            print(f"   Overall Score: {result['overall_score']}")
            print(f"   Processing Time: {result['processing_time_seconds']}s")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_batch_anomalies(batch_size=10):
    """Test batch anomaly prediction"""
    print(f"\n🔄 Testing batch prediction with {batch_size} anomalies...")
    
    # Generate sample batch data
    batch_anomalies = []
    sample_descriptions = [
        "Fuite importante d'huile au niveau du palier",
        "Vibrations anormales détectées sur le moteur",
        "Température excessive dans le système de refroidissement",
        "Pression insuffisante dans le circuit hydraulique",
        "Bruit anormal et grincement dans les roulements",
        "Arrêt d'urgence suite à une alarme de sécurité",
        "Maintenance préventive - contrôle de routine",
        "Détection de fumée dans le compartiment électrique",
        "Usure prématurée des joints d'étanchéité",
        "Problème de lubrification du système de transmission"
    ]
    
    sample_equipment = [
        ("POMPE FUEL PRINCIPALE N°1", "98b82203-7170-45bf-879e-f47ba6e12c86"),
        ("COMPRESSEUR AIR N°2", "12345678-1234-5678-9abc-def123456789"),
        ("TURBINE VAPEUR PRINCIPALE", "87654321-4321-8765-cba9-876543210fed"),
        ("MOTEUR ELECTRIQUE N°3", "11111111-2222-3333-4444-555555555555"),
        ("POMPE CIRCULATION N°4", "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    ]
    
    for i in range(batch_size):
        desc_idx = i % len(sample_descriptions)
        equip_idx = i % len(sample_equipment)
        
        anomaly = {
            "anomaly_id": f"BATCH-{i+1:03d}",
            "description": sample_descriptions[desc_idx],
            "equipment_name": sample_equipment[equip_idx][0],
            "equipment_id": sample_equipment[equip_idx][1]
        }
        batch_anomalies.append(anomaly)
    
    try:
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/predict", json=batch_anomalies)
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            batch_info = result['batch_info']
            
            print(f"✅ Batch prediction successful!")
            print(f"   Total Anomalies: {batch_info['total_anomalies']}")
            print(f"   Successful: {batch_info['successful_predictions']}")
            print(f"   Failed: {batch_info['failed_predictions']}")
            print(f"   Processing Time: {batch_info['processing_time_seconds']}s")
            print(f"   Avg Time/Anomaly: {batch_info['average_time_per_anomaly']}s")
            print(f"   Total Request Time: {request_time:.2f}s")
            
            # Show first few results
            print(f"\n📊 Sample Results (first 3):")
            for i, result in enumerate(result['results'][:3]):
                print(f"   {i+1}. {result['anomaly_id']}: Overall={result.get('overall_score', 'N/A')}")
            
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_api_health():
    """Test API health and model info"""
    print("🏥 Checking API health...")
    
    try:
        # Health check
        health_response = requests.get(f"{API_BASE_URL}/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ API Health: {health_data['status']}")
            print(f"   Models Loaded: {health_data['models_loaded']}")
        
        # Model info
        info_response = requests.get(f"{API_BASE_URL}/models/info")
        if info_response.status_code == 200:
            info_data = info_response.json()
            print(f"✅ Models Available:")
            for model_name, model_info in info_data['models'].items():
                print(f"   - {model_name}: {model_info['features']} features")
        
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_large_batch():
    """Test larger batch (100 anomalies)"""
    print(f"\n🚀 Testing larger batch (100 anomalies)...")
    return test_batch_anomalies(100)

def main():
    """Run all API tests"""
    print("🧪 Equipment Prediction API - Batch Testing")
    print("=" * 50)
    
    # Test API health first
    if not test_api_health():
        print("❌ API is not healthy. Make sure the API server is running.")
        print("   Start with: python api_server.py")
        return
    
    print()
    
    # Test single anomaly
    if not test_single_anomaly():
        return
    
    # Test small batch
    if not test_batch_anomalies(10):
        return
    
    # Test larger batch
    if not test_large_batch():
        return
    
    print("\n🎉 All tests completed successfully!")
    print("\nBatch API Features Demonstrated:")
    print("✅ Single anomaly processing (backward compatibility)")
    print("✅ Small batch processing (10 anomalies)")
    print("✅ Large batch processing (100 anomalies)")
    print("✅ Performance metrics and progress tracking")
    print("✅ Error handling and validation")

if __name__ == "__main__":
    main() 