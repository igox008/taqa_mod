#!/usr/bin/env python3
"""
Equipment Anomaly Training Pipeline - Usage Example
===================================================

This script demonstrates how to use the training pipeline to continuously
improve your ML models with new anomaly data.
"""

import requests
import json
import time
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:5000"  # Adjust for your deployment
HEADERS = {"Content-Type": "application/json"}

def train_models_with_new_data():
    """Example: Train models with new anomaly data"""
    
    print("🎯 Training ML Models with New Anomaly Data")
    print("=" * 50)
    
    # Example training data - Replace with your actual anomaly data
    training_data = [
        {
            "anomaly_id": "TRAIN_001",
            "description": "Excessive vibration detected in main turbine bearing, temperature readings showing 15°C above normal operating range during peak load conditions",
            "equipment_name": "Main Turbine Bearing Assembly", 
            "equipment_id": "7d34d01e-6874-40c0-bbdc-77b8bc8ebba8",
            "availability_score": 2,      # Low availability due to vibration
            "fiability_score": 2,         # Reliability concern  
            "process_safety_score": 3     # Moderate safety impact
        },
        {
            "anomaly_id": "TRAIN_002", 
            "description": "Hydraulic oil leak observed at pump connection joint, estimated 5L/hour loss rate with visible staining on equipment floor",
            "equipment_name": "Hydraulic Pump Unit",
            "equipment_id": "64927d6a-f0ff-42fe-a137-fb5a78219e91",
            "availability_score": 3,      # Some impact on availability
            "fiability_score": 1,         # High reliability concern due to leak
            "process_safety_score": 2     # Safety risk from oil leak
        },
        {
            "anomaly_id": "TRAIN_003",
            "description": "Control system responding slowly to setpoint changes, delays of 3-5 seconds observed during automated switching sequences",
            "equipment_name": "Process Control System",
            "equipment_id": "60c3aaa1-ef3d-48de-a360-bee0b615ffab", 
            "availability_score": 4,      # Minor availability impact
            "fiability_score": 3,         # Some reliability concerns
            "process_safety_score": 4     # Low safety impact
        }
    ]
    
    print(f"📤 Sending {len(training_data)} training records...")
    
    # Send training request
    try:
        response = requests.post(
            f"{API_BASE_URL}/train",
            headers=HEADERS,
            json=training_data,
            timeout=300  # 5 minutes timeout for training
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Training completed successfully!")
            print(f"📊 Training Statistics:")
            print(f"   • Total records added: {result['statistics']['total_records_added']}")
            print(f"   • Training time: {result['statistics']['training_time_seconds']} seconds")
            print(f"   • Models retrained: {result['statistics']['models_retrained']}")
            print(f"   • Backup location: {result['statistics']['backup_location']}")
            
            return True
            
        else:
            error_data = response.json()
            print(f"❌ Training failed: {error_data.get('error', 'Unknown error')}")
            if 'validation_errors' in error_data:
                print("📋 Validation errors:")
                for error in error_data['validation_errors']:
                    print(f"   • {error}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False

def check_training_status():
    """Check current training status and history"""
    
    print("\n🔍 Checking Training Status")
    print("=" * 30)
    
    try:
        response = requests.get(f"{API_BASE_URL}/train/status")
        
        if response.status_code == 200:
            status = response.json()
            print(f"📊 Training Status:")
            print(f"   • Currently training: {'Yes' if status['is_training'] else 'No'}")
            print(f"   • Total sessions completed: {status['training_sessions_completed']}")
            print(f"   • Available backups: {status['available_backups']}")
            
            if status['last_training']:
                last = status['last_training']
                print(f"   • Last training: {last['timestamp']}")
                print(f"   • Records added: {last['records_added']}")
                print(f"   • Training time: {last['training_time']}s")
                
        else:
            print(f"❌ Failed to get status: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")

def reload_models():
    """Reload models after training to use updated versions"""
    
    print("\n🔄 Reloading Models")
    print("=" * 20)
    
    try:
        response = requests.post(f"{API_BASE_URL}/train/reload")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Models reloaded successfully!")
            print(f"📊 Models loaded: {result['models_loaded']}")
        else:
            error_data = response.json()
            print(f"❌ Model reload failed: {error_data.get('error', 'Unknown error')}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")

def list_available_backups():
    """List all available model backups"""
    
    print("\n📦 Available Model Backups")
    print("=" * 30)
    
    try:
        response = requests.get(f"{API_BASE_URL}/train/backups")
        
        if response.status_code == 200:
            backups = response.json()
            print(f"📊 Found {backups['count']} backups:")
            
            for backup in backups['backups']:
                print(f"   • {backup['name']} (Created: {backup['created']})")
                
        else:
            print(f"❌ Failed to list backups: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")

def test_prediction_after_training():
    """Test predictions after training to verify improvements"""
    
    print("\n🧪 Testing Predictions After Training")
    print("=" * 40)
    
    # Test anomaly data
    test_anomaly = {
        "anomaly_id": "TEST_001",
        "description": "Bearing temperature elevated beyond normal operating range with slight vibration detected",
        "equipment_name": "Main Turbine Bearing",
        "equipment_id": "7d34d01e-6874-40c0-bbdc-77b8bc8ebba8"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            headers=HEADERS,
            json=test_anomaly
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"📊 Prediction Results:")
            print(f"   • Availability: {result['predictions']['availability']['score']}")
            print(f"   • Reliability: {result['predictions']['reliability']['score']}")
            print(f"   • Process Safety: {result['predictions']['process_safety']['score']}")
            print(f"   • Overall Score: {result['overall_score']}")
            print(f"   • Risk Level: {result['risk_assessment']['overall_risk_level']}")
            
        else:
            error_data = response.json()
            print(f"❌ Prediction failed: {error_data.get('error', 'Unknown error')}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")

def main():
    """Main training pipeline demonstration"""
    
    print("🚀 Equipment Anomaly Training Pipeline Demo")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Check initial status
    check_training_status()
    
    # Step 2: List available backups
    list_available_backups()
    
    # Step 3: Train with new data
    training_success = train_models_with_new_data()
    
    if training_success:
        # Step 4: Wait a moment then check status again
        time.sleep(2)
        check_training_status()
        
        # Step 5: Reload models to use updated versions
        reload_models()
        
        # Step 6: Test predictions with updated models
        test_prediction_after_training()
        
        print("\n🎉 Training pipeline demo completed successfully!")
        print("\n💡 Next Steps:")
        print("   1. Integrate this into your monitoring system")
        print("   2. Set up automated training with new anomaly data")
        print("   3. Monitor model performance improvements")
        print("   4. Use backups to restore if needed")
        
    else:
        print("\n❌ Training failed - check the error messages above")

if __name__ == "__main__":
    main() 