#!/usr/bin/env python3
"""
Model Metrics Endpoint Test Script
==================================

This script tests the new /models/metrics endpoint and displays 
comprehensive model performance statistics.
"""

import requests
import json
from datetime import datetime

# API Configuration
API_URL = "http://localhost:5000"

def test_metrics_endpoint():
    """Test the model metrics endpoint and display results"""
    
    print("📊 Testing Model Metrics Endpoint")
    print("=" * 50)
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        print("\n🔍 Fetching model metrics...")
        response = requests.get(f"{API_URL}/models/metrics", timeout=60)
        
        if response.status_code == 200:
            metrics = response.json()
            print("✅ Metrics retrieved successfully!")
            
            # Display key metrics
            print("\n" + "="*60)
            print("📋 MODEL PERFORMANCE SUMMARY")
            print("="*60)
            
            summary = metrics.get("summary", {})
            print(f"Overall Model Health: {summary.get('overall_model_health', 'N/A')}%")
            print(f"Health Status: {summary.get('health_status', 'N/A')}")
            print(f"Models Evaluated: {summary.get('models_evaluated', 'N/A')}")
            print(f"Total Training Records: {summary.get('total_training_records', 'N/A')}")
            
            # Display individual model performance
            performance = metrics.get("model_performance", {})
            for model_name, model_data in performance.items():
                if "regression_metrics" in model_data:
                    print(f"\n🎯 {model_name.upper()} MODEL:")
                    print(f"   R² Score: {model_data['regression_metrics']['r2_score']}")
                    print(f"   MAE: {model_data['regression_metrics']['mae']}")
                    print(f"   RMSE: {model_data['regression_metrics']['rmse']}")
                    print(f"   Accuracy (±0.5): {model_data['accuracy_metrics']['accuracy_within_0.5']}%")
                    print(f"   Accuracy (±1.0): {model_data['accuracy_metrics']['accuracy_within_1.0']}%")
                    print(f"   Test Samples: {model_data['test_samples']}")
                else:
                    print(f"\n❌ {model_name.upper()} MODEL: {model_data.get('error', 'No data')}")
            
            # Display training statistics
            print(f"\n" + "="*60)
            print("📈 TRAINING DATA STATISTICS")
            print("="*60)
            
            training_stats = metrics.get("training_statistics", {})
            for model_name, stats in training_stats.items():
                if "total_records" in stats:
                    print(f"\n📊 {model_name.upper()} DATA:")
                    print(f"   Total Records: {stats['total_records']}")
                    print(f"   Valid Targets: {stats['valid_target_records']}")
                    print(f"   Target Mean: {stats['target_statistics']['mean']}")
                    print(f"   Target Std: {stats['target_statistics']['std']}")
                    
                    # Score distribution
                    dist = stats['target_distribution']
                    print(f"   Score Distribution: 1:{dist['score_1']} | 2:{dist['score_2']} | 3:{dist['score_3']} | 4:{dist['score_4']} | 5:{dist['score_5']}")
                    
                    # Data quality
                    quality = stats['data_quality']
                    print(f"   Missing Descriptions: {quality['missing_descriptions']}")
                    print(f"   Missing Equipment IDs: {quality['missing_equipment_ids']}")
                    print(f"   Duplicate Records: {quality['duplicate_records']}")
                else:
                    print(f"\n❌ {model_name.upper()} DATA: {stats.get('error', 'No data')}")
            
            # Display model file information
            print(f"\n" + "="*60)
            print("📁 MODEL FILES INFORMATION")
            print("="*60)
            
            model_files = metrics.get("model_files", {})
            for model_name, file_info in model_files.items():
                if file_info.get("exists", False):
                    print(f"\n📄 {model_name.upper()} MODEL FILE:")
                    print(f"   Filename: {file_info['filename']}")
                    print(f"   Size: {file_info['size_mb']} MB")
                    print(f"   Last Modified: {file_info['last_modified']}")
                else:
                    print(f"\n❌ {model_name.upper()} MODEL FILE: Not found")
            
            # Display training history
            print(f"\n" + "="*60)
            print("🕰️  TRAINING HISTORY")
            print("="*60)
            
            history = metrics.get("training_history", {})
            print(f"Total Training Sessions: {history.get('total_training_sessions', 0)}")
            print(f"Available Backups: {history.get('available_backups', 0)}")
            print(f"Currently Training: {history.get('is_currently_training', False)}")
            
            last_training = history.get("last_training")
            if last_training:
                print(f"\nLast Training Session:")
                print(f"   Timestamp: {last_training['timestamp']}")
                print(f"   Records Added: {last_training['records_added']}")
                print(f"   Training Time: {last_training['training_time']}s")
                print(f"   Success: {last_training['success']}")
            else:
                print("\nNo training sessions completed yet.")
            
            # Display recommendations
            recommendations = summary.get("recommendations", [])
            if recommendations:
                print(f"\n" + "="*60)
                print("💡 RECOMMENDATIONS")
                print("="*60)
                for i, rec in enumerate(recommendations, 1):
                    print(f"{i}. {rec}")
            
            print(f"\n✅ Metrics test completed successfully!")
            print(f"📊 Full response contains {len(str(metrics))} characters of detailed metrics")
            
            return True
            
        else:
            print(f"❌ Failed to get metrics: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error', 'Unknown error')}")
                print(f"Message: {error_data.get('message', 'No message')}")
            except:
                print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the Flask app is running.")
        print("   Start the API with: python3 app.py")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out. Model metrics calculation might take longer.")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def quick_metrics_summary():
    """Get just a quick summary of model metrics"""
    
    print("\n🚀 Quick Metrics Summary")
    print("=" * 30)
    
    try:
        response = requests.get(f"{API_URL}/models/metrics", timeout=30)
        
        if response.status_code == 200:
            metrics = response.json()
            summary = metrics.get("summary", {})
            
            print(f"📊 Overall Health: {summary.get('overall_model_health', 'N/A')}% ({summary.get('health_status', 'N/A')})")
            print(f"🎯 Models Active: {summary.get('models_evaluated', 0)}/3")
            print(f"📈 Training Records: {summary.get('total_training_records', 0)}")
            
            # Show last training
            history = metrics.get("training_history", {})
            if history.get("last_training"):
                last_training = history["last_training"]["timestamp"]
                print(f"🕰️  Last Training: {last_training}")
            else:
                print("🕰️  Last Training: Never")
            
            return True
        else:
            print(f"❌ Failed to get metrics: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Quick summary failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🧪 Model Metrics Endpoint Test Suite")
    print("=" * 50)
    
    # Test if API is available
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ API health check failed. Ensure the Flask app is running.")
            return False
    except:
        print("❌ Cannot connect to API. Start with: python3 app.py")
        return False
    
    print("✅ API is responding")
    
    # Run tests
    print("\n1️⃣  Running full metrics test...")
    success = test_metrics_endpoint()
    
    if success:
        print("\n2️⃣  Running quick summary test...")
        quick_metrics_summary()
        
        print("\n🎉 All tests completed successfully!")
        print("\n💡 Usage Tips:")
        print("   - Use GET /models/metrics for comprehensive model analysis")
        print("   - Monitor the overall_model_health score regularly")
        print("   - Check recommendations for improvement suggestions")
        print("   - Compare metrics before/after training sessions")
        
    else:
        print("\n❌ Tests failed. Check API status and try again.")
        
    return success

if __name__ == "__main__":
    main() 