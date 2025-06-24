#!/usr/bin/env python3
"""
Model Check Script for TAQA Hybrid System
Verifies all required files are present and loadable
"""

import os
import sys
import json
import joblib
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and get its size"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        size_mb = size / (1024 * 1024)
        print(f"✅ {description}: {filepath} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT FOUND")
        return False

def check_json_loadable(filepath, description):
    """Check if JSON file can be loaded"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ {description}: JSON loadable, {len(data)} root keys")
        return True
    except Exception as e:
        print(f"❌ {description}: JSON load error - {e}")
        return False

def check_joblib_loadable(filepath, description):
    """Check if joblib file can be loaded"""
    try:
        model = joblib.load(filepath)
        print(f"✅ {description}: Model loadable")
        return True
    except Exception as e:
        print(f"❌ {description}: Model load error - {e}")
        return False

def main():
    """Check all required files for TAQA Hybrid System"""
    print("🔍 TAQA HYBRID SYSTEM - MODEL CHECK")
    print("=" * 50)
    
    all_good = True
    
    # Required files for Hybrid System
    required_files = [
        {
            'path': 'taqa_priority_lookup.json',
            'description': 'Lookup System Data',
            'type': 'json'
        },
        {
            'path': 'improved_balanced_taqa_model.joblib',
            'description': 'ML Model (Balanced)',
            'type': 'joblib'
        },
        {
            'path': 'hybrid_taqa_api.py',
            'description': 'Hybrid API Code',
            'type': 'file'
        },
        {
            'path': 'app.py',
            'description': 'Flask Application',
            'type': 'file'
        },
        {
            'path': 'requirements.txt',
            'description': 'Dependencies',
            'type': 'file'
        },
        {
            'path': 'templates/index.html',
            'description': 'Web Interface',
            'type': 'file'
        }
    ]
    
    print("\n📁 FILE EXISTENCE CHECK:")
    print("-" * 30)
    
    for file_info in required_files:
        exists = check_file_exists(file_info['path'], file_info['description'])
        if not exists:
            all_good = False
    
    print("\n🧪 LOADING TEST:")
    print("-" * 20)
    
    # Test JSON loading
    if os.path.exists('taqa_priority_lookup.json'):
        lookup_ok = check_json_loadable('taqa_priority_lookup.json', 'Lookup System')
        if not lookup_ok:
            all_good = False
    
    # Test model loading
    if os.path.exists('improved_balanced_taqa_model.joblib'):
        model_ok = check_joblib_loadable('improved_balanced_taqa_model.joblib', 'ML Model')
        if not model_ok:
            all_good = False
    
    print("\n🔧 HYBRID SYSTEM TEST:")
    print("-" * 25)
    
    # Test hybrid system initialization
    try:
        from hybrid_taqa_api import HybridTAQAAPI
        api = HybridTAQAAPI()
        print("✅ Hybrid System: Successfully initialized")
        
        # Test prediction
        test_result = api.predict_single(
            description="Test maintenance",
            equipment_type="POMPE TEST",
            section="34MC"
        )
        print(f"✅ Test Prediction: {test_result['priority_score']} via {test_result['method']}")
        
    except Exception as e:
        print(f"❌ Hybrid System: Initialization error - {e}")
        all_good = False
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 ALL CHECKS PASSED - Ready for deployment!")
        print("✅ Lookup system working")
        print("✅ ML model working") 
        print("✅ Hybrid system working")
    else:
        print("❌ ISSUES FOUND - Fix before deployment!")
        print("\n💡 Common solutions:")
        print("   - Re-run: python improved_balanced_model.py")
        print("   - Check file upload size limits")
        print("   - Verify all dependencies in requirements.txt")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    exit(main()) 