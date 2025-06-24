#!/usr/bin/env python3
"""
Deployment script for TAQA Anomaly Priority Classifier
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'scikit-learn', 'joblib', 'flask'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def check_dataset():
    """Check if the dataset file exists"""
    print("\n📊 Checking dataset...")
    
    if not os.path.exists('data_set.csv'):
        print("❌ data_set.csv not found!")
        print("Please ensure the dataset file is in the current directory.")
        return False
    
    print("✅ Dataset found")
    return True

def train_model():
    """Train the anomaly classification model"""
    print("\n🤖 Training model...")
    
    try:
        # Run the anomaly classifier
        result = subprocess.run([sys.executable, 'anomaly_classifier.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Model training completed successfully!")
            return True
        else:
            print("❌ Model training failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        return False

def check_model_files():
    """Check if model files were created"""
    print("\n🔍 Checking model files...")
    
    model_files = list(Path('.').glob('anomaly_priority_classifier_*.joblib'))
    
    if not model_files:
        print("❌ No model files found!")
        return False
    
    print(f"✅ Found {len(model_files)} model file(s):")
    for model_file in model_files:
        print(f"   - {model_file.name}")
    
    return True

def start_web_interface():
    """Start the web interface"""
    print("\n🌐 Starting web interface...")
    print("The web interface will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([sys.executable, 'web_interface.py'])
    except KeyboardInterrupt:
        print("\n👋 Web interface stopped")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")

def main():
    """Main deployment function"""
    print("🚀 TAQA Anomaly Priority Classifier - Deployment")
    print("=" * 50)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed. Exiting.")
        return
    
    # Step 2: Check dataset
    if not check_dataset():
        print("❌ Dataset check failed. Exiting.")
        return
    
    # Step 3: Train model (if not already trained)
    if not check_model_files():
        print("\n📈 No trained model found. Training new model...")
        if not train_model():
            print("❌ Model training failed. Exiting.")
            return
    else:
        print("✅ Trained model found. Skipping training.")
    
    # Step 4: Verify model files
    if not check_model_files():
        print("❌ Model verification failed. Exiting.")
        return
    
    # Step 5: Start web interface
    print("\n🎉 Deployment completed successfully!")
    print("\nOptions:")
    print("1. Start web interface (recommended)")
    print("2. Exit")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == '1':
        start_web_interface()
    else:
        print("👋 Goodbye!")

if __name__ == "__main__":
    main() 