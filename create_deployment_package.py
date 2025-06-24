#!/usr/bin/env python3
"""
Create Deployment Package for TAQA Hybrid System
Packages only essential files for DigitalOcean deployment
"""

import os
import shutil
import zipfile
from pathlib import Path

def create_deployment_package():
    """Create a clean deployment package"""
    
    # Essential files for deployment
    essential_files = [
        'app.py',
        'hybrid_taqa_api.py',
        'startup_check.py',
        'check_models.py',
        'requirements.txt',
        'Procfile',
        'runtime.txt',
        'taqa_priority_lookup.json',
        'improved_balanced_taqa_model.joblib'
    ]
    
    # Essential directories
    essential_dirs = [
        'templates/'
    ]
    
    # Create deployment directory
    deploy_dir = 'deployment_package'
    if os.path.exists(deploy_dir):
        shutil.rmtree(deploy_dir)
    os.makedirs(deploy_dir)
    
    print("📦 Creating deployment package...")
    
    # Copy essential files
    for file in essential_files:
        if os.path.exists(file):
            shutil.copy2(file, deploy_dir)
            size = os.path.getsize(file) / (1024 * 1024)
            print(f"✅ Copied: {file} ({size:.1f} MB)")
        else:
            print(f"⚠️ Missing: {file}")
    
    # Copy essential directories
    for dir_name in essential_dirs:
        if os.path.exists(dir_name):
            shutil.copytree(dir_name, os.path.join(deploy_dir, dir_name))
            print(f"✅ Copied directory: {dir_name}")
        else:
            print(f"⚠️ Missing directory: {dir_name}")
    
    # Create deployment zip
    zip_path = 'taqa_deployment.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(deploy_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, deploy_dir)
                zipf.write(file_path, arcname)
    
    print(f"\n📦 Deployment package created: {zip_path}")
    print(f"📊 Package size: {os.path.getsize(zip_path) / (1024 * 1024):.1f} MB")
    
    # Display deployment instructions
    print("\n🚀 DEPLOYMENT INSTRUCTIONS:")
    print("=" * 40)
    print("1. Upload taqa_deployment.zip to DigitalOcean")
    print("2. Extract the files")
    print("3. The Procfile will run startup_check.py first")
    print("4. If models are missing, fallbacks will be created")
    print("5. The app will start with gunicorn")
    
    print("\n🌐 Files in deployment package:")
    for root, dirs, files in os.walk(deploy_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, deploy_dir)
            size = os.path.getsize(file_path) / 1024
            print(f"   {rel_path} ({size:.1f} KB)")

def main():
    create_deployment_package()

if __name__ == "__main__":
    main() 