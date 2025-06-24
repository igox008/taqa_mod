#!/bin/bash

# Azure App Service startup script for TAQA Anomaly Classifier
echo "🚀 Starting TAQA Anomaly Classifier on Azure"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Verify critical files exist
echo "🔍 Verifying deployment files..."
if [ ! -f "taqa_lookup_api.py" ]; then
    echo "❌ Missing taqa_lookup_api.py"
    exit 1
fi

if [ ! -f "taqa_priority_lookup.json" ]; then
    echo "❌ Missing taqa_priority_lookup.json"
    exit 1
fi

if [ ! -f "templates/index.html" ]; then
    echo "❌ Missing templates/index.html"
    exit 1
fi

echo "✅ All required files found"

# Start the application
echo "🌐 Starting Flask application..."
python app.py 