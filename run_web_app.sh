#!/bin/bash

echo "🚀 Starting Equipment Prediction Web Application..."
echo "======================================================"

# Check if virtual environment exists
if [ ! -d "web_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv web_env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source web_env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the Flask application
echo "Starting web server..."
echo "🌐 Web application will be available at: http://localhost:5000"
echo "======================================================"
python app.py 