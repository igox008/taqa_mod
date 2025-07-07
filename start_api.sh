#!/bin/bash

# Equipment Prediction API Startup Script
echo "🚀 Starting Equipment Prediction API..."

# Check if we're in the right directory
if [ ! -f "api_server.py" ]; then
    echo "❌ Error: api_server.py not found. Please run this script from the project directory."
    exit 1
fi

# Check if models exist
echo "📋 Checking for model files..."
models=("availability_model.pkl" "fiability_model.pkl" "process_safety_model.pkl")
missing_models=()

for model in "${models[@]}"; do
    if [ ! -f "$model" ]; then
        missing_models+=("$model")
    fi
done

if [ ${#missing_models[@]} -ne 0 ]; then
    echo "❌ Error: Missing model files:"
    printf '   - %s\n' "${missing_models[@]}"
    echo "Please ensure all model files are present before starting the API."
    exit 1
fi

echo "✅ All model files found"

# Check if virtual environment exists
if [ ! -d "api_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv api_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source api_env/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements_api.txt

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the API with gunicorn
echo "🌐 Starting API server with gunicorn..."
echo "📡 API will be available at: http://localhost:5000"
echo "📋 Endpoints:"
echo "   GET  /health      - Health check"
echo "   GET  /models/info - Model information"
echo "   POST /predict     - Make predictions"
echo ""
echo "🔍 Logs will be saved to logs/api.log"
echo "🛑 Press Ctrl+C to stop the server"
echo "=" * 50

# Start gunicorn
gunicorn -c gunicorn.conf.py wsgi:app \
    --access-logfile logs/api_access.log \
    --error-logfile logs/api_error.log \
    --capture-output 