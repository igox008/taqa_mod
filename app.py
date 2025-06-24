# Azure deployment configuration for TAQA Anomaly Classifier
# Production-ready Flask app with lookup-based system

from flask import Flask, render_template, request, jsonify
import os
import sys
from pathlib import Path

app = Flask(__name__)

# Initialize the classifier API
classifier_api = None

def initialize_classifier():
    """Initialize the classifier with the lookup-based system"""
    global classifier_api
    
    try:
        # Use the new lookup-based system
        from taqa_lookup_api import TAQALookupAPI
        classifier_api = TAQALookupAPI()
        print("✅ TAQA Lookup-based priority system loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading lookup system: {e}")
        print("Make sure taqa_lookup_api.py and taqa_priority_lookup.json exist")
        return False

@app.route('/')
def index():
    """Main page with the anomaly classification form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions using lookup system"""
    try:
        data = request.get_json()
        
        if classifier_api is None:
            return jsonify({'error': 'Lookup system not loaded'}), 500
        
        # Extract only the relevant data (no status or date needed)
        description = data.get('description', '')
        equipment_type = data.get('equipment_type', '')
        section = data.get('section', '')
        
        # Make prediction using lookup system
        result = classifier_api.predict_single(
            description=description,
            equipment_type=equipment_type,
            section=section
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    if classifier_api is None:
        return jsonify({'status': 'unhealthy', 'error': 'Lookup system not loaded'}), 500
    
    return jsonify(classifier_api.health_check())

@app.route('/model_info')
def model_info():
    """Get lookup system information"""
    if classifier_api is None:
        return jsonify({'error': 'Lookup system not loaded'}), 500
    
    return jsonify(classifier_api.get_model_info())

@app.route('/equipment_types')
def equipment_types():
    """Get known equipment types for autocomplete"""
    if classifier_api is None:
        return jsonify({'error': 'Lookup system not loaded'}), 500
    
    try:
        # Get equipment types from lookup data
        equipment_list = list(classifier_api.lookup_data['equipment_priorities'].keys())
        equipment_list.sort()
        return jsonify({'equipment_types': equipment_list[:50]})  # Return top 50
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize classifier on startup
classifier_loaded = initialize_classifier()

# Azure App Service configuration
if __name__ == '__main__':
    if not classifier_loaded:
        print("❌ Failed to initialize lookup system.")
        sys.exit(1)
    
    # Production settings for Azure
    port = int(os.environ.get('PORT', 8000))  # Azure uses PORT env variable
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("🚀 TAQA Anomaly Classifier - Azure Production")
    print("📊 Lookup-based system with 82% accuracy")
    print(f"🌐 Starting on port {port}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=debug) 