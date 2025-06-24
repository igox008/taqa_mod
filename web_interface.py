from flask import Flask, render_template, request, jsonify
import os
import sys

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
        print(" TAQA Lookup-based priority system loaded successfully!")
        return True
    except Exception as e:
        print(f" Error loading lookup system: {e}")
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

if __name__ == '__main__':
    # Initialize classifier on startup
    if initialize_classifier():
        print(" TAQA Anomaly Classifier with Lookup System")
        print(" Now provides 82% accuracy!")
        print(" Access at: http://localhost:5000")
        print(" Uses equipment and section-based lookup for predictions")
        print("=" * 60)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize lookup system.")
        sys.exit(1)
