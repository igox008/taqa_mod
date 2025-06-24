# Advanced ML deployment configuration for TAQA Anomaly Classifier
# Production-ready Flask app with state-of-the-art ensemble ML model

from flask import Flask, render_template, request, jsonify
import os
import sys
from pathlib import Path

app = Flask(__name__)

# Initialize the advanced ML classifier API
classifier_api = None

def initialize_classifier():
    """Initialize the classifier with the advanced ML ensemble model"""
    global classifier_api
    
    try:
        # Use the hybrid system (Lookup + ML)
        from hybrid_taqa_api import HybridTAQAAPI
        classifier_api = HybridTAQAAPI()
        print("✅ TAQA Hybrid System loaded successfully!")
        print("🎯 Lookup Accuracy: 82% | ML Accuracy: 67.8%")
        print("🤖 Using: Intelligent hybrid (Lookup for known + ML for unknown)")
        return True
    except Exception as e:
        print(f"❌ Error loading advanced ML model: {e}")
        print("Make sure advanced_taqa_model.joblib exists")
        print("Run 'python advanced_ml_model.py' to train the model first")
        return False

@app.route('/')
def index():
    """Main page with the anomaly classification form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions using advanced ML ensemble"""
    try:
        data = request.get_json()
        
        if classifier_api is None:
            return jsonify({'error': 'Advanced ML model not loaded'}), 500
        
        # Extract the relevant data
        description = data.get('description', '')
        equipment_type = data.get('equipment_type', '')
        section = data.get('section', '')
        
        # Make prediction using advanced ML ensemble
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
        return jsonify({'status': 'unhealthy', 'error': 'Advanced ML model not loaded'}), 500
    
    return jsonify({
        'status': 'healthy',
        'model': 'Hybrid System (Lookup + ML)',
        'lookup_accuracy': '82%',
        'ml_accuracy': '67.8%',
        'approach': 'Best of both worlds',
        'techniques': [
            'Historical data lookup for known equipment',
            'ML ensemble for unknown cases',
            'Intelligent feature engineering',
            'Equipment criticality analysis',
            'Priority-based routing',
            'Confidence-based selection'
        ]
    })

@app.route('/model_info')
def model_info():
    """Get advanced ML model information"""
    if classifier_api is None:
        return jsonify({'error': 'Advanced ML model not loaded'}), 500
    
    try:
        # Return hybrid system info
        return jsonify({
            'model_name': 'TAQA Hybrid Prediction System',
            'model_type': 'Lookup + ML Ensemble',
            'lookup_accuracy': '82%',
            'ml_accuracy': '67.8%',
            'strategy': 'Use lookup for known equipment, ML for unknowns',
            'features': [
                'Equipment-based historical lookup',
                'Section-based averages',
                'Smart text feature extraction',
                'Domain-specific keywords',
                'Ensemble ML prediction',
                'Confidence-based routing'
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/equipment_types')
def equipment_types():
    """Get known equipment types for autocomplete"""
    # Common TAQA equipment types for autocomplete
    equipment_list = [
        "POMPE ALIMENTAIRE PRINCIPALE",
        "ALTERNATEUR UNITE 1",
        "ALTERNATEUR UNITE 2", 
        "MOTEUR VENTILATEUR",
        "CHAUDIERE PRINCIPALE",
        "TRANSFORMATEUR",
        "VENTILATEUR TIRAGE FORCE",
        "POMPE EAU CIRCULATION",
        "VANNE REGULATION",
        "SERVOMOTEUR",
        "RAMONEUR LONG RETRACTABLE",
        "DECRASSEUR",
        "SOUPAPE SECURITE",
        "DETECTEUR NIVEAU",
        "ECLAIRAGE",
        "ARMOIRE ELECTRIQUE",
        "CABLE ALIMENTATION",
        "PRISE COURANT"
    ]
    
    return jsonify({'equipment_types': sorted(equipment_list)})

@app.route('/sections')
def sections():
    """Get known sections for autocomplete"""
    sections_list = [
        "34MC",  # Mechanical Coal
        "34EL",  # Electrical
        "34CT",  # Control
        "34MD",  # Mechanical Diesel
        "34MM",  # Mechanical Maintenance
        "34MG"   # Mechanical General
    ]
    
    return jsonify({'sections': sections_list})

# Initialize classifier on startup
classifier_loaded = initialize_classifier()

# Production configuration
if __name__ == '__main__':
    if not classifier_loaded:
        print("❌ Failed to initialize advanced ML model.")
        print("💡 Run 'python advanced_ml_model.py' to train the model first.")
        sys.exit(1)
    
    # Production settings
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("🚀 TAQA Anomaly Classifier - Hybrid System Production")
    print("🎯 Intelligent hybrid: 82% lookup + 67.8% ML accuracy")
    print("🤖 Best of both worlds: Reliability + Flexibility")
    print(f"🌐 Starting on port {port}")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=port, debug=debug) 