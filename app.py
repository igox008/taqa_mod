from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
from comprehensive_prediction_system import ComprehensiveEquipmentPredictor
import traceback

app = Flask(__name__)

# Global predictor instance
predictor = None

def initialize_models():
    """Initialize and load all ML models"""
    global predictor
    try:
        print("Initializing ML models...")
        predictor = ComprehensiveEquipmentPredictor()
        
        # Try to load existing models
        if not predictor.load_models():
            print("Training models from scratch...")
            if not predictor.train_all_models():
                raise Exception("Failed to train models")
        
        print("✅ All models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        traceback.print_exc()
        return False

def load_equipment_data():
    """Load equipment data for dropdown menu"""
    try:
        # Load equipment data from all three CSV files
        equipment_data = []
        
        # Load availability equipment data
        avail_df = pd.read_csv('equipment_simple.csv')
        for _, row in avail_df.iterrows():
            # Handle missing/NaN descriptions
            description = row['Equipment_Description']
            if pd.isna(description) or description is None:
                description = f"Equipment {row['Equipment_ID']}"
            else:
                description = str(description).strip()
                
            # Truncate long descriptions
            if len(description) > 100:
                display_name = description[:100] + "..."
            else:
                display_name = description
                
            equipment_data.append({
                'id': str(row['Equipment_ID']),
                'name': display_name,
                'full_name': description,
                'avg_availability': float(row['Average_Score']) if not pd.isna(row['Average_Score']) else 0.0
            })
        
        # Sort by equipment name for better UX
        equipment_data.sort(key=lambda x: x['name'])
        
        return equipment_data
        
    except Exception as e:
        print(f"Error loading equipment data: {e}")
        import traceback
        traceback.print_exc()
        return []

@app.route('/')
def index():
    """Main page with prediction form"""
    equipment_data = load_equipment_data()
    return render_template('index.html', equipment_list=equipment_data)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        data = request.get_json()
        description = data.get('description', '').strip()
        equipment_id = data.get('equipment_id', '').strip()
        equipment_name = data.get('equipment_name', '').strip()
        
        # Validate input
        if not description:
            return jsonify({
                'error': True,
                'message': 'Please provide an anomaly description'
            })
        
        if not equipment_id:
            return jsonify({
                'error': True,
                'message': 'Please select an equipment'
            })
        
        # Make prediction using comprehensive predictor
        if predictor is None:
            return jsonify({
                'error': True,
                'message': 'Models not loaded. Please restart the application.'
            })
        
        # Get comprehensive prediction
        results = predictor.predict_all(description, equipment_name, equipment_id)
        
        # Format response
        response = {
            'error': False,
            'predictions': {
                'availability': {
                    'score': results['predictions']['availability'],
                    'level': get_score_level(results['predictions']['availability']),
                    'color': get_score_color(results['predictions']['availability'])
                },
                'fiability': {
                    'score': results['predictions']['fiability'],
                    'level': get_score_level(results['predictions']['fiability']),
                    'color': get_score_color(results['predictions']['fiability'])
                },
                'process_safety': {
                    'score': results['predictions']['process_safety'],
                    'level': get_score_level(results['predictions']['process_safety']),
                    'color': get_score_color(results['predictions']['process_safety'])
                },
                'overall': {
                    'score': results['predictions']['overall_score'],
                    'level': get_score_level(results['predictions']['overall_score']),
                    'color': get_score_color(results['predictions']['overall_score'])
                }
            },
            'risk_assessment': results['risk_assessment'],
            'detailed_analysis': results['detailed_analysis'],
            'recommendations': results['maintenance_recommendations']
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return jsonify({
            'error': True,
            'message': f'Prediction error: {str(e)}'
        })

@app.route('/equipment/<equipment_id>')
def get_equipment_info(equipment_id):
    """Get equipment information by ID"""
    try:
        # Load equipment data and find the requested equipment
        equipment_data = load_equipment_data()
        
        for equipment in equipment_data:
            if equipment['id'] == equipment_id:
                return jsonify({
                    'error': False,
                    'equipment': equipment
                })
        
        return jsonify({
            'error': True,
            'message': 'Equipment not found'
        })
        
    except Exception as e:
        return jsonify({
            'error': True,
            'message': f'Error: {str(e)}'
        })

def get_score_level(score):
    """Convert numeric score to descriptive level"""
    if score >= 4.0:
        return "Excellent"
    elif score >= 3.5:
        return "Good"
    elif score >= 3.0:
        return "Fair"
    elif score >= 2.0:
        return "Poor"
    else:
        return "Critical"

def get_score_color(score):
    """Get color code for score visualization"""
    if score >= 4.0:
        return "success"  # Green
    elif score >= 3.5:
        return "info"     # Light blue
    elif score >= 3.0:
        return "warning"  # Yellow
    elif score >= 2.0:
        return "danger"   # Orange/Red
    else:
        return "dark"     # Dark red

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': predictor is not None
    })

if __name__ == '__main__':
    print("🚀 Starting Equipment Prediction Web Application...")
    print("=" * 60)
    
    # Initialize models
    if initialize_models():
        print("🌐 Starting web server...")
        print("📋 Available at: http://localhost:5000")
        print("=" * 60)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize models. Cannot start web application.") 