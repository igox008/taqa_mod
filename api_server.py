from flask import Flask, request, jsonify
import traceback
from comprehensive_prediction_system import ComprehensiveEquipmentPredictor

app = Flask(__name__)

# Initialize the predictor globally
predictor = None

def initialize_models():
    """Initialize all ML models"""
    global predictor
    try:
        print("Loading ML models...")
        predictor = ComprehensiveEquipmentPredictor()
        predictor.load_models()
        print("✅ All models loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        traceback.print_exc()
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": predictor is not None,
        "message": "Equipment Prediction API is running"
    })

@app.route('/predict', methods=['POST'])
def predict_anomaly():
    """
    Predict equipment anomaly scores
    
    Expected JSON payload:
    {
        "anomaly_id": "unique_id_for_this_anomaly",
        "description": "anomaly description text",
        "equipment_name": "equipment name/description", 
        "equipment_id": "equipment UUID"
    }
    """
    try:
        # Check if models are loaded
        if predictor is None:
            return jsonify({
                "error": "Models not loaded",
                "message": "ML models are not initialized"
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "message": "Please send JSON data in request body"
            }), 400
        
        # Extract required fields
        anomaly_id = data.get('anomaly_id')
        description = data.get('description')
        equipment_name = data.get('equipment_name')
        equipment_id = data.get('equipment_id')
        
        # Validate required fields
        missing_fields = []
        if not anomaly_id:
            missing_fields.append('anomaly_id')
        if not description:
            missing_fields.append('description')
        if not equipment_name:
            missing_fields.append('equipment_name')
        if not equipment_id:
            missing_fields.append('equipment_id')
        
        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields,
                "message": "Please provide all required fields"
            }), 400
        
        # Make predictions using the comprehensive system
        print(f"Making predictions for anomaly: {anomaly_id}")
        print(f"Description: {description[:100]}...")  # Log first 100 chars
        
        results = predictor.predict_all(
            description=description,
            equipment_name=equipment_name, 
            equipment_id=equipment_id
        )
        
        # Format response
        response = {
            "anomaly_id": anomaly_id,
            "equipment_id": equipment_id,
            "equipment_name": equipment_name,
            "predictions": {
                "availability": {
                    "score": round(results['availability_prediction'], 3),
                    "description": "Equipment uptime and operational readiness"
                },
                "reliability": {
                    "score": round(results['reliability_prediction'], 3), 
                    "description": "Equipment integrity and dependability"
                },
                "process_safety": {
                    "score": round(results['process_safety_prediction'], 3),
                    "description": "Safety risk assessment and hazard identification"
                }
            },
            "overall_score": round(results['overall_score'], 3),
            "total_sum": round(results['total_sum'], 3),
            "risk_assessment": {
                "overall_risk_level": results['overall_risk_level'],
                "priority_level": results['priority_level'],
                "requires_immediate_attention": results['requires_immediate_attention']
            },
            "status": "success"
        }
        
        print(f"✅ Predictions completed for anomaly: {anomaly_id}")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "Prediction failed", 
            "message": str(e),
            "status": "error"
        }), 500

@app.route('/models/info', methods=['GET'])
def models_info():
    """Get information about loaded models"""
    if predictor is None:
        return jsonify({
            "error": "Models not loaded",
            "models_loaded": False
        }), 500
    
    return jsonify({
        "models_loaded": True,
        "models": {
            "availability": {
                "description": "Predicts equipment uptime and operational readiness",
                "features": 23,
                "target_range": "1-5 (higher = better availability)"
            },
            "reliability": {
                "description": "Predicts equipment integrity and dependability", 
                "features": 23,
                "target_range": "1-5 (higher = better reliability)"
            },
            "process_safety": {
                "description": "Predicts safety risk assessment and hazard identification",
                "features": 29, 
                "target_range": "1-5 (higher = better safety)"
            }
        },
        "usage": {
            "endpoint": "/predict",
            "method": "POST",
            "required_fields": ["anomaly_id", "description", "equipment_name", "equipment_id"]
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "Available endpoints: /health, /predict, /models/info"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "message": "Check the HTTP method for this endpoint"
    }), 405

if __name__ == '__main__':
    print("🚀 Starting Equipment Prediction API...")
    print("="*50)
    
    # Initialize models on startup
    if initialize_models():
        print("\n🌐 Starting Flask server...")
        print("📡 API Endpoints:")
        print("  GET  /health - Health check")
        print("  GET  /models/info - Model information") 
        print("  POST /predict - Make predictions")
        print("\n📝 Example POST request to /predict:")
        print("""{
    "anomaly_id": "ANO-2024-001", 
    "description": "Fuite importante d'huile au niveau du palier",
    "equipment_name": "POMPE FUEL PRINCIPALE N°1",
    "equipment_id": "98b82203-7170-45bf-879e-f47ba6e12c86"
}""")
        print("="*50)
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Failed to initialize models. Exiting...")
        exit(1) 