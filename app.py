from flask import Flask, request, jsonify
import traceback
import time
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
        print(f"❌ App will start but predictions will not work")
        traceback.print_exc()
        # Don't exit in production - let the app start but return errors
        predictor = None
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    models_loaded = predictor is not None
    status = "healthy" if models_loaded else "degraded"
    message = "Equipment Prediction API is running"
    
    if not models_loaded:
        message = "API is running but ML models failed to load"
    
    return jsonify({
        "status": status,
        "models_loaded": models_loaded,
        "message": message,
        "app_name": "Equipment Prediction API",
        "version": "1.0.0"
    })

def validate_anomaly_data(anomaly_data, index=None):
    """Validate a single anomaly data object"""
    if not isinstance(anomaly_data, dict):
        return f"Anomaly{f' at index {index}' if index is not None else ''} must be an object"
    
    required_fields = ['anomaly_id', 'description', 'equipment_name', 'equipment_id']
    missing_fields = []
    
    for field in required_fields:
        if not anomaly_data.get(field):
            missing_fields.append(field)
    
    if missing_fields:
        return f"Missing required fields{f' at index {index}' if index is not None else ''}: {missing_fields}"
    
    return None

def process_single_anomaly(anomaly_data):
    """Process a single anomaly and return results"""
    try:
        results = predictor.predict_all(
            description=anomaly_data['description'],
            equipment_name=anomaly_data['equipment_name'], 
            equipment_id=anomaly_data['equipment_id']
        )
        
        return {
            "anomaly_id": anomaly_data['anomaly_id'],
            "equipment_id": anomaly_data['equipment_id'],
            "equipment_name": anomaly_data['equipment_name'],
            "predictions": {
                "availability": {
                    "score": round(results['predictions']['availability'], 3),
                    "description": "Equipment uptime and operational readiness"
                },
                "reliability": {
                    "score": round(results['predictions']['fiability'], 3), 
                    "description": "Equipment integrity and dependability"
                },
                "process_safety": {
                    "score": round(results['predictions']['process_safety'], 3),
                    "description": "Safety risk assessment and hazard identification"
                }
            },
            "overall_score": round(results['predictions']['overall_score'], 3),
            "risk_assessment": {
                "overall_risk_level": results['risk_assessment']['overall_risk_level'],
                "recommended_action": results['risk_assessment']['recommended_action'],
                "critical_factors": results['risk_assessment']['critical_factors'],
                "weakest_aspect": results['risk_assessment']['weakest_aspect']
            },
            "maintenance_recommendations": results['maintenance_recommendations'],
            "status": "success"
        }
    except Exception as e:
        return {
            "anomaly_id": anomaly_data.get('anomaly_id', 'unknown'),
            "equipment_id": anomaly_data.get('equipment_id', 'unknown'),
            "equipment_name": anomaly_data.get('equipment_name', 'unknown'),
            "status": "error",
            "error": str(e)
        }

@app.route('/predict', methods=['POST'])
def predict_anomaly():
    """
    Predict equipment anomaly scores for single or multiple anomalies
    
    Expected JSON payload for single anomaly:
    {
        "anomaly_id": "unique_id_for_this_anomaly",
        "description": "anomaly description text",
        "equipment_name": "equipment name/description", 
        "equipment_id": "equipment UUID"
    }
    
    Expected JSON payload for multiple anomalies:
    [
        {
            "anomaly_id": "anomaly_1",
            "description": "first anomaly description",
            "equipment_name": "equipment 1",
            "equipment_id": "uuid_1"
        },
        {
            "anomaly_id": "anomaly_2", 
            "description": "second anomaly description",
            "equipment_name": "equipment 2",
            "equipment_id": "uuid_2"
        }
        // ... up to 6000 anomalies
    ]
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
        
        start_time = time.time()
        
        # Determine if we have single anomaly or multiple anomalies
        if isinstance(data, list):
            # Multiple anomalies (batch processing)
            print(f"🔄 Processing batch of {len(data)} anomalies...")
            
            if len(data) == 0:
                return jsonify({
                    "error": "Empty anomaly list",
                    "message": "Please provide at least one anomaly"
                }), 400
            
            if len(data) > 6000:
                return jsonify({
                    "error": "Too many anomalies",
                    "message": f"Maximum 6000 anomalies allowed, got {len(data)}"
                }), 400
            
            # Validate all anomalies first
            validation_errors = []
            for i, anomaly in enumerate(data):
                error = validate_anomaly_data(anomaly, i)
                if error:
                    validation_errors.append(error)
            
            if validation_errors:
                return jsonify({
                    "error": "Validation failed",
                    "validation_errors": validation_errors,
                    "message": "Please fix the validation errors"
                }), 400
            
            # Process all anomalies
            results = []
            successful_predictions = 0
            failed_predictions = 0
            
            for i, anomaly in enumerate(data):
                if i % 100 == 0:  # Progress logging every 100 items
                    print(f"📊 Progress: {i}/{len(data)} anomalies processed")
                
                result = process_single_anomaly(anomaly)
                results.append(result)
                
                if result['status'] == 'success':
                    successful_predictions += 1
                else:
                    failed_predictions += 1
            
            processing_time = round(time.time() - start_time, 2)
            
            response = {
                "batch_info": {
                    "total_anomalies": len(data),
                    "successful_predictions": successful_predictions,
                    "failed_predictions": failed_predictions,
                    "processing_time_seconds": processing_time,
                    "average_time_per_anomaly": round(processing_time / len(data), 3)
                },
                "results": results,
                "status": "completed"
            }
            
            print(f"✅ Batch processing completed: {successful_predictions} successful, {failed_predictions} failed in {processing_time}s")
            return jsonify(response)
            
        else:
            # Single anomaly (backward compatibility)
            print(f"🔄 Processing single anomaly...")
            
            # Validate single anomaly
            error = validate_anomaly_data(data)
            if error:
                return jsonify({
                    "error": "Validation failed",
                    "message": error
                }), 400
            
            # Process single anomaly
            print(f"Making predictions for anomaly: {data['anomaly_id']}")
            print(f"Description: {data['description'][:100]}...")  # Log first 100 chars
            
            result = process_single_anomaly(data)
            processing_time = round(time.time() - start_time, 3)
            result["processing_time_seconds"] = processing_time
            
            if result['status'] == 'success':
                print(f"✅ Predictions completed for anomaly: {data['anomaly_id']} in {processing_time}s")
            else:
                print(f"❌ Prediction failed for anomaly: {data['anomaly_id']}")
            
            return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "Prediction failed", 
            "message": str(e),
            "status": "error"
        }), 500

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check system status"""
    import os
    import sys
    
    # Check file existence
    required_files = [
        'availability_model.pkl',
        'fiability_model.pkl', 
        'process_safety_model.pkl',
        'equipment_simple.csv',
        'severe_words_simple.csv'
    ]
    
    file_status = {}
    for file in required_files:
        file_status[file] = {
            "exists": os.path.exists(file),
            "size_mb": round(os.path.getsize(file) / 1024 / 1024, 2) if os.path.exists(file) else 0
        }
    
    return jsonify({
        "predictor_loaded": predictor is not None,
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "files": file_status,
        "environment": dict(os.environ)
    })

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
            "supports": "Single anomaly object OR array of anomaly objects (max 6000)",
            "required_fields": ["anomaly_id", "description", "equipment_name", "equipment_id"],
            "batch_processing": "Automatically detects single vs multiple anomalies"
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

# Initialize models when module loads (for production deployment)
print("🚀 Starting Equipment Prediction API...")
print("🔄 Initializing ML models...")
initialize_models()

if __name__ == '__main__':
    # Development server only
    print("🌐 Starting development server...")
    print("📡 API Endpoints:")
    print("  GET  /health - Health check")
    print("  GET  /models/info - Model information") 
    print("  POST /predict - Make predictions (single or batch)")
    app.run(host='0.0.0.0', port=5000, debug=True) 