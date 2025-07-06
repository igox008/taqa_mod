#!/usr/bin/env python3
"""
Web interface for Equipment Parameter Predictor
"""

from flask import Flask, request, render_template, jsonify
from equipment_anomaly_predictor_fast import FastParameterPredictor, SmartParameterPredictor
import pandas as pd
from datetime import datetime
import os
import joblib
import traceback
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the predictors
logger.info("Loading model components...")
try:
    fast_predictor = FastParameterPredictor()
    fast_predictor.load_model('models/parameter_predictor.joblib')
    smart_predictor = SmartParameterPredictor()
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Get available sections from CSV
try:
    data = pd.read_csv('data_set.csv')
    AVAILABLE_SECTIONS = sorted(data['Section propriétaire'].unique())
    
    # Filter out NaN values and empty strings before sorting equipment types
    equipment_series = data['Description de l\'équipement']
    AVAILABLE_EQUIPMENT = sorted(equipment_series[equipment_series.notna() & (equipment_series != '')].unique())
    
    logger.info(f"Loaded {len(AVAILABLE_SECTIONS)} sections and {len(AVAILABLE_EQUIPMENT)} equipment types")
except Exception as e:
    logger.error(f"Error loading categories: {str(e)}")
    logger.error(traceback.format_exc())
    raise

@app.route('/')
def home():
    return render_template('index.html', sections=AVAILABLE_SECTIONS, equipment_types=AVAILABLE_EQUIPMENT)  # Add equipment_types

def get_severity_level(total_score):
    """Determine severity level based on total score"""
    if total_score <= 7:
        return "MINEUR - Maintenance de routine"
    elif total_score <= 9:
        return "MODÉRÉ - Planifier intervention"
    else:
        return "CRITIQUE - Intervention immédiate requise"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logger.info(f"Received prediction request with data: {data}")
        
        # Validate input data
        required_fields = ['description', 'equipment_type', 'section']
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field: {field}")
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Validate text fields are not empty
        if not data['description'].strip():
            return jsonify({'error': 'Description cannot be empty'}), 400
        if not data['equipment_type'].strip():
            return jsonify({'error': 'Equipment type cannot be empty'}), 400
        if not data['section'].strip():
            return jsonify({'error': 'Section cannot be empty'}), 400
            
        # Make predictions with both models
        logger.debug("Making predictions with parameters:")
        logger.debug(f"Description: {data['description']}")
        logger.debug(f"Equipment: {data['equipment_type']}")
        logger.debug(f"Section: {data['section']}")
        
        # Get predictions from both models
        fast_result = fast_predictor.predict(
            description=data['description'],
            equipment_type=data['equipment_type'],
            section=data['section']
        )
        
        smart_result = smart_predictor.predict(
            description=data['description'],
            equipment_type=data['equipment_type'],
            section=data['section']
        )
        
        logger.info(f"Fast predictor result: {fast_result}")
        logger.info(f"Smart predictor result: {smart_result}")
        
        # Calculate criticality for both predictions (only numeric values)
        fast_criticality = fast_result['Fiabilité'] + fast_result['Disponibilité'] + fast_result['Process Safety']
        smart_criticality = smart_result['Fiabilité'] + smart_result['Disponibilité'] + smart_result['Process Safety']
        
        # Format response with new severity levels
        response = {
            'smart_predictor': {
                'predictions': {
                    'Fiabilité': smart_result['Fiabilité'],
                    'Disponibilité': smart_result['Disponibilité'],
                    'Process Safety': smart_result['Process Safety']
                },
                'criticality': smart_criticality,
                'severity_level': get_severity_level(smart_criticality)
            },
            'fast_predictor': {
                'predictions': {
                    'Fiabilité': fast_result['Fiabilité'],
                    'Disponibilité': fast_result['Disponibilité'],
                    'Process Safety': fast_result['Process Safety']
                },
                'criticality': fast_criticality,
                'severity_level': get_severity_level(fast_criticality)
            },
            'timestamp': datetime.now().isoformat()
        }
            
        logger.info(f"Sending response: {response}")
        return jsonify(response)
        
    except Exception as e:
        logger.error("Error in prediction:")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'An error occurred while processing the prediction',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 