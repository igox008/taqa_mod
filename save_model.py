#!/usr/bin/env python3
"""
Train and save the Equipment Parameter Predictor model
"""

import pandas as pd
from equipment_anomaly_predictor_fast import FastParameterPredictor
import os

def main():
    print("Loading data...")
    data = pd.read_csv('data_set.csv')
    
    print("Initializing predictor...")
    predictor = FastParameterPredictor()
    
    print("Preparing data...")
    features, targets = predictor.prepare_data(data)
    
    print("Training model...")
    predictor.train(features, targets)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    print("Saving model...")
    model_path = 'models/parameter_predictor.joblib'
    predictor.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Test loading
    print("\nTesting saved model...")
    test_predictor = FastParameterPredictor()
    test_predictor.load_model(model_path)
    
    # Make a test prediction
    test_desc = "Fuite d'huile importante sur le joint de la pompe"
    test_equip = "PUMP"
    test_section = data['Section propriétaire'].iloc[0]
    
    print("\nMaking test prediction:")
    print(f"Description: {test_desc}")
    print(f"Equipment: {test_equip}")
    print(f"Section: {test_section}")
    
    result = test_predictor.predict(test_desc, test_equip, test_section)
    print("\nPrediction results:")
    for param, value in result.items():
        print(f"{param}: {value}/5")

if __name__ == '__main__':
    main() 