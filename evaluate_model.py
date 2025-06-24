import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
import joblib
import json

def evaluate_corrected_model():
    """Comprehensive evaluation of the corrected model"""
    
    print("=" * 60)
    print("TAQA CORRECTED MODEL - ACCURACY EVALUATION")
    print("=" * 60)
    
    # Load the corrected model
    try:
        model_data = joblib.load('corrected_anomaly_priority_regressor.joblib')
        pipeline = model_data['pipeline']
        print(" Corrected model loaded successfully")
    except:
        print(" Could not load corrected model")
        return
    
    # Load model info
    try:
        with open('corrected_regressor_model_info.json', 'r') as f:
            model_info = json.load(f)
        print(" Model info loaded")
    except:
        print("  Model info not found")
        model_info = {}
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    
    # Display stored metrics
    if 'mae' in model_info:
        print(f" Mean Absolute Error (MAE): {model_info['mae']:.4f}")
        print(f" Mean Squared Error (MSE): {model_info['mse']:.4f}")
        print(f" R Score: {model_info['r2']:.4f}")
        print(f" Training Samples: {model_info.get('training_samples', 'Unknown')}")
    
    # Accuracy interpretation
    mae = model_info.get('mae', 0)
    r2 = model_info.get('r2', 0)
    
    print(f"\n" + "=" * 60)
    print("ACCURACY INTERPRETATION")
    print("=" * 60)
    
    print(f" Average Prediction Error: {mae:.2f} priority points")
    print(f" Model Explains {r2*100:.1f}% of Priority Variance")
    
    if mae <= 0.2:
        accuracy_rating = "EXCELLENT"
        emoji = ""
    elif mae <= 0.3:
        accuracy_rating = "VERY GOOD" 
        emoji = ""
    elif mae <= 0.5:
        accuracy_rating = "GOOD"
        emoji = ""
    else:
        accuracy_rating = "NEEDS IMPROVEMENT"
        emoji = ""
    
    print(f"{emoji} Overall Accuracy Rating: {accuracy_rating}")
    
    # Priority distribution analysis
    if 'priority_distribution' in model_info:
        print(f"\n" + "=" * 60)
        print("TRAINING DATA DISTRIBUTION")
        print("=" * 60)
        
        distribution = model_info['priority_distribution']
        total_samples = sum(distribution.values())
        
        for priority, count in sorted(distribution.items()):
            percentage = (count / total_samples) * 100
            print(f"Priority {priority}: {count:,} samples ({percentage:.1f}%)")
    
    print(f"\n" + "=" * 60)
    print("REAL-WORLD PERFORMANCE TEST")
    print("=" * 60)
    
    # Test with realistic scenarios
    from corrected_api_wrapper import CorrectedAnomalyRegressorAPI
    
    api = CorrectedAnomalyRegressorAPI('corrected_anomaly_priority_regressor.joblib')
    
    test_scenarios = [
        {
            'description': 'SAFETY: Fuite importante de vapeur - arrêt immédiat requis',
            'equipment': 'TURBINE',
            'section': '34MC',
            'expected_range': [4.0, 5.0],
            'category': 'High Priority Safety'
        },
        {
            'description': 'urgent: défaut critique alternateur - risque panne majeure',
            'equipment': 'ALTERNATEUR',
            'section': '34EL',
            'expected_range': [3.5, 5.0],
            'category': 'High Priority Critical'
        },
        {
            'description': 'vibration anormale pompe alimentaire - surveillance nécessaire',
            'equipment': 'POMPE',
            'section': '34MC',
            'expected_range': [2.5, 4.0],
            'category': 'Medium-High Priority'
        },
        {
            'description': 'prévoir contrôle mesure de température moteur',
            'equipment': 'MOTEUR',
            'section': '34CT',
            'expected_range': [1.5, 3.0],
            'category': 'Medium Priority Maintenance'
        },
        {
            'description': 'amélioration éclairage salle technique - confort',
            'equipment': 'ÉCLAIRAGE',
            'section': '34EL',
            'expected_range': [1.0, 2.5],
            'category': 'Low Priority Improvement'
        },
        {
            'description': 'nettoyage équipements électriques secondaires',
            'equipment': 'ÉQUIPEMENTS ÉLECTRIQUES',
            'section': '34EL',
            'expected_range': [1.0, 2.5],
            'category': 'Low Priority Housekeeping'
        }
    ]
    
    correct_predictions = 0
    total_predictions = len(test_scenarios)
    
    for i, scenario in enumerate(test_scenarios, 1):
        result = api.predict_single(
            description=scenario['description'],
            equipment_type=scenario['equipment'],
            section=scenario['section']
        )
        
        predicted_score = result.get('priority_score', 0)
        expected_min, expected_max = scenario['expected_range']
        is_correct = expected_min <= predicted_score <= expected_max
        
        if is_correct:
            correct_predictions += 1
            status = ""
        else:
            status = ""
        
        print(f"{status} Test {i}: {predicted_score:.2f} (Expected: {expected_min}-{expected_max})")
        print(f"    {scenario['description'][:60]}...")
        print(f"     {scenario['category']}")
        print(f"    {result.get('explanation', 'No explanation')}")
        print()
    
    real_world_accuracy = (correct_predictions / total_predictions) * 100
    
    print("=" * 60)
    print("FINAL ACCURACY SUMMARY")
    print("=" * 60)
    
    print(f" Statistical Accuracy (MAE): {mae:.3f} points")
    print(f" Variance Explained (R): {r2*100:.1f}%")
    print(f" Real-World Scenario Accuracy: {correct_predictions}/{total_predictions} ({real_world_accuracy:.1f}%)")
    
    print(f"\n OVERALL MODEL RATING: {accuracy_rating}")
    
    if real_world_accuracy >= 80 and mae <= 0.25:
        print(" PRODUCTION READY: Model performs excellently!")
    elif real_world_accuracy >= 70 and mae <= 0.35:
        print(" GOOD FOR USE: Model performs well with minor improvements possible.")
    else:
        print("  NEEDS IMPROVEMENT: Consider retraining with more balanced data.")
    
    print(f"\n Evaluation completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    evaluate_corrected_model()
