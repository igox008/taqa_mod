#!/usr/bin/env python3
"""
Debug script to test the model predictions
"""

from api_wrapper import AnomalyRegressorAPI
import pandas as pd

def debug_model():
    """Debug the model predictions"""
    
    # Load the model
    try:
        api = AnomalyRegressorAPI('anomaly_priority_regressor_random_forest.joblib')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test cases with different priority levels
    test_cases = [
        {
            'description': 'SAFETY: Fuite importante de vapeur - arrêt immédiat requis',
            'equipment_type': 'TURBINE',
            'section': '34MC',
            'expected': 'High (4-5)'
        },
        {
            'description': 'urgent: défaut critique alternateur',
            'equipment_type': 'ALTERNATEUR',
            'section': '34EL',
            'expected': 'High (4-5)'
        },
        {
            'description': 'Prévoir contrôle mesure de température',
            'equipment_type': 'MOTEUR',
            'section': '34CT',
            'expected': 'Medium (2-3)'
        },
        {
            'description': 'Amélioration éclairage salle technique',
            'equipment_type': 'ECLAIRAGE',
            'section': '34EL',
            'expected': 'Low (1-2)'
        },
        {
            'description': 'Bruit anormal au niveau du moteur',
            'equipment_type': 'MOTEUR',
            'section': '34MC',
            'expected': 'Medium (2-3)'
        }
    ]
    
    print("\n=== Model Predictions ===")
    for i, test in enumerate(test_cases, 1):
        result = api.predict_single(
            description=test['description'],
            equipment_type=test['equipment_type'],
            section=test['section']
        )
        
        score = result.get('priority_score', 'Error')
        print(f"Test {i}: {score:.2f} (Expected: {test['expected']})")
        print(f"  Description: {test['description']}")
        print(f"  Equipment: {test['equipment_type']}")
        print()
    
    # Check the training data distribution
    print("=== Training Data Analysis ===")
    df = pd.read_csv('data_set.csv')
    print("Priority distribution in training data:")
    print(df['Priorité'].value_counts().sort_index())
    print(f"\nMean priority: {df['Priorité'].mean():.2f}")
    print(f"Standard deviation: {df['Priorité'].std():.2f}")
    
    # Check for missing values
    missing_priorities = df['Priorité'].isna().sum()
    print(f"Missing priorities: {missing_priorities}")
    
    # Look at some high priority examples
    high_priority = df[df['Priorité'] >= 4.0]
    print(f"\nHigh priority examples (>=4.0): {len(high_priority)}")
    if len(high_priority) > 0:
        print("Sample high priority descriptions:")
        for desc in high_priority['Description'].head(3):
            print(f"  - {desc}")

if __name__ == "__main__":
    debug_model() 