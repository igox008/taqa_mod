#!/usr/bin/env python3
"""
Test script to debug the advanced ML model predictions
"""

import pandas as pd
import numpy as np
from advanced_api_wrapper import AdvancedTAQAAPI

def test_advanced_model_detailed():
    """Test the advanced model with detailed analysis"""
    print("🔍 DETAILED TESTING OF ADVANCED ML MODEL")
    print("=" * 60)
    
    try:
        api = AdvancedTAQAAPI('advanced_taqa_model.joblib')
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test cases with very different expected priorities
    test_cases = [
        {
            'name': 'CRITICAL - Steam leak with immediate shutdown',
            'description': "URGENT SAFETY: Fuite massive de vapeur haute pression pompe alimentaire principale - arrêt immédiat requis danger personnel",
            'equipment': "POMPE ALIMENTAIRE PRINCIPALE",
            'section': "34MC",
            'expected_range': (3.5, 5.0)
        },
        {
            'name': 'HIGH - Equipment failure',
            'description': "Défaut critique alternateur unité 1 - vibrations anormales arrêt automatique protection",
            'equipment': "ALTERNATEUR UNITE 1",
            'section': "34EL",
            'expected_range': (3.0, 4.0)
        },
        {
            'name': 'MEDIUM - Routine maintenance',
            'description': "Maintenance préventive pompe circulation - révision programmée selon planning",
            'equipment': "POMPE CIRCULATION",
            'section': "34MM",
            'expected_range': (2.0, 3.0)
        },
        {
            'name': 'LOW - Minor improvement',
            'description': "Amélioration éclairage bureau technique - installation nouvelles LED",
            'equipment': "ECLAIRAGE BUREAU",
            'section': "34EL",
            'expected_range': (1.0, 2.0)
        }
    ]
    
    print(f"Testing {len(test_cases)} scenarios with detailed analysis...\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"🧪 TEST {i}: {test['name']}")
        print(f"📝 Description: {test['description']}")
        print(f"⚙️ Equipment: {test['equipment']}")
        print(f"🏢 Section: {test['section']}")
        print(f"🎯 Expected: {test['expected_range'][0]}-{test['expected_range'][1]}")
        print("-" * 50)
        
        # Get prediction
        result = api.predict_single(
            description=test['description'],
            equipment_type=test['equipment'],
            section=test['section']
        )
        
        score = result['priority_score']
        confidence = result['confidence']
        
        print(f"📊 PREDICTION: {score:.3f}")
        print(f"🎭 Method: {result['method']}")
        print(f"🔒 Confidence: {confidence:.3f}")
        print(f"💡 Explanation: {result['explanation']}")
        
        # Feature analysis
        if 'feature_analysis' in result:
            features = result['feature_analysis']
            print(f"🔍 Feature Analysis:")
            print(f"   Safety Score: {features['safety_score']:.3f}")
            print(f"   Urgency Score: {features['urgency_score']:.3f}")
            print(f"   Technical Score: {features['technical_score']:.3f}")
        
        # Model contributions
        if 'model_contributions' in result:
            print(f"🤖 Model Contributions:")
            for model_name, data in result['model_contributions'].items():
                print(f"   {model_name}: {data['prediction']:.3f} (weight: {data['weight']:.3f})")
        
        # Check if prediction is in expected range
        expected_min, expected_max = test['expected_range']
        is_correct = expected_min <= score <= expected_max
        
        if is_correct:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            print(f"⚠️ Expected: {expected_min}-{expected_max}, Got: {score:.3f}")
        
        print(f"🎯 Result: {status}")
        print("=" * 60)
        print()
    
    # Test specific text features
    print("🔤 TESTING TEXT FEATURE EXTRACTION")
    print("-" * 40)
    
    test_texts = [
        "URGENT SAFETY fuite vapeur danger arrêt immédiat",
        "maintenance préventive routine contrôle",
        "amélioration éclairage bureau"
    ]
    
    for text in test_texts:
        features = api.extract_advanced_text_features(text)
        print(f"Text: '{text}'")
        print(f"  Safety: {features['safety_score']:.3f}")
        print(f"  Urgency: {features['urgency_score']:.3f}")
        print(f"  Technical: {features['technical_score']:.3f}")
        print()

def check_model_data_distribution():
    """Check the training data distribution"""
    print("📊 CHECKING TRAINING DATA DISTRIBUTION")
    print("-" * 50)
    
    try:
        df = pd.read_csv('data_set.csv')
        df_clean = df.dropna(subset=['Priorité'])
        
        print(f"Total training records: {len(df_clean)}")
        print("\nPriority distribution:")
        priority_counts = df_clean['Priorité'].value_counts().sort_index()
        for priority, count in priority_counts.items():
            percentage = (count / len(df_clean)) * 100
            print(f"  Priority {priority}: {count:,} records ({percentage:.1f}%)")
        
        print(f"\nPriority statistics:")
        print(f"  Mean: {df_clean['Priorité'].mean():.3f}")
        print(f"  Median: {df_clean['Priorité'].median():.3f}")
        print(f"  Std: {df_clean['Priorité'].std():.3f}")
        print(f"  Min: {df_clean['Priorité'].min():.3f}")
        print(f"  Max: {df_clean['Priorité'].max():.3f}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")

if __name__ == "__main__":
    test_advanced_model_detailed()
    print()
    check_model_data_distribution() 