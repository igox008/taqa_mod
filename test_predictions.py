#!/usr/bin/env python3
"""
Test script to analyze predictions for different descriptions
"""

import pandas as pd
import joblib
from datetime import datetime
import numpy as np

# Load the model components
print("Loading model components...")
model_components = joblib.load('models/anomaly_predictor.joblib')
model = model_components['model']
feature_names = model_components['feature_names']
encoders = model_components['encoders']
tfidf_vectorizers = model_components['tfidf_vectorizers']

# Test cases
test_cases = [
    {
        'description': 'Fuite importante d\'huile avec vibrations anormales. Température très élevée et bruit critique.',
        'expected_severity': 'High',
        'keywords': ['fuite importante', 'vibrations', 'température', 'bruit critique']
    },
    {
        'description': 'Légère fuite d\'eau observée. Contrôle de routine nécessaire.',
        'expected_severity': 'Medium',
        'keywords': ['fuite', 'contrôle', 'routine']
    },
    {
        'description': 'Prévoir un nettoyage et graissage de routine.',
        'expected_severity': 'Low',
        'keywords': ['nettoyage', 'graissage', 'routine']
    },
    {
        'description': 'Rupture du joint principal avec fuite massive. Arrêt d\'urgence nécessaire.',
        'expected_severity': 'Critical',
        'keywords': ['rupture', 'fuite massive', 'urgence']
    },
    {
        'description': 'Usure normale des roulements. À surveiller.',
        'expected_severity': 'Low',
        'keywords': ['usure', 'normale', 'surveiller']
    }
]

def extract_text_features(description):
    """Extract text features from description"""
    description_col = pd.Series([description])
    text_features = pd.DataFrame()
    
    # Critical keywords with weights
    critical_keywords = {
        'leak_keywords': {
            'fuite': 3,
            'fuite importante': 4,
            'fuite d\'eau': 3,
            'fuite d\'huile': 3,
            'fuite de vapeur': 3,
            'fuite massive': 4
        },
        'vibration_keywords': {
            'vibration': 3,
            'bruit anormal': 3,
            'bruit': 2,
            'vibrant': 2
        },
        'seal_keywords': {
            'non étanche': 3,
            'non-étanche': 3,
            'étanche': 2,
            'étanchéité': 2,
            'joint': 2
        },
        'damage_keywords': {
            'percement': 4,
            'rupture': 4,
            'cassé': 3,
            'défaillance': 3,
            'panne': 3
        },
        'temperature_keywords': {
            'surchauffe': 4,
            'température': 2,
            'chauffage': 2,
            'refroidissement': 2
        },
        'pressure_keywords': {
            'pression': 2,
            'surpression': 4,
            'dépression': 3
        },
        'wear_keywords': {
            'usure': 2,
            'encrassement': 2,
            'corrosion': 3,
            'érosion': 3
        },
        'mechanical_keywords': {
            'coincement': 3,
            'blocage': 3,
            'grippé': 3,
            'difficile': 2
        }
    }
    
    # Create weighted binary flags for critical terms
    desc_lower = description_col.str.lower().iloc[0]
    keyword_scores = {}
    
    for category, keywords in critical_keywords.items():
        category_score = 0
        for keyword, weight in keywords.items():
            if keyword in desc_lower:
                category_score += weight
                col_name = f'has_{keyword.replace(" ", "_").replace("\'", "")}'
                text_features[col_name] = [1]
            else:
                col_name = f'has_{keyword.replace(" ", "_").replace("\'", "")}'
                text_features[col_name] = [0]
        keyword_scores[category] = category_score
    
    # Severity indicators with weights
    severity_patterns = {
        'severity_high': {
            'importante?': 4,
            'grave': 4,
            'urgent': 4,
            'critique': 4,
            'massive': 4,
            'arrêt': 4
        },
        'severity_medium': {
            'prévoir': 2,
            'contrôle': 2,
            'vérification': 2,
            'surveiller': 2
        },
        'severity_maintenance': {
            'nettoyage': 1,
            'entretien': 1,
            'graissage': 1,
            'remplacement': 2,
            'routine': 1,
            'normale': 1
        }
    }
    
    # Add severity scores
    severity_scores = {}
    for severity, patterns in severity_patterns.items():
        severity_score = 0
        for pattern, weight in patterns.items():
            if pattern in desc_lower:
                severity_score += weight
        text_features[severity] = [severity_score]
        severity_scores[severity] = severity_score
    
    # Text statistics
    text_features['description_length'] = [len(desc_lower)]
    text_features['word_count'] = [len(desc_lower.split())]
    text_features['capital_ratio'] = [sum(1 for c in description if c.isupper()) / len(description) if len(description) > 0 else 0]
    
    # TF-IDF features
    cleaned_description = description_col.str.lower()
    cleaned_description = cleaned_description.str.replace(r'[^\w\s]', ' ', regex=True)
    cleaned_description = cleaned_description.str.replace(r'\s+', ' ', regex=True)
    
    tfidf_features = tfidf_vectorizers['description'].transform(cleaned_description.fillna(''))
    tfidf_df = pd.DataFrame(
        tfidf_features.toarray(),
        columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
    )
    
    return pd.concat([text_features, tfidf_df], axis=1), keyword_scores, severity_scores

def predict_anomaly(description, equipment_type="Pompe centrifuge", section="Production"):
    """Make prediction using loaded model"""
    try:
        # Create feature dataframe
        new_data = pd.DataFrame({
            'Description': [description],
            'Description de l\'équipement': [equipment_type],
            'Section propriétaire': [section],
            'Fiabilité Intégrité': [3],
            'Disponibilté': [3],
            'Process Safety': [3],
            'Date de détéction de l\'anomalie': [datetime.now()]
        })
        
        # Extract features
        text_features, keyword_scores, severity_scores = extract_text_features(description)
        
        # Equipment features
        equipment_features = pd.DataFrame()
        
        # Handle equipment encoding
        if equipment_type in encoders['equipment'].classes_:
            equipment_features['equipment_type_encoded'] = [encoders['equipment'].transform([equipment_type])[0]]
        else:
            equipment_features['equipment_type_encoded'] = [0]
        
        # Handle section encoding
        if section in encoders['section'].classes_:
            equipment_features['section_encoded'] = [encoders['section'].transform([section])[0]]
        else:
            equipment_features['section_encoded'] = [0]
        
        # Default values for frequency and statistics
        equipment_features['equipment_frequency'] = [1]
        equipment_features['system_frequency'] = [1]
        equipment_features['equipment_avg_criticality'] = [6]
        equipment_features['equipment_std_criticality'] = [2]
        equipment_features['section_avg_criticality'] = [6]
        equipment_features['section_std_criticality'] = [2]
        
        # Combine all features
        new_features = pd.concat([
            new_data[['Fiabilité Intégrité', 'Disponibilté', 'Process Safety']],
            text_features,
            equipment_features
        ], axis=1)
        
        # Ensure correct feature order
        for col in feature_names:
            if col not in new_features.columns:
                new_features[col] = 0
        new_features = new_features[feature_names]
        
        # Make prediction
        criticality_pred = model.predict(new_features)[0]
        proba = model.predict_proba(new_features)[0]
        max_proba = proba[int(criticality_pred) - min(model.classes_)]
        
        return {
            'criticality': int(criticality_pred),
            'confidence': float(max_proba),
            'keyword_scores': keyword_scores,
            'severity_scores': severity_scores
        }
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise

def analyze_predictions():
    """Analyze predictions for all test cases"""
    print("\nAnalyzing predictions for test cases...")
    print("-" * 80)
    
    results = []
    for case in test_cases:
        print(f"\nTest Case: {case['expected_severity']}")
        print(f"Description: {case['description']}")
        print(f"Expected keywords: {', '.join(case['keywords'])}")
        
        result = predict_anomaly(case['description'])
        
        # Analyze keyword scores
        print("\nKeyword category scores:")
        for category, score in result['keyword_scores'].items():
            if score > 0:
                print(f"  {category}: {score}")
        
        # Analyze severity scores
        print("\nSeverity scores:")
        for severity, score in result['severity_scores'].items():
            if score > 0:
                print(f"  {severity}: {score}")
        
        print(f"\nPrediction: Criticality {result['criticality']} (Confidence: {result['confidence']:.2%})")
        
        results.append({
            'description': case['description'],
            'expected_severity': case['expected_severity'],
            'predicted_criticality': result['criticality'],
            'confidence': result['confidence'],
            'keyword_scores': result['keyword_scores'],
            'severity_scores': result['severity_scores']
        })
        
        print("-" * 80)
    
    return results

if __name__ == "__main__":
    results = analyze_predictions()
    
    # Print summary
    print("\nPrediction Summary:")
    print("-" * 80)
    for result in results:
        print(f"\nExpected Severity: {result['expected_severity']}")
        print(f"Predicted Criticality: {result['predicted_criticality']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        # Calculate total severity score
        total_severity = sum(result['severity_scores'].values())
        total_keywords = sum(result['keyword_scores'].values())
        print(f"Total Severity Score: {total_severity}")
        print(f"Total Keyword Score: {total_keywords}")
        print("-" * 40) 