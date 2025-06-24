import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import json
from datetime import datetime

def create_priority_rules(row):
    """Rule-based priority assignment for missing values"""
    desc = str(row.get('Description', '')).lower()
    
    # High priority indicators (4.0-5.0)
    high_priority_indicators = [
        'safety', 'sécurité', 'urgent', 'critique', 'arrêt', 'immédiat',
        'fuite', 'danger', 'panne', 'défaillance', 'grave', 'critical'
    ]
    
    # Low priority indicators (1.0-2.0)  
    low_priority_indicators = [
        'amélioration', 'optimisation', 'prévoir', 'planifier', 'modernisation'
    ]
    
    # Count indicators
    high_count = sum(indicator in desc for indicator in high_priority_indicators)
    low_count = sum(indicator in desc for indicator in low_priority_indicators)
    
    # Apply rules
    if high_count >= 2:
        return 4.5  # High priority
    elif high_count >= 1:
        return 3.8  # Medium-high priority  
    elif low_count >= 1:
        return 1.3  # Low priority
    else:
        return 2.2  # Default medium priority

def improved_text_features(text):
    """Enhanced text feature extraction"""
    if pd.isna(text) or text == '':
        return {
            'length': 0,
            'safety_keywords': 0,
            'urgent_keywords': 0,
            'maintenance_keywords': 0,
            'improvement_keywords': 0,
            'critical_keywords': 0
        }
    
    text_lower = text.lower()
    
    # Enhanced keyword categories
    safety_keywords = ['safety', 'sécurité', 'danger', 'risque', 'fuite']
    urgent_keywords = ['urgent', 'immédiat', 'critique', 'arrêt', 'panne', 'défaut']
    maintenance_keywords = ['prévoir', 'contrôle', 'vérifier', 'maintenance']
    improvement_keywords = ['amélioration', 'optimisation', 'modernisation']
    critical_keywords = ['critique', 'critical', 'grave', 'important']
    
    return {
        'length': len(text),
        'safety_keywords': sum(keyword in text_lower for keyword in safety_keywords),
        'urgent_keywords': sum(keyword in text_lower for keyword in urgent_keywords),
        'maintenance_keywords': sum(keyword in text_lower for keyword in maintenance_keywords),
        'improvement_keywords': sum(keyword in text_lower for keyword in improvement_keywords),
        'critical_keywords': sum(keyword in text_lower for keyword in critical_keywords)
    }

def fix_and_retrain_model():
    """Fix the data and retrain the model properly"""
    
    print("=== FIXING AND RETRAINING MODEL ===")
    
    # Load data
    df = pd.read_csv('data_set.csv')
    print(f"Original data: {len(df)} rows")
    print(f"Missing priorities: {df['Priorité'].isna().sum()}")
    
    # ACTUALLY fix missing priorities
    missing_mask = df['Priorité'].isna()
    print(f"Applying rules to {missing_mask.sum()} missing priorities...")
    
    df.loc[missing_mask, 'Priorité'] = df[missing_mask].apply(create_priority_rules, axis=1)
    
    print(f"After fixing - Missing priorities: {df['Priorité'].isna().sum()}")
    print("New priority distribution:")
    print(df['Priorité'].value_counts().sort_index())
    
    # Enhanced feature engineering
    df['Description_clean'] = df['Description'].fillna('').astype(str)
    
    # Extract text features
    text_features = df['Description_clean'].apply(improved_text_features)
    text_df = pd.DataFrame(text_features.tolist())
    
    for col in text_df.columns:
        df[f'text_{col}'] = text_df[col]
    
    # Date features
    df['Date de detection de l\'anomalie'] = pd.to_datetime(df['Date de detection de l\'anomalie'], errors='coerce')
    df['Year'] = df['Date de detection de l\'anomalie'].dt.year.fillna(2024)
    df['Month'] = df['Date de detection de l\'anomalie'].dt.month.fillna(6)
    df['DayOfWeek'] = df['Date de detection de l\'anomalie'].dt.dayofweek.fillna(1)
    
    # Equipment features
    equipment_categories = ['pump', 'motor', 'valve', 'turbine', 'generator', 'transformer', 'boiler', 'fan']
    df['Equipment_clean'] = df['Description equipement'].fillna('').astype(str)
    
    for category in equipment_categories:
        df[f'equipment_{category}'] = df['Equipment_clean'].str.lower().str.contains(category, na=False).astype(int)
    
    # Section features  
    df['Section proprietaire'] = df['Section proprietaire'].fillna('UNKNOWN')
    section_categories = ['mechanical', 'electrical', 'control', 'maintenance']
    section_keywords = {'mechanical': ['MC'], 'electrical': ['EL'], 'control': ['CT'], 'maintenance': ['MAINT']}
    
    for category, keywords in section_keywords.items():
        df[f'section_{category}'] = df['Section proprietaire'].str.contains('|'.join(keywords), na=False).astype(int)
    
    # Prepare features
    numerical_features = [
        'Year', 'Month', 'DayOfWeek',
        'text_length', 'text_safety_keywords', 'text_urgent_keywords',
        'text_maintenance_keywords', 'text_improvement_keywords', 'text_critical_keywords'
    ]
    
    equipment_features = [col for col in df.columns if col.startswith('equipment_')]
    section_features = [col for col in df.columns if col.startswith('section_')]
    
    all_numerical = numerical_features + equipment_features + section_features
    X_numerical = df[all_numerical].fillna(0)
    
    # TF-IDF for text
    tfidf = TfidfVectorizer(max_features=50, stop_words=None, ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_text = tfidf.fit_transform(df['Description_clean']).toarray()
    
    # Combine features
    X = np.hstack([X_numerical.values, X_text])
    y = df['Priorité'].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Priority range: {y.min():.2f} to {y.max():.2f}")
    print(f"Priority mean: {y.mean():.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=10, random_state=42, n_jobs=-1)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Test performance
    y_pred = pipeline.predict(X_test)
    y_pred = np.clip(y_pred, 1.0, 5.0)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R: {r2:.4f}")
    
    # Test predictions
    print(f"\n=== Testing Predictions ===")
    test_cases = [
        "SAFETY: Fuite importante de vapeur - arrêt immédiat requis",
        "urgent: défaut critique alternateur", 
        "Prévoir contrôle mesure de température",
        "Amélioration éclairage salle technique"
    ]
    
    for desc in test_cases:
        # Create test features
        test_text_features = improved_text_features(desc)
        test_numerical = [2024, 6, 1] + [test_text_features[f] for f in ['length', 'safety_keywords', 'urgent_keywords', 'maintenance_keywords', 'improvement_keywords', 'critical_keywords']]
        test_numerical += [0] * (len(equipment_features) + len(section_features))
        
        test_text = tfidf.transform([desc]).toarray()
        test_sample = np.hstack([test_numerical, test_text[0]]).reshape(1, -1)
        
        pred = pipeline.predict(test_sample)[0]
        pred = np.clip(pred, 1.0, 5.0)
        print(f"{desc[:50]}... -> {pred:.2f}")
    
    # Save corrected model
    feature_names = all_numerical + [f'tfidf_{i}' for i in range(X_text.shape[1])]
    
    joblib.dump({
        'pipeline': pipeline,
        'tfidf': tfidf,
        'feature_names': feature_names
    }, 'corrected_anomaly_priority_regressor.joblib')
    
    # Save model info
    model_info = {
        'model_name': 'Corrected Random Forest',
        'features_count': len(feature_names),
        'training_samples': len(df),
        'mae': mae,
        'mse': mse, 
        'r2': r2,
        'priority_range': [1.0, 5.0],
        'created_date': datetime.now().isoformat(),
        'missing_data_fixed': True,
        'priority_distribution': df['Priorité'].value_counts().sort_index().to_dict()
    }
    
    with open('corrected_regressor_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nCorrected model saved as: corrected_anomaly_priority_regressor.joblib")
    print(f"Model info saved as: corrected_regressor_model_info.json")

if __name__ == "__main__":
    fix_and_retrain_model()
