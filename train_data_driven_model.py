import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import json
from datetime import datetime
import re

def extract_real_taqa_features(text):
    """Extract features based on ACTUAL TAQA patterns, not assumptions"""
    if pd.isna(text) or text == '':
        return {
            'length': 0,
            'leak_words': 0,
            'maintenance_words': 0,
            'equipment_words': 0,
            'technical_words': 0,
            'procurement_words': 0,
            'repair_words': 0
        }
    
    text_lower = text.lower()
    
    # Based on actual TAQA data analysis
    leak_words = ['fuite', 'étanche', 'étanchéité', 'presse étoupe']
    maintenance_words = ['prévoir', 'contrôle', 'maintenance', 'vérifier', 'nettoyage', 'révision']
    equipment_words = ['pompe', 'moteur', 'vanne', 'alternateur', 'ventilateur', 'turbine', 'chaudière']
    technical_words = ['température', 'pression', 'vibration', 'niveau', 'débit', 'bruit']
    procurement_words = ['achat', 'consommables', 'pose du', 'branchement', 'installation']
    repair_words = ['remise en état', 'réparation', 'remplacement', 'confection', 'soudage', 'révision']
    
    return {
        'length': len(text),
        'leak_words': sum(word in text_lower for word in leak_words),
        'maintenance_words': sum(word in text_lower for word in maintenance_words),
        'equipment_words': sum(word in text_lower for word in equipment_words),
        'technical_words': sum(word in text_lower for word in technical_words),
        'procurement_words': sum(word in text_lower for word in procurement_words),
        'repair_words': sum(word in text_lower for word in repair_words)
    }

def train_data_driven_model():
    """Train model based on actual TAQA data patterns"""
    
    print("=" * 60)
    print("TRAINING DATA-DRIVEN TAQA MODEL")
    print("Learning from actual patterns, not assumptions")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('data_set.csv')
    print(f"Total records: {len(df)}")
    
    # Only use records with actual priorities (no missing values)
    df_clean = df.dropna(subset=['Priorité', 'Description'])
    print(f"Records with priorities: {len(df_clean)}")
    
    print("Actual priority distribution:")
    print(df_clean['Priorité'].value_counts().sort_index())
    
    # Extract REAL features based on TAQA data
    df_clean['Description_clean'] = df_clean['Description'].fillna('').astype(str)
    
    # Extract text features based on actual patterns
    text_features = df_clean['Description_clean'].apply(extract_real_taqa_features)
    text_df = pd.DataFrame(text_features.tolist())
    
    for col in text_df.columns:
        df_clean[f'feature_{col}'] = text_df[col]
    
    # Date features
    df_clean['Date de detection de l\'anomalie'] = pd.to_datetime(df_clean['Date de detection de l\'anomalie'], errors='coerce')
    df_clean['Year'] = df_clean['Date de detection de l\'anomalie'].dt.year.fillna(2024)
    df_clean['Month'] = df_clean['Date de detection de l\'anomalie'].dt.month.fillna(6)
    df_clean['DayOfWeek'] = df_clean['Date de detection de l\'anomalie'].dt.dayofweek.fillna(1)
    
    # Equipment detection based on actual equipment names
    equipment_types = ['pompe', 'moteur', 'vanne', 'alternateur', 'ventilateur', 'turbine', 'chaudière', 'transformateur']
    df_clean['Equipment_clean'] = df_clean['Description equipement'].fillna('').astype(str).str.lower()
    
    for equip in equipment_types:
        df_clean[f'equipment_{equip}'] = df_clean['Equipment_clean'].str.contains(equip, na=False).astype(int)
    
    # Section features
    df_clean['Section proprietaire'] = df_clean['Section proprietaire'].fillna('UNKNOWN')
    sections = ['34MC', '34EL', '34CT', '34MD', '34MM', '34MG']
    for section in sections:
        df_clean[f'section_{section}'] = df_clean['Section proprietaire'].str.contains(section, na=False).astype(int)
    
    # Prepare features
    feature_cols = [
        'Year', 'Month', 'DayOfWeek',
        'feature_length', 'feature_leak_words', 'feature_maintenance_words',
        'feature_equipment_words', 'feature_technical_words', 'feature_procurement_words', 'feature_repair_words'
    ]
    
    # Add equipment and section features
    feature_cols += [col for col in df_clean.columns if col.startswith(('equipment_', 'section_'))]
    
    X_numerical = df_clean[feature_cols].fillna(0)
    
    # TF-IDF with more conservative parameters
    tfidf = TfidfVectorizer(
        max_features=200,
        ngram_range=(1, 3),  # Include 3-grams for better French phrase detection
        min_df=5,  # Must appear in at least 5 documents
        max_df=0.8,  # Ignore very common terms
        lowercase=True
    )
    
    X_text = tfidf.fit_transform(df_clean['Description_clean']).toarray()
    
    # Combine features
    X = np.hstack([X_numerical.values, X_text])
    y = df_clean['Priorité'].values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Priority range: {y.min():.1f} to {y.max():.1f}")
    print(f"Priority mean: {y.mean():.2f}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, 
        stratify=pd.cut(y, bins=4, labels=False)  # Stratify by priority ranges
    )
    
    # Train Random Forest with better parameters for regression
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])
    
    # Fit model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    # Clip to valid range
    y_pred_train = np.clip(y_pred_train, 1.0, 4.0)
    y_pred_test = np.clip(y_pred_test, 1.0, 4.0)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\nModel Performance:")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test R: {test_r2:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    print(f"CV MAE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Test on actual examples from the training data
    print(f"\n=== TESTING ON REAL TAQA EXAMPLES ===")
    
    # Get some examples from each priority level
    test_examples = []
    for priority in [1.0, 2.0, 3.0, 4.0]:
        examples = df_clean[df_clean['Priorité'] == priority].sample(min(2, len(df_clean[df_clean['Priorité'] == priority])))
        for _, row in examples.iterrows():
            test_examples.append((row['Description'], row['Description equipement'] or '', priority))
    
    for desc, equip, actual_priority in test_examples:
        # Create test features
        test_text_features = extract_real_taqa_features(desc)
        test_numerical = [2024, 6, 1] + [test_text_features[f] for f in ['length', 'leak_words', 'maintenance_words', 'equipment_words', 'technical_words', 'procurement_words', 'repair_words']]
        test_numerical += [0] * (len([col for col in df_clean.columns if col.startswith(('equipment_', 'section_'))]))
        
        test_text = tfidf.transform([desc]).toarray()
        test_sample = np.hstack([test_numerical, test_text[0]]).reshape(1, -1)
        
        pred = pipeline.predict(test_sample)[0]
        pred = np.clip(pred, 1.0, 4.0)
        
        error = abs(pred - actual_priority)
        status = "" if error <= 0.5 else ""
        
        print(f"{status} Predicted: {pred:.2f} | Actual: {actual_priority} | Error: {error:.2f}")
        print(f"   Description: {desc[:80]}...")
        print()
    
    # Save the data-driven model
    feature_names = feature_cols + [f'tfidf_{i}' for i in range(X_text.shape[1])]
    
    joblib.dump({
        'pipeline': pipeline,
        'tfidf': tfidf,
        'feature_names': feature_names,
        'feature_extractors': extract_real_taqa_features
    }, 'data_driven_taqa_regressor.joblib')
    
    # Save model info
    model_info = {
        'model_name': 'Data-Driven TAQA Regressor',
        'approach': 'Learned from actual TAQA patterns',
        'training_samples': len(df_clean),
        'test_mae': test_mae,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'cv_mae': -cv_scores.mean(),
        'priority_range': [1.0, 4.0],
        'actual_priority_distribution': df_clean['Priorité'].value_counts().sort_index().to_dict(),
        'created_date': datetime.now().isoformat()
    }
    
    with open('data_driven_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nData-driven model saved as: data_driven_taqa_regressor.joblib")
    print(f"This model learns TAQA actual patterns, not external assumptions!")

if __name__ == "__main__":
    train_data_driven_model()
