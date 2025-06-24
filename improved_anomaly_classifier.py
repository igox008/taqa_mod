#!/usr/bin/env python3
"""
Improved TAQA Anomaly Priority Regressor
Addresses the issues with data imbalance and missing values
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import json
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def improved_text_features(text):
    """Enhanced text feature extraction with better French keyword detection"""
    if pd.isna(text) or text == '':
        return {
            'length': 0,
            'safety_keywords': 0,
            'urgent_keywords': 0,
            'maintenance_keywords': 0,
            'improvement_keywords': 0,
            'technical_keywords': 0,
            'critical_keywords': 0
        }
    
    text_lower = text.lower()
    
    # Enhanced keyword categories
    safety_keywords = [
    'safety', 'sécurité', 'danger', 'risque', 'fuite', 'explosion', 'incendie', 'brûlure',
    'électrocution', 'choc électrique', 'intoxication', 'exposition', 'asphyxie', 'évanouissement',
    'urgence', 'toxicité', 'contact électrique', 'zone dangereuse', 'alarme', 'détection gaz',
    'masque', 'protection', 'EPI', 'chaleur excessive', 'inflammable', 'brûlure chimique',
    'déversement', 'dégazage', 'étouffement', 'bruit excessif', 'radiation', 'courant de fuite',
    'déflagration', 'surchauffe', 'sol glissant', 'machine non protégée', 'signalement sécurité',
    'incident grave', 'verrouillage sécurité', 'capteur gaz', 'fuite huile', 'gants de sécurité',
    'risque de chute', 'danger mécanique', 'objet tranchant', 'fumée', 'incapacité', 'étincelle',
    'protocole sécurité', 'non-conformité']
    
    urgent_keywords = [
    'urgent', 'immédiat', 'critique', 'arrêt', 'panne', 'défaut', 'défaillance',
    'hors service', 'blocage', 'urgence maintenance', 'anomalie grave', 'alerte rouge',
    'incident critique', 'interruption', 'erreur fatale', 'urgence technique', 'coupure',
    'intervention immédiate', 'shutdown', 'crash', 'dysfonctionnement', 'urgence totale',
    'perte fonctionnelle', 'système à l’arrêt', 'retard critique', 'risque immédiat',
    'priorité haute', 'alerte immédiate', 'urgence de sécurité', 'redémarrage urgent',
    'arrêt brutal', 'plantage', 'fail-stop', 'urgence électrique', 'dérèglement sévère',
    'bruit anormal', 'instabilité critique', 'arrêt imprévu', 'alarme critique',
    'court-circuit', 'fuite massive', 'fonction indisponible', 'problème majeur',
    'urgence incendie', 'interruption critique', 'panne générale', 'incident majeur',
    'péril imminent', 'urgence gaz']

    maintenance_keywords = [
        'prévoir', 'contrôle', 'vérifier', 'surveiller', 'maintenance',
        'inspection', 'entretien', 'diagnostic', 'suivi', 'calibrer',
        'nettoyage', 'graissage', 'programmation', 'vidange', 'révision',
        'audit', 'étalonnage', 'remplacement', 'routine', 'usure',
        'contrôle périodique', 'test de bon fonctionnement', 'correction', 'remise en état',
        'planification', 'maintenance préventive', 'maintenance corrective', 'plan de maintenance',
        'fiabilité', 'détection', 'analyse de panne', 'maintenance planifiée',
        'surveillance continue', 'intervention planifiée', 'réajustement', 'réalignement',
        'mise à jour technique', 'étude vibratoire', 'test pression', 'inspection visuelle',
        'lubrification', 'vidange huile', 'intervention mineure', 'changement filtre',
        'vérification capteurs', 'programme d entretien', 'conformité', 'registre maintenance',
        'rapport entretien']
    
    improvement_keywords = [
        'amélioration', 'optimisation', 'modernisation', 'upgrade',
        'fiabilisation', 'efficacité', 'performance', 'productivité', 'réduction des pertes',
        'gain énergétique', 'automatisation', 'digitalisation', 'rénovation', 'innovation',
        'réorganisation', 'refonte', 'réglage optimal', 'renforcement', 'rationalisation',
        'amélioration continue', 'augmentation capacité', 'renouvellement', 'augmentation rendement',
        'sécurisation', 'nouvelle technologie', 'reconfiguration', 'mise à niveau',
        'recalibrage', 'refroidissement amélioré', 'meilleure gestion', 'modification bénéfique',
        'efficience accrue', 'temps de réponse amélioré', 'réduction bruit', 'système intelligent',
        'pilotage avancé', 'surveillance augmentée', 'machine modernisée', 'ergonomie améliorée',
        'renouvellement équipement', 'design amélioré', 'gain de fiabilité', 'optimisation flux',
        'réduction émission', 'diminution consommation', 'allègement procédure',
        'traitement intelligent', 'adaptation au besoin', 'diagnostic avancé'
    ]    
    technical_keywords = [
        'température', 'pression', 'vibration', 'bruit', 'niveau', 'débit',
        'tension', 'intensité', 'ampérage', 'courant', 'fréquence', 'humidité',
        'poussière', 'pollution', 'vitesse', 'rotation', 'flux', 'couple',
        'densité', 'étanchéité', 'variation', 'oscillation', 'anomalie capteur',
        'signal instable', 'mesure erronée', 'retour capteur', 'temps de cycle',
        'état moteur', 'charge électrique', 'régulation', 'point de consigne',
        'analyse vibratoire', 'sonde thermique', 'alarme capteur', 'taux de fuite',
        'température ambiante', 'pression hydraulique', 'bruit mécanique',
        'niveau réservoir', 'débit massique', 'analyse de gaz', 'tension batterie',
        'température chaudière', 'pression vapeur', 'dérive capteur', 'interférence signal',
        'lecture fausse', 'plage de mesure', 'instabilité capteur'
    ]

    critical_keywords = [
        'critique', 'critical', 'grave', 'important', 'majeur',
        'alarme critique', 'alarme rouge', 'alarme immédiate', 'alarme urgente',
        'alarme système', 'alarme machine', 'alarme électrique', 'alarme hydraulique',
        'alarme pneumatique', 'alarme thermique', 'alarme de fuite', 'alarme de fuite de gaz',
        'alarme de fuite de liquide', 'alarme de fuite de vapeur', 'alarme de fuite de liquide',]
    return {
        'length': len(text),
        'safety_keywords': sum(keyword in text_lower for keyword in safety_keywords),
        'urgent_keywords': sum(keyword in text_lower for keyword in urgent_keywords),
        'maintenance_keywords': sum(keyword in text_lower for keyword in maintenance_keywords),
        'improvement_keywords': sum(keyword in text_lower for keyword in improvement_keywords),
        'technical_keywords': sum(keyword in text_lower for keyword in technical_keywords),
        'critical_keywords': sum(keyword in text_lower for keyword in critical_keywords)
    }

def create_priority_rules(row):
    """Rule-based priority assignment for missing values"""
    # Extract text features
    desc = str(row.get('Description', '')).lower()
    equip = str(row.get('Description equipement', '')).lower()
    
    # High priority indicators (4.0-5.0)
    high_priority_indicators = [
        'safety', 'sécurité', 'urgent', 'critique', 'arrêt', 'immédiat',
        'fuite', 'danger', 'panne', 'défaillance', 'grave'
    ]
    
    # Low priority indicators (1.0-2.0)
    low_priority_indicators = [
        'amélioration', 'optimisation', 'prévoir', 'planifier'
    ]
    
    # Count indicators
    high_count = sum(indicator in desc for indicator in high_priority_indicators)
    low_count = sum(indicator in desc for indicator in low_priority_indicators)
    
    # Apply rules
    if high_count >= 2:
        return 4.5  # High priority
    elif high_count >= 1:
        return 3.5  # Medium-high priority
    elif low_count >= 1:
        return 1.5  # Low priority
    else:
        return 2.5  # Default medium priority

def prepare_enhanced_data(df):
    """Enhanced data preparation with better handling of missing values and imbalance"""
    
    print("=== Enhanced Data Preparation ===")
    
    # Handle missing priorities with rule-based assignment
    missing_priorities = df['Priorité'].isna()
    print(f"Missing priorities before: {missing_priorities.sum()}")
    
    # Apply rule-based priority assignment
    df.loc[missing_priorities, 'Priorité'] = df[missing_priorities].apply(create_priority_rules, axis=1)
    
    print(f"Missing priorities after: {df['Priorité'].isna().sum()}")
    print(f"New priority distribution:")
    print(df['Priorité'].value_counts().sort_index())
    
    # Enhanced text preprocessing
    df['Description_clean'] = df['Description'].fillna('').astype(str)
    df['Equipment_clean'] = df['Description equipement'].fillna('').astype(str)
    
    # Extract enhanced text features
    text_features = df['Description_clean'].apply(improved_text_features)
    text_df = pd.DataFrame(text_features.tolist())
    
    # Add text features to dataframe
    for col in text_df.columns:
        df[f'text_{col}'] = text_df[col]
    
    # Enhanced date features
    df['Date de detection de l\'anomalie'] = pd.to_datetime(df['Date de detection de l\'anomalie'], errors='coerce')
    df['Year'] = df['Date de detection de l\'anomalie'].dt.year
    df['Month'] = df['Date de detection de l\'anomalie'].dt.month
    df['DayOfWeek'] = df['Date de detection de l\'anomalie'].dt.dayofweek
    df['Quarter'] = df['Date de detection de l\'anomalie'].dt.quarter
    
    # Handle date NaNs
    current_year = datetime.now().year
    df['Year'] = df['Year'].fillna(current_year)
    df['Month'] = df['Month'].fillna(6)
    df['DayOfWeek'] = df['DayOfWeek'].fillna(1)
    df['Quarter'] = df['Quarter'].fillna(2)
    
    # Enhanced equipment categorization
    equipment_categories = {
        'pump': ['pompe', 'pump'],
        'motor': ['moteur', 'motor'],
        'valve': ['vanne', 'valve', 'clapet'],
        'turbine': ['turbine'],
        'generator': ['alternateur', 'génératrice', 'generator'],
        'transformer': ['transformateur', 'transfo'],
        'boiler': ['chaudière', 'chaudiere', 'boiler'],
        'fan': ['ventilateur', 'fan', 'soufflante'],
        'compressor': ['compresseur', 'compressor'],
        'heater': ['réchauffeur', 'chauffage', 'heater'],
        'electrical': ['électrique', 'electrical', 'elec']
    }
    
    for category, keywords in equipment_categories.items():
        df[f'equipment_{category}'] = df['Equipment_clean'].str.lower().str.contains('|'.join(keywords), na=False).astype(int)
    
    # Enhanced section categorization
    section_mapping = {
        'mechanical': ['MC', 'mechanical'],
        'electrical': ['EL', 'electrical'],
        'control': ['CT', 'control'],
        'maintenance': ['maintenance', 'MAINT']
    }
    
    df['Section proprietaire'] = df['Section proprietaire'].fillna('UNKNOWN')
    for category, keywords in section_mapping.items():
        df[f'section_{category}'] = df['Section proprietaire'].str.contains('|'.join(keywords), na=False).astype(int)
    
    return df

def enhanced_model_training(df):
    """Enhanced model training with better feature selection and handling of imbalance"""
    
    print("\n=== Enhanced Model Training ===")
    
    # Define features
    numerical_features = [
        'Year', 'Month', 'DayOfWeek', 'Quarter',
        'text_length', 'text_safety_keywords', 'text_urgent_keywords',
        'text_maintenance_keywords', 'text_improvement_keywords', 
        'text_technical_keywords', 'text_critical_keywords'
    ]
    
    # Add equipment features
    equipment_features = [col for col in df.columns if col.startswith('equipment_')]
    section_features = [col for col in df.columns if col.startswith('section_')]
    
    # Combine all numerical features
    all_numerical = numerical_features + equipment_features + section_features
    
    # Prepare features and target
    X_numerical = df[all_numerical].fillna(0)
    
    # TF-IDF for text (with better parameters)
    tfidf = TfidfVectorizer(
        max_features=100,  # Reduced for better generalization
        stop_words=None,  # Keep French stopwords
        ngram_range=(1, 2),
        min_df=3,  # Ignore very rare terms
        max_df=0.95  # Ignore very common terms
    )
    
    X_text = tfidf.fit_transform(df['Description_clean']).toarray()
    text_feature_names = [f'tfidf_{i}' for i in range(X_text.shape[1])]
    
    # Combine features
    X = np.hstack([X_numerical.values, X_text])
    feature_names = all_numerical + text_feature_names
    
    y = df['Priorité'].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"  Priority {val}: {count} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5, labels=False)
    )
    
    # Enhanced models with better parameters
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        'Linear Regression': LinearRegression(),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
    best_model = None
    best_score = float('inf')
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Clip predictions to valid range
        y_pred_train = np.clip(y_pred_train, 1.0, 5.0)
        y_pred_test = np.clip(y_pred_test, 1.0, 5.0)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results[name] = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_mse': test_mse,
            'test_r2': test_r2,
            'model': model
        }
        
        print(f"  Train MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Test MSE: {test_mse:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        
        if test_mae < best_score:
            best_score = test_mae
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} (MAE: {best_score:.4f})")
    
    # Create enhanced pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', best_model)
    ])
    
    # Refit the pipeline
    pipeline.fit(X_train, y_train)
    
    # Test predictions on various priority levels
    print(f"\n=== Testing Model Predictions ===")
    test_cases = [
        "SAFETY: Fuite importante de vapeur - arrêt immédiat requis",
        "urgent: défaut critique alternateur",
        "Prévoir contrôle mesure de température",
        "Amélioration éclairage salle technique",
        "Bruit anormal au niveau du moteur"
    ]
    
    for desc in test_cases:
        # Create test sample
        test_features = improved_text_features(desc)
        test_numerical = [2024, 6, 1, 2] + [test_features[f] for f in ['length', 'safety_keywords', 'urgent_keywords', 'maintenance_keywords', 'improvement_keywords', 'technical_keywords', 'critical_keywords']]
        test_numerical += [0] * (len(equipment_features) + len(section_features))  # No equipment/section info
        
        test_text = tfidf.transform([desc]).toarray()
        test_sample = np.hstack([test_numerical, test_text[0]]).reshape(1, -1)
        
        pred = pipeline.predict(test_sample)[0]
        pred = np.clip(pred, 1.0, 5.0)
        print(f"'{desc[:50]}...' → {pred:.2f}")
    
    return pipeline, tfidf, feature_names, results[best_name]

def main():
    """Main execution function"""
    print("Enhanced TAQA Anomaly Priority Regressor")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('data_set.csv')
    print(f"Loaded {len(df)} records")
    print(f"Original priority distribution:")
    print(df['Priorité'].value_counts().sort_index())
    
    # Enhanced data preparation
    df = prepare_enhanced_data(df)
    
    # Enhanced model training
    model_pipeline, tfidf_vectorizer, feature_names, metrics = enhanced_model_training(df)
    
    # Save enhanced model
    model_filename = 'enhanced_anomaly_priority_regressor.joblib'
    joblib.dump({
        'pipeline': model_pipeline,
        'tfidf': tfidf_vectorizer,
        'feature_names': feature_names
    }, model_filename)
    
    # Save enhanced model info
    model_info = {
        'model_name': 'Enhanced Random Forest',
        'features_count': len(feature_names),
        'training_samples': len(df),
        'mae': metrics['test_mae'],
        'mse': metrics['test_mse'],
        'r2': metrics['test_r2'],
        'priority_range': [1.0, 5.0],
        'created_date': datetime.now().isoformat(),
        'improvements': [
            'Rule-based handling of missing priorities',
            'Enhanced French keyword detection',
            'Better feature engineering',
            'Improved model parameters',
            'Extended priority range support'
        ]
    }
    
    with open('enhanced_regressor_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nEnhanced model saved as: {model_filename}")
    print(f"Model info saved as: enhanced_regressor_model_info.json")
    print("\nModel Performance:")
    print(f"  MAE: {metrics['test_mae']:.4f}")
    print(f"  MSE: {metrics['test_mse']:.4f}")
    print(f"  R²: {metrics['test_r2']:.4f}")

if __name__ == "__main__":
    main() 