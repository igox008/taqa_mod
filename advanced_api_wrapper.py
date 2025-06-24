#!/usr/bin/env python3
"""
Advanced TAQA API Wrapper
High-performance API for the advanced ensemble ML model
"""

import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedTAQAAPI:
    """Advanced API for TAQA anomaly priority prediction"""
    
    def __init__(self, model_path='advanced_taqa_model.joblib'):
        """Initialize the advanced API"""
        try:
            print("🔄 Loading advanced TAQA model...")
            self.model_data = joblib.load(model_path)
            
            self.models = self.model_data['models']
            self.ensemble_models = self.model_data['ensemble_models']
            self.ensemble_weights = self.model_data['ensemble_weights']
            self.vectorizers = self.model_data['vectorizers']
            self.scalers = self.model_data['scalers']
            self.feature_extractors = self.model_data['feature_extractors']
            self.feature_names = self.model_data['feature_names']
            
            print(f"✅ Loaded ensemble with {len(self.ensemble_models)} models")
            print(f"✅ Feature count: {len(self.feature_names)}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def extract_advanced_text_features(self, text):
        """Extract sophisticated text features (same as training)"""
        if pd.isna(text) or text == '':
            return {
                'length': 0, 'word_count': 0, 'sentence_count': 0,
                'avg_word_length': 0, 'capital_ratio': 0, 'digit_ratio': 0,
                'punctuation_ratio': 0, 'unique_word_ratio': 0,
                'safety_score': 0, 'urgency_score': 0, 'technical_score': 0,
                'maintenance_score': 0, 'equipment_score': 0,
                'sentiment_score': 0, 'complexity_score': 0
            }
        
        text_str = str(text).lower()
        
        # Basic metrics
        length = len(text_str)
        words = text_str.split()
        word_count = len(words)
        sentence_count = len([s for s in text_str.split('.') if s.strip()])
        
        # Advanced metrics
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        capital_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        digit_ratio = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        punctuation_ratio = sum(1 for c in text if c in '.,!?;:') / len(text) if text else 0
        unique_word_ratio = len(set(words)) / len(words) if words else 0
        
        # Domain-specific scoring
        safety_keywords = ['fuite', 'urgent', 'arrêt', 'danger', 'risque', 'sécurité', 'défaut', 'panne']
        urgency_keywords = ['immédiat', 'urgent', 'critique', 'priorité', 'maintenance', 'réparation']
        technical_keywords = ['température', 'pression', 'vibration', 'débit', 'niveau', 'mesure', 'contrôle']
        maintenance_keywords = ['prévoir', 'maintenance', 'révision', 'inspection', 'vérification', 'nettoyage']
        equipment_keywords = ['pompe', 'moteur', 'vanne', 'alternateur', 'chaudière', 'turbine', 'ventilateur']
        
        safety_score = sum(1 for word in safety_keywords if word in text_str) / len(safety_keywords)
        urgency_score = sum(1 for word in urgency_keywords if word in text_str) / len(urgency_keywords)
        technical_score = sum(1 for word in technical_keywords if word in text_str) / len(technical_keywords)
        maintenance_score = sum(1 for word in maintenance_keywords if word in text_str) / len(maintenance_keywords)
        equipment_score = sum(1 for word in equipment_keywords if word in text_str) / len(equipment_keywords)
        
        # Sentiment and complexity
        positive_words = ['amélioration', 'optimisation', 'bon', 'correct', 'normal']
        negative_words = ['défaut', 'panne', 'problème', 'erreur', 'anomalie', 'dysfonctionnement']
        
        sentiment_score = (sum(1 for word in positive_words if word in text_str) - 
                          sum(1 for word in negative_words if word in text_str)) / max(len(words), 1)
        
        complexity_score = (avg_word_length * 0.3 + sentence_count * 0.2 + 
                          punctuation_ratio * 0.5) if sentence_count > 0 else 0
        
        return {
            'length': length,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'capital_ratio': capital_ratio,
            'digit_ratio': digit_ratio,
            'punctuation_ratio': punctuation_ratio,
            'unique_word_ratio': unique_word_ratio,
            'safety_score': safety_score,
            'urgency_score': urgency_score,
            'technical_score': technical_score,
            'maintenance_score': maintenance_score,
            'equipment_score': equipment_score,
            'sentiment_score': sentiment_score,
            'complexity_score': complexity_score
        }
    
    def create_feature_vector(self, description, equipment_type=None, section=None, date=None):
        """Create feature vector for prediction"""
        
        # Extract text features
        text_features = self.extract_advanced_text_features(description)
        
        # Build feature dictionary
        features = {}
        
        # Add text features with prefix
        for key, value in text_features.items():
            features[f'text_{key}'] = value
        
        # Temporal features (use current date if not provided)
        if date is None:
            date = datetime.now()
        elif isinstance(date, str):
            date = pd.to_datetime(date, errors='coerce')
            if pd.isna(date):
                date = datetime.now()
        
        features['temporal_year'] = date.year
        features['temporal_month'] = date.month
        features['temporal_day'] = date.day
        features['temporal_dayofweek'] = date.weekday()
        features['temporal_quarter'] = (date.month - 1) // 3 + 1
        features['temporal_is_weekend'] = 1 if date.weekday() >= 5 else 0
        features['temporal_is_month_start'] = 1 if date.day <= 7 else 0
        features['temporal_is_month_end'] = 1 if date.day >= 24 else 0
        features['temporal_day_of_year'] = date.timetuple().tm_yday
        features['temporal_month_sin'] = np.sin(2 * np.pi * features['temporal_month'] / 12)
        features['temporal_month_cos'] = np.cos(2 * np.pi * features['temporal_month'] / 12)
        features['temporal_day_sin'] = np.sin(2 * np.pi * features['temporal_dayofweek'] / 7)
        features['temporal_day_cos'] = np.cos(2 * np.pi * features['temporal_dayofweek'] / 7)
        
        # Equipment features
        equipment_clean = str(equipment_type).lower() if equipment_type else ''
        description_clean = str(description).lower() if description else ''
        
        # Equipment categories
        equipment_categories = {
            'critical_rotating': ['pompe', 'moteur', 'ventilateur', 'turbine', 'alternateur'],
            'critical_static': ['chaudière', 'transformateur', 'échangeur'],
            'control_systems': ['vanne', 'clapet', 'servomoteur', 'régulateur'],
            'auxiliary': ['éclairage', 'prises', 'câblage', 'armoire'],
            'safety_systems': ['soupape', 'détecteur', 'alarme', 'sécurité'],
            'maintenance_tools': ['ramoneur', 'décrasseur', 'nettoyeur']
        }
        
        for category, keywords in equipment_categories.items():
            equipment_match = any(keyword in equipment_clean for keyword in keywords)
            description_match = any(keyword in description_clean for keyword in keywords)
            features[f'equipment_cat_{category}'] = 1 if (equipment_match or description_match) else 0
        
        # Equipment complexity and criticality
        complexity_indicators = ['pompe alimentaire', 'moteur principal', 'alternateur unité', 'chaudière']
        features['equipment_complexity'] = sum(1 for indicator in complexity_indicators if indicator in equipment_clean)
        
        high_criticality = ['évents ballon chaudière', 'pompe alimentaire', 'alternateur']
        medium_criticality = ['ventilateur', 'pompe', 'moteur']
        
        features['equipment_criticality'] = 1  # Default
        for equipment_name in high_criticality:
            if equipment_name in equipment_clean:
                features['equipment_criticality'] = 3
                break
        else:
            for equipment_name in medium_criticality:
                if equipment_name in equipment_clean:
                    features['equipment_criticality'] = 2
                    break
        
        # Section dummies
        sections = ['34MC', '34EL', '34CT', '34MD', '34MM', '34MG']
        for sect in sections:
            features[f'section_{sect}'] = 1 if (section and sect in str(section)) else 0
        
        # Status dummies (assume Terminé as default)
        features['status_Terminé'] = 1
        features['status_En cours'] = 0
        features['status_Nouveau'] = 0
        
        # Create numerical vector
        numerical_cols = [col for col in self.feature_names if col.startswith(('text_', 'temporal_', 'equipment_', 'section_', 'status_'))]
        X_numerical = np.array([features.get(col, 0) for col in numerical_cols])
        
        # Text vectorization
        X_tfidf_word = self.vectorizers['tfidf_word'].transform([description]).toarray()
        X_tfidf_char = self.vectorizers['tfidf_char'].transform([description]).toarray()
        
        # Polynomial features
        if 'polynomial' in self.feature_extractors:
            key_features = ['text_safety_score', 'text_urgency_score', 'text_technical_score', 'equipment_criticality']
            key_indices = [i for i, col in enumerate(numerical_cols) if col in key_features]
            if key_indices:
                X_poly = self.feature_extractors['polynomial'].transform(X_numerical[key_indices].reshape(1, -1))
            else:
                X_poly = np.array([]).reshape(1, 0)
        else:
            X_poly = np.array([]).reshape(1, 0)
        
        # Combine features
        X_combined = np.hstack([X_numerical.reshape(1, -1), X_tfidf_word, X_tfidf_char, X_poly])
        
        # Apply feature selection
        X_selected = self.feature_extractors['selector'].transform(X_combined)
        
        return X_selected
    
    def predict_single(self, description, equipment_type=None, section=None, date=None):
        """Predict priority for a single anomaly"""
        try:
            # Create feature vector
            X = self.create_feature_vector(description, equipment_type, section, date)
            
            # Scale features
            X_scaled = self.scalers['robust'].transform(X)
            
            # Get predictions from ensemble
            predictions = []
            model_contributions = {}
            
            for i, (name, model) in enumerate(self.ensemble_models):
                if name in ['Random_Forest', 'Extra_Trees', 'Gradient_Boosting']:
                    pred = model.predict(X)[0]
                else:
                    pred = model.predict(X_scaled)[0]
                
                pred = np.clip(pred, 1.0, 5.0)
                predictions.append(pred)
                model_contributions[name] = {
                    'prediction': pred,
                    'weight': self.ensemble_weights[i]
                }
            
            # Weighted ensemble prediction
            ensemble_pred = np.average(predictions, weights=self.ensemble_weights)
            ensemble_pred = np.clip(ensemble_pred, 1.0, 5.0)
            
            # Priority label
            if ensemble_pred >= 4.0:
                priority_label = "Très Haute"
                urgency = "Immédiat"
                color = "red"
            elif ensemble_pred >= 3.0:
                priority_label = "Haute"
                urgency = "Urgent"
                color = "orange"
            elif ensemble_pred >= 2.0:
                priority_label = "Moyenne"
                urgency = "Planifié"
                color = "yellow"
            else:
                priority_label = "Basse"
                urgency = "Préventif"
                color = "green"
            
            # Confidence based on model agreement
            prediction_std = np.std(predictions)
            confidence = max(0.5, 1.0 - (prediction_std / 2.0))
            
            # Feature analysis
            text_features = self.extract_advanced_text_features(description)
            key_factors = []
            
            if text_features['safety_score'] > 0:
                key_factors.append("Indicateurs de sécurité détectés")
            if text_features['urgency_score'] > 0:
                key_factors.append("Mots d'urgence identifiés")
            if text_features['technical_score'] > 0:
                key_factors.append("Termes techniques présents")
            
            explanation = f"Prédiction basée sur analyse ML avancée. " + \
                         f"Facteurs clés: {', '.join(key_factors) if key_factors else 'Analyse textuelle générale'}"
            
            return {
                'priority_score': round(ensemble_pred, 2),
                'priority_label': priority_label,
                'urgency': urgency,
                'confidence': round(confidence, 2),
                'color': color,
                'explanation': explanation,
                'model_contributions': model_contributions,
                'method': 'Advanced ML Ensemble',
                'feature_analysis': {
                    'safety_score': text_features['safety_score'],
                    'urgency_score': text_features['urgency_score'],
                    'technical_score': text_features['technical_score']
                }
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                'priority_score': 2.0,
                'priority_label': "Erreur",
                'urgency': "À déterminer",
                'confidence': 0.0,
                'color': "gray",
                'explanation': f"Erreur lors de la prédiction: {str(e)}",
                'method': 'Error',
                'error': str(e)
            }

def test_advanced_api():
    """Test the advanced API"""
    print("🧪 Testing Advanced TAQA API")
    print("=" * 50)
    
    try:
        api = AdvancedTAQAAPI('advanced_taqa_model.joblib')
    except:
        print("❌ Model not found. Please train the advanced model first.")
        return
    
    test_cases = [
        {
            'description': "URGENT: Fuite importante de vapeur - arrêt immédiat requis",
            'equipment': "POMPE ALIMENTAIRE PRINCIPALE",
            'section': "34MC"
        },
        {
            'description': "Contrôle préventif de routine - vérification des niveaux",
            'equipment': "RESERVOIR D'EAU",
            'section': "34MM"
        },
        {
            'description': "Défaut alternateur unité - vibrations anormales détectées",
            'equipment': "ALTERNATEUR UNITE 1",
            'section': "34EL"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['description'][:50]}...")
        
        result = api.predict_single(
            description=test['description'],
            equipment_type=test['equipment'],
            section=test['section']
        )
        
        print(f"  📊 Score: {result['priority_score']} | Label: {result['priority_label']}")
        print(f"  💡 {result['explanation']}")
        print()

if __name__ == "__main__":
    test_advanced_api() 