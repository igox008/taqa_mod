#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from datetime import datetime

class ImprovedAnomalyPredictor:
    def __init__(self):
        self.model = None
        self.tfidf = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def preprocess_text(self, text):
        if pd.isna(text):
            text = "Description non disponible"

        critical_patterns = {
            'emergency': {
                'words': ['urgence', 'urgent', 'critique', 'immédiat', 'arrêt'],
                'weight': 5
            },
            'severe_damage': {
                'words': ['rupture', 'cassé', 'destruction', 'percement'],
                'weight': 4
            },
            'major_leak': {
                'words': ['fuite importante', 'fuite massive', 'fuite majeure'],
                'weight': 4
            },
            'minor_leak': {
                'words': ['fuite légère', 'petite fuite', 'léger suintement'],
                'weight': 2
            },
            'abnormal': {
                'words': ['anormal', 'inhabituel', 'étrange'],
                'weight': 3
            },
            'routine': {
                'words': ['routine', 'normal', 'habituel', 'régulier'],
                'weight': 1
            }
        }
        
        features = {}
        text_lower = str(text).lower()
        
        severity_score = 0
        for category, info in critical_patterns.items():
            category_score = 0
            for word in info['words']:
                if word in text_lower:
                    category_score += info['weight']
            features[f'severity_{category}'] = category_score
            severity_score += category_score
        
        features['total_severity'] = severity_score
        features['word_count'] = len(text_lower.split())
        features['has_urgent_terms'] = 1 if any(word in text_lower for word in critical_patterns['emergency']['words']) else 0
        
        return features

    def _transform_with_unknown_handling(self, encoder, values, default_value='unknown'):
        try:
            return encoder.transform(values)
        except ValueError as e:
            if default_value not in encoder.classes_:
                encoder.classes_ = np.append(encoder.classes_, default_value)
            
            values = values.copy()
            mask = ~values.isin(encoder.classes_[:-1])  
            values[mask] = default_value
            return encoder.transform(values)

    def extract_features(self, data):
        data = data.copy()
        data['Description'] = data['Description'].fillna("Description non disponible")
        
        text_features_list = []
        for text in data['Description']:
            features = self.preprocess_text(text)
            feature_vector = [
                features['severity_emergency'] * 2.0,
                features['severity_severe_damage'] * 1.8,
                features['severity_major_leak'] * 1.6,
                features['severity_minor_leak'] * 1.0,
                features['severity_abnormal'] * 1.4,
                features['severity_routine'] * 0.5,
                features['total_severity'] * 1.5,
                features['has_urgent_terms'] * 2.0
            ]
            text_features_list.append(feature_vector)
        
        text_features = np.array(text_features_list)
        
        data['Description de l\'équipement'] = data['Description de l\'équipement'].fillna("Équipement inconnu")
        data['Section propriétaire'] = data['Section propriétaire'].fillna("Section inconnue")
        
        if 'equipment_encoder' not in self.label_encoders:
            self.label_encoders['equipment_encoder'] = LabelEncoder()
            equipment_encoded = self.label_encoders['equipment_encoder'].fit_transform(data['Description de l\'équipement'])
        else:
            equipment_encoded = self._transform_with_unknown_handling(
                self.label_encoders['equipment_encoder'],
                data['Description de l\'équipement'],
                default_value='Équipement inconnu'
            )
        
        if 'section_encoder' not in self.label_encoders:
            self.label_encoders['section_encoder'] = LabelEncoder()
            section_encoded = self.label_encoders['section_encoder'].fit_transform(data['Section propriétaire'])
        else:
            section_encoded = self._transform_with_unknown_handling(
                self.label_encoders['section_encoder'],
                data['Section propriétaire'],
                default_value='Section inconnue'
            )
        
        severity_scores = (
            data['Fiabilité Intégrité'].values * 2.0 +
            data['Disponibilté'].values * 1.5 +
            data['Process Safety'].values * 2.5
        ).reshape(-1, 1)
        
        feature_matrix = np.hstack([
            text_features * 2.0,
            severity_scores,
            equipment_encoded.reshape(-1, 1),
            section_encoded.reshape(-1, 1)
        ])
        
        if not hasattr(self.scaler, 'mean_'):
            feature_matrix = self.scaler.fit_transform(feature_matrix)
        else:
            feature_matrix = self.scaler.transform(feature_matrix)
        
        return feature_matrix

    def train(self, data_file):
        print("Loading data...")
        data = pd.read_csv(data_file)

        print(f"Initial dataset size: {len(data)}")
        nan_criticite = data['Criticité'].isna().sum()
        if nan_criticite > 0:
            print(f"Found {nan_criticite} rows with missing criticality values")
            data = data.dropna(subset=['Criticité'])
            print(f"Dataset size after removing missing values: {len(data)}")
        
        data['severity_score'] = (
            data['Fiabilité Intégrité'] * 2.0 +
            data['Disponibilté'] * 1.5 +
            data['Process Safety'] * 2.5
        )
        
        print("Extracting features...")
        X = self.extract_features(data)
        y = data['Criticité'].astype(int)
        
        self.criticality_min = y.min()
        self.criticality_max = y.max()
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Training model...")
        self.model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=3,
            subsample=0.8,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        print("\nEvaluating model...")
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        print(f"Train accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return self.model

    def predict(self, description, equipment_type, section, fiabilite=3, disponibilite=3, process_safety=3):
        data = pd.DataFrame({
            'Description': [description],
            'Description de l\'équipement': [equipment_type],
            'Section propriétaire': [section],
            'Fiabilité Intégrité': [fiabilite],
            'Disponibilté': [disponibilite],
            'Process Safety': [process_safety]
        })
        
        X = self.extract_features(data)
        
        text_features = self.preprocess_text(description)
        
        base_severity = (
            fiabilite * 2.0 +
            disponibilite * 1.5 +
            process_safety * 2.5
        )
        
        text_severity = text_features['total_severity']
        emergency_factor = text_features['severity_emergency'] * 2.0
        damage_factor = text_features['severity_severe_damage'] * 1.8
        
        total_severity = (base_severity + text_severity * 0.5 + emergency_factor + damage_factor) / 2
        
        criticality = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        prediction_idx = int(criticality - min(self.model.classes_))
        raw_confidence = probabilities[prediction_idx]
        
        predicted_severity = criticality / self.criticality_max * 20
        severity_diff = abs(predicted_severity - total_severity)
        
        if severity_diff > 5:
            confidence_penalty = min(severity_diff * 0.1, 0.5)
            raw_confidence *= (1 - confidence_penalty)
        confidence = raw_confidence * 100
        
        confidence = max(min(confidence, 100.0), 0.0)
        
        if text_features['has_urgent_terms'] and text_severity > 15:
            criticality = max(criticality, 11)
        elif text_features['has_urgent_terms'] and text_severity > 10:
            criticality = max(criticality, 9)
        elif text_features['severity_major_leak'] > 0 or text_severity > 8:
            criticality = max(criticality, 7)
        elif text_features['severity_routine'] > 0 and text_severity < 3:
            criticality = min(criticality, 5)
        
        criticality = max(min(criticality, self.criticality_max), self.criticality_min)
        
        return {
            'criticality': int(criticality),
            'confidence': round(confidence, 2),
            'severity_scores': text_features,
            'total_severity': float(total_severity)
        }

    def save_model(self, filename='models/improved_anomaly_predictor.joblib'):
        if not hasattr(self, 'criticality_min') or not hasattr(self, 'criticality_max'):
            raise ValueError("Model must be trained before saving")
            
        model_components = {
            'model': self.model,
            'tfidf': self.tfidf,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'criticality_min': self.criticality_min,
            'criticality_max': self.criticality_max
        }
        joblib.dump(model_components, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename='models/improved_anomaly_predictor.joblib'):
        components = joblib.load(filename)
        self.model = components['model']
        self.tfidf = components['tfidf']
        self.label_encoders = components['label_encoders']
        self.scaler = components['scaler']
        self.criticality_min = components['criticality_min']
        self.criticality_max = components['criticality_max']

def main():
    predictor = ImprovedAnomalyPredictor()
    predictor.train('data_set.csv')
    predictor.save_model()
            
    test_cases = [
        "Fuite importante d'huile avec vibrations anormales. Température très élevée et bruit critique.",
        "Légère fuite d'eau observée. Contrôle de routine nécessaire.",
        "Prévoir un nettoyage et graissage de routine.",
        "Rupture du joint principal avec fuite massive. Arrêt d'urgence nécessaire.",
        "Usure normale des roulements. À surveiller."
    ]
    
    print("\nTesting predictions:")
    for desc in test_cases:
        result = predictor.predict(desc, "Pompe centrifuge", "Production")
        print(f"\nDescription: {desc}")
        print(f"Criticality: {result['criticality']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("Severity scores:", result['severity_scores'])

if __name__ == "__main__":
    main() 