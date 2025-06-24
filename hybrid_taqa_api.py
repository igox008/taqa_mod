#!/usr/bin/env python3
"""
Hybrid TAQA API - Best of Both Worlds
Combines lookup system reliability with ML model flexibility
"""

import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HybridTAQAAPI:
    """Hybrid system combining lookup and ML approaches"""
    
    def __init__(self, 
                 lookup_file='taqa_priority_lookup.json',
                 ml_model_file='improved_balanced_taqa_model.joblib'):
        """Initialize hybrid system"""
        self.lookup_data = None
        self.ml_model = None
        self.ml_loaded = False
        self.lookup_loaded = False
        
        # Load lookup system
        try:
            with open(lookup_file, 'r', encoding='utf-8') as f:
                self.lookup_data = json.load(f)
            self.lookup_loaded = True
            print("✅ Lookup system loaded")
        except Exception as e:
            print(f"⚠️ Lookup system not available: {e}")
        
        # Load ML model
        try:
            self.ml_model_data = joblib.load(ml_model_file)
            self.ml_loaded = True
            print("✅ ML model loaded")
        except Exception as e:
            print(f"⚠️ ML model not available: {e}")
        
        if not self.lookup_loaded and not self.ml_loaded:
            raise Exception("Neither lookup system nor ML model could be loaded!")
    
    def lookup_predict(self, description, equipment_type, section):
        """Make prediction using lookup system"""
        if not self.lookup_loaded:
            return None
        
        try:
            # Equipment-based lookup
            equipment_clean = str(equipment_type).lower().strip() if equipment_type else ""
            
            if equipment_clean and equipment_clean in self.lookup_data.get('equipment_priorities', {}):
                equipment_data = self.lookup_data['equipment_priorities'][equipment_clean]
                # Extract the mean priority from the equipment data
                if isinstance(equipment_data, dict) and 'mean' in equipment_data:
                    equipment_priority = equipment_data['mean']
                    confidence = min(0.95, 0.8 + (equipment_data.get('count', 1) * 0.03))
                else:
                    equipment_priority = equipment_data
                    confidence = 0.95
                
                return {
                    'priority_score': equipment_priority,
                    'method': 'Equipment Lookup',
                    'confidence': confidence,
                    'explanation': f'Based on historical data for {equipment_type}'
                }
            
            # Section-based lookup
            section_clean = str(section).strip() if section else ""
            if section_clean and section_clean in self.lookup_data.get('section_averages', {}):
                section_data = self.lookup_data['section_averages'][section_clean]
                # Extract priority from section data
                if isinstance(section_data, dict) and 'mean' in section_data:
                    section_priority = section_data['mean']
                else:
                    section_priority = section_data
                
                return {
                    'priority_score': section_priority,
                    'method': 'Section Average',
                    'confidence': 0.7,
                    'explanation': f'Based on {section} section average'
                }
            
            return None
            
        except Exception as e:
            print(f"Lookup error: {e}")
            return None
    
    def extract_smart_text_features(self, text):
        """Extract smart text features for ML model"""
        if pd.isna(text) or text == '':
            return {
                'length': 0, 'word_count': 0,
                'safety_urgent_score': 0, 'equipment_critical_score': 0,
                'maintenance_routine_score': 0, 'improvement_minor_score': 0,
                'technical_complexity': 0, 'action_immediacy': 0
            }
        
        text_str = str(text).lower()
        words = text_str.split()
        word_count = len(words)
        
        # Critical/Safety indicators
        critical_keywords = [
            'urgent', 'safety', 'sécurité', 'danger', 'risque', 'arrêt', 'immédiat',
            'critique', 'fuite', 'panne', 'défaut', 'dysfonctionnement', 'anomalie'
        ]
        
        # Equipment criticality
        critical_equipment = [
            'pompe alimentaire', 'alternateur', 'chaudière', 'turbine', 
            'transformateur', 'moteur principal', 'système sécurité'
        ]
        
        # Routine maintenance
        routine_keywords = [
            'maintenance', 'préventive', 'contrôle', 'vérification', 'révision',
            'nettoyage', 'inspection', 'programmée', 'planning'
        ]
        
        # Minor improvements
        improvement_keywords = [
            'amélioration', 'optimisation', 'installation', 'éclairage',
            'bureau', 'confort', 'esthétique', 'général'
        ]
        
        # Technical complexity
        technical_keywords = [
            'pression', 'température', 'vibration', 'débit', 'niveau',
            'mesure', 'paramètre', 'analyse', 'diagnostic'
        ]
        
        # Immediacy indicators
        immediacy_keywords = [
            'immédiat', 'urgent', 'maintenant', 'direct', 'aussitôt',
            'rapidement', 'priorité', 'emergency'
        ]
        
        # Calculate scores
        safety_urgent_score = sum(1 for word in critical_keywords if word in text_str) / len(critical_keywords)
        equipment_critical_score = sum(1 for equip in critical_equipment if equip in text_str) / len(critical_equipment)
        maintenance_routine_score = sum(1 for word in routine_keywords if word in text_str) / len(routine_keywords)
        improvement_minor_score = sum(1 for word in improvement_keywords if word in text_str) / len(improvement_keywords)
        technical_complexity = sum(1 for word in technical_keywords if word in text_str) / len(technical_keywords)
        action_immediacy = sum(1 for word in immediacy_keywords if word in text_str) / len(immediacy_keywords)
        
        return {
            'length': len(text_str),
            'word_count': word_count,
            'safety_urgent_score': safety_urgent_score,
            'equipment_critical_score': equipment_critical_score,
            'maintenance_routine_score': maintenance_routine_score,
            'improvement_minor_score': improvement_minor_score,
            'technical_complexity': technical_complexity,
            'action_immediacy': action_immediacy
        }
    
    def ml_predict(self, description, equipment_type, section):
        """Make prediction using ML model"""
        if not self.ml_loaded:
            return None
        
        try:
            # Extract text features
            text_features = self.extract_smart_text_features(description)
            
            # Build feature vector
            features = {}
            for key, value in text_features.items():
                features[f'text_{key}'] = value
            
            # Equipment features
            equipment_clean = str(equipment_type).lower() if equipment_type else ''
            
            high_crit_equipment = [
                'pompe alimentaire', 'alternateur unite', 'chaudière principale', 
                'turbine', 'transformateur', 'évents ballon'
            ]
            features['equipment_high_criticality'] = sum(
                1 for equip in high_crit_equipment if equip in equipment_clean
            )
            
            medium_crit_equipment = ['pompe', 'moteur', 'ventilateur', 'vanne']
            features['equipment_medium_criticality'] = sum(
                1 for equip in medium_crit_equipment if equip in equipment_clean
            )
            
            # Section features
            sections = ['34MC', '34EL', '34CT', '34MD', '34MM', '34MG']
            for sect in sections:
                features[f'section_{sect}'] = 1 if (section and sect in str(section)) else 0
            
            # Priority indicator
            features['priority_indicator'] = 0
            if (features['text_safety_urgent_score'] > 0.1 or 
                features['equipment_high_criticality'] > 0 or 
                features['text_action_immediacy'] > 0.1):
                features['priority_indicator'] = 2
            elif (features['text_improvement_minor_score'] > 0.1 or 
                  features['text_maintenance_routine_score'] > 0.2):
                features['priority_indicator'] = -1
            
            # Create numerical vector
            feature_names = self.ml_model_data['feature_names']
            numerical_cols = [col for col in feature_names if col.startswith(('text_', 'equipment_', 'section_', 'priority_'))]
            X_numerical = np.array([features.get(col, 0) for col in numerical_cols])
            
            # Text vectorization
            X_text = self.ml_model_data['vectorizers']['tfidf'].transform([description]).toarray()
            
            # Combine features
            X = np.hstack([X_numerical.reshape(1, -1), X_text])
            
            # Scale features
            X_scaled = self.ml_model_data['scalers']['robust'].transform(X)
            
            # Get ensemble prediction
            ensemble_models = self.ml_model_data['ensemble_models']
            ensemble_weights = self.ml_model_data['ensemble_weights']
            
            predictions = []
            for i, (name, model) in enumerate(ensemble_models):
                if name in ['Random_Forest', 'Extra_Trees']:
                    pred = model.predict(X)[0]
                else:
                    pred = model.predict(X_scaled)[0]
                pred = np.clip(pred, 1.0, 4.0)
                predictions.append(pred)
            
            # Weighted ensemble
            ensemble_pred = np.average(predictions, weights=ensemble_weights)
            ensemble_pred = np.clip(ensemble_pred, 1.0, 4.0)
            
            # Calculate confidence
            prediction_std = np.std(predictions)
            confidence = max(0.5, 1.0 - (prediction_std / 2.0))
            
            return {
                'priority_score': round(ensemble_pred, 2),
                'method': 'ML Ensemble',
                'confidence': round(confidence, 2),
                'explanation': 'ML-based prediction with domain-specific features'
            }
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            return None
    
    def predict_single(self, description, equipment_type=None, section=None):
        """Hybrid prediction combining lookup and ML"""
        try:
            # Try lookup first (higher accuracy for known cases)
            lookup_result = self.lookup_predict(description, equipment_type, section)
            
            if lookup_result and lookup_result.get('confidence', 0) >= 0.8:
                # High confidence lookup result
                score = lookup_result['priority_score']
                lookup_result['priority_label'] = self.get_priority_label(score)
                lookup_result['color'] = self.get_priority_color(score)
                lookup_result['urgency'] = self.get_urgency_level(score)
                return lookup_result
            
            # Try ML model
            ml_result = self.ml_predict(description, equipment_type, section)
            
            if ml_result:
                score = ml_result['priority_score']
                ml_result['priority_label'] = self.get_priority_label(score)
                ml_result['color'] = self.get_priority_color(score)
                ml_result['urgency'] = self.get_urgency_level(score)
                return ml_result
            
            # Fallback to lookup if ML fails
            if lookup_result:
                score = lookup_result['priority_score']
                lookup_result['priority_label'] = self.get_priority_label(score)
                lookup_result['color'] = self.get_priority_color(score)
                lookup_result['urgency'] = self.get_urgency_level(score)
                return lookup_result
            
            # Last resort: default prediction
            return {
                'priority_score': 2.0,
                'priority_label': 'Moyenne',
                'urgency': 'Planifié',
                'confidence': 0.3,
                'color': 'yellow',
                'explanation': 'Prédiction par défaut - données insuffisantes',
                'method': 'Default'
            }
            
        except Exception as e:
            return {
                'priority_score': 2.0,
                'priority_label': 'Erreur',
                'urgency': 'À déterminer',
                'confidence': 0.0,
                'color': 'gray',
                'explanation': f'Erreur: {str(e)}',
                'method': 'Error'
            }
    
    def get_priority_label(self, score):
        """Get priority label from score"""
        if score >= 4.0:
            return "Très Haute"
        elif score >= 3.0:
            return "Haute"
        elif score >= 2.0:
            return "Moyenne"
        else:
            return "Basse"
    
    def get_priority_color(self, score):
        """Get color for priority score"""
        if score >= 3.5:
            return "red"
        elif score >= 2.5:
            return "orange"
        elif score >= 1.5:
            return "yellow"
        else:
            return "green"
    
    def get_urgency_level(self, score):
        """Get urgency level from score"""
        if score >= 4.0:
            return "Immédiat"
        elif score >= 3.0:
            return "Urgent"
        elif score >= 2.0:
            return "Planifié"
        else:
            return "Préventif"

def test_hybrid_system():
    """Test the hybrid system"""
    print("🔬 TESTING HYBRID TAQA SYSTEM")
    print("=" * 50)
    
    try:
        api = HybridTAQAAPI()
    except Exception as e:
        print(f"❌ Error initializing hybrid system: {e}")
        return
    
    test_cases = [
        {
            'name': 'Known Equipment (Lookup)',
            'description': "Maintenance évents ballon chaudière",
            'equipment': "évents ballon chaudière",
            'section': "34MC"
        },
        {
            'name': 'Critical Safety (ML)',
            'description': "URGENT: Fuite massive vapeur haute pression - danger personnel arrêt immédiat",
            'equipment': "POMPE NOUVELLE",
            'section': "34MC"
        },
        {
            'name': 'Routine Maintenance (ML)',
            'description': "Maintenance préventive programmée révision",
            'equipment': "EQUIPEMENT INCONNU",
            'section': "34MM"
        },
        {
            'name': 'Minor Improvement (ML)',
            'description': "Amélioration éclairage bureau confort",
            'equipment': "NOUVEAU SYSTEME",
            'section': "34EL"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🧪 TEST {i}: {test['name']}")
        print(f"📝 Description: {test['description']}")
        print(f"⚙️ Equipment: {test['equipment']}")
        print(f"🏢 Section: {test['section']}")
        
        result = api.predict_single(
            description=test['description'],
            equipment_type=test['equipment'],
            section=test['section']
        )
        
        print(f"📊 Score: {result['priority_score']}")
        print(f"🏷️ Label: {result['priority_label']}")
        print(f"🎭 Method: {result['method']}")
        print(f"🔒 Confidence: {result['confidence']}")
        print(f"💡 Explanation: {result['explanation']}")
        print("-" * 50)

if __name__ == "__main__":
    test_hybrid_system() 