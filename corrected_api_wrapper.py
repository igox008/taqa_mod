import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

class CorrectedAnomalyRegressorAPI:
    """
    Corrected API wrapper for the TAQA Anomaly Priority Regressor
    Uses the properly trained model with fixed missing data
    """
    
    def __init__(self, model_path: str = None):
        self.model_data = None
        self.pipeline = None
        self.tfidf = None
        self.feature_names = None
        self.model_info = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load the corrected trained model from file"""
        try:
            self.model_data = joblib.load(model_path)
            self.pipeline = self.model_data['pipeline']
            self.tfidf = self.model_data['tfidf']
            self.feature_names = self.model_data['feature_names']
            print(f"Corrected model loaded successfully from {model_path}")
            
            # Try to load corrected model info
            try:
                with open('corrected_regressor_model_info.json', 'r') as f:
                    self.model_info = json.load(f)
            except FileNotFoundError:
                self.model_info = {'priority_range': [1.0, 5.0]}
                
        except Exception as e:
            print(f"Error loading corrected model: {e}")
            raise
    
    def _improved_text_features(self, text):
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
    
    def predict_single(self, 
                      description: str, 
                      equipment_type: str = "", 
                      section: str = "", 
                      status: str = "Terminé",
                      detection_date: Optional[str] = None) -> Dict:
        """Predict priority using the corrected model"""
        
        if self.pipeline is None:
            raise ValueError("No corrected model loaded. Please load a model first.")
        
        if detection_date is None:
            detection_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create corrected sample data
        sample_data = self._create_corrected_sample_data(description, equipment_type, section, status, detection_date)
        
        try:
            prediction = self.pipeline.predict(sample_data)[0]
            prediction = float(np.clip(prediction, 1.0, 5.0))
            
            # Get explanation
            explanation = self._get_prediction_explanation(description, prediction)
            confidence = self._calculate_confidence(description, prediction)
            
            return {
                'priority_score': round(prediction, 2),
                'priority_range': self.model_info.get('priority_range', [1.0, 5.0]),
                'explanation': explanation,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'input_data': {
                    'description': description,
                    'equipment_type': equipment_type,
                    'section': section,
                    'status': status,
                    'detection_date': detection_date
                }
            }
            
        except Exception as e:
            return {
                'error': f"Corrected prediction failed: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def _create_corrected_sample_data(self, description: str, equipment_type: str, section: str, status: str, detection_date: str) -> np.ndarray:
        """Create corrected sample data for prediction"""
        
        # Parse detection date
        try:
            detection_dt = pd.to_datetime(detection_date)
        except:
            detection_dt = pd.Timestamp.now()
        
        # Extract text features
        text_features = self._improved_text_features(description)
        
        # Numerical features (matching training exactly)
        numerical_features = [
            detection_dt.year,
            detection_dt.month,
            detection_dt.dayofweek,
            text_features['length'],
            text_features['safety_keywords'],
            text_features['urgent_keywords'],
            text_features['maintenance_keywords'],
            text_features['improvement_keywords'],
            text_features['critical_keywords']
        ]
        
        # Equipment features (matching training)
        equipment_categories = ['pump', 'motor', 'valve', 'turbine', 'generator', 'transformer', 'boiler', 'fan']
        equipment_lower = equipment_type.lower()
        for category in equipment_categories:
            has_equipment = category in equipment_lower
            numerical_features.append(1 if has_equipment else 0)
        
        # Section features (matching training)
        section_keywords = {'mechanical': ['MC'], 'electrical': ['EL'], 'control': ['CT'], 'maintenance': ['MAINT']}
        for category, keywords in section_keywords.items():
            has_section = any(keyword in section for keyword in keywords)
            numerical_features.append(1 if has_section else 0)
        
        # TF-IDF features for text
        text_tfidf = self.tfidf.transform([description]).toarray()
        
        # Combine features
        sample_features = np.hstack([numerical_features, text_tfidf[0]])
        
        return sample_features.reshape(1, -1)
    
    def _get_prediction_explanation(self, description: str, prediction: float) -> str:
        """Get human-readable explanation for the prediction"""
        text_features = self._improved_text_features(description)
        
        explanations = []
        
        # Priority level explanation
        if prediction >= 4.0:
            explanations.append("High priority detected")
        elif prediction >= 3.0:
            explanations.append("Medium-high priority")
        elif prediction >= 2.0:
            explanations.append("Medium priority")
        else:
            explanations.append("Low priority")
        
        # Feature-based explanations
        if text_features['safety_keywords'] > 0:
            explanations.append("safety-related keywords detected")
        if text_features['urgent_keywords'] > 0:
            explanations.append("urgent keywords detected")
        if text_features['critical_keywords'] > 0:
            explanations.append("critical keywords detected")
        if text_features['improvement_keywords'] > 0:
            explanations.append("improvement-related content")
        if text_features['maintenance_keywords'] > 0:
            explanations.append("maintenance-related content")
        
        return " - " + ", ".join(explanations) if explanations else "Standard priority assessment"
    
    def _calculate_confidence(self, description: str, prediction: float) -> str:
        """Calculate confidence level for the prediction"""
        text_features = self._improved_text_features(description)
        
        total_keywords = (text_features['safety_keywords'] + 
                         text_features['urgent_keywords'] + 
                         text_features['critical_keywords'] +
                         text_features['improvement_keywords'] +
                         text_features['maintenance_keywords'])
        
        if total_keywords >= 3:
            return "High"
        elif total_keywords >= 1:
            return "Medium"
        else:
            return "Low"
    
    def health_check(self) -> Dict:
        """Perform a health check on the corrected model"""
        try:
            test_result = self.predict_single(
                description="Test safety issue: fuite importante",
                equipment_type="Test Equipment",
                section="34MC"
            )
            
            return {
                'status': 'healthy',
                'model_loaded': self.pipeline is not None,
                'test_prediction_successful': 'error' not in test_result,
                'test_prediction_score': test_result.get('priority_score', 'N/A'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'model_loaded': self.pipeline is not None,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded corrected model"""
        if self.model_info is None:
            return {'error': 'No corrected model loaded'}
        
        return {
            'model_info': self.model_info,
            'model_loaded': self.pipeline is not None,
            'timestamp': datetime.now().isoformat()
        }

# Test the corrected API
if __name__ == "__main__":
    api = CorrectedAnomalyRegressorAPI('corrected_anomaly_priority_regressor.joblib')
    
    test_cases = [
        ('SAFETY: Fuite importante de vapeur - arrêt immédiat requis', 'TURBINE', '34MC'),
        ('urgent: défaut critique alternateur', 'ALTERNATEUR', '34EL'), 
        ('Prévoir contrôle mesure de température', 'MOTEUR', '34CT'),
        ('Amélioration éclairage salle technique', 'ÉCLAIRAGE', '34EL')
    ]
    
    print("=== CORRECTED MODEL TEST ===")
    for desc, equip, section in test_cases:
        result = api.predict_single(desc, equip, section)
        score = result.get('priority_score', 'ERROR')
        explanation = result.get('explanation', '')
        confidence = result.get('confidence', '')
        print(f"Score: {score} | Confidence: {confidence}")
        print(f"Description: {desc[:60]}...")
        print(f"Explanation: {explanation}")
        print("-" * 70)
