import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

class EnhancedAnomalyRegressorAPI:
    """
    Enhanced API wrapper for the TAQA Anomaly Priority Regressor
    Works with the improved model that handles missing data and provides better predictions
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the enhanced API wrapper
        
        Args:
            model_path: Path to the enhanced trained model file (.joblib)
        """
        self.model_data = None
        self.pipeline = None
        self.tfidf = None
        self.feature_names = None
        self.model_info = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load the enhanced trained model from file
        
        Args:
            model_path: Path to the .joblib model file
        """
        try:
            self.model_data = joblib.load(model_path)
            self.pipeline = self.model_data['pipeline']
            self.tfidf = self.model_data['tfidf']
            self.feature_names = self.model_data['feature_names']
            print(f"Enhanced model loaded successfully from {model_path}")
            
            # Try to load enhanced model info
            try:
                with open('enhanced_regressor_model_info.json', 'r') as f:
                    self.model_info = json.load(f)
            except FileNotFoundError:
                print("Enhanced model info file not found, using default settings")
                self.model_info = {'priority_range': [1.0, 5.0]}
                
        except Exception as e:
            print(f"Error loading enhanced model: {e}")
            raise
    
    def _improved_text_features(self, text):
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
        safety_keywords = ['safety', 'sécurité', 'danger', 'risque', 'fuite', 'explosion', 'incendie', 'brûlure']
        urgent_keywords = ['urgent', 'immédiat', 'critique', 'arrêt', 'panne', 'défaut', 'défaillance']
        maintenance_keywords = ['prévoir', 'contrôle', 'vérifier', 'surveiller', 'maintenance']
        improvement_keywords = ['amélioration', 'optimisation', 'modernisation', 'upgrade']
        technical_keywords = ['température', 'pression', 'vibration', 'bruit', 'niveau', 'débit']
        critical_keywords = ['critique', 'critical', 'grave', 'important', 'majeur']
        
        return {
            'length': len(text),
            'safety_keywords': sum(keyword in text_lower for keyword in safety_keywords),
            'urgent_keywords': sum(keyword in text_lower for keyword in urgent_keywords),
            'maintenance_keywords': sum(keyword in text_lower for keyword in maintenance_keywords),
            'improvement_keywords': sum(keyword in text_lower for keyword in improvement_keywords),
            'technical_keywords': sum(keyword in text_lower for keyword in technical_keywords),
            'critical_keywords': sum(keyword in text_lower for keyword in critical_keywords)
        }
    
    def predict_single(self, 
                      description: str, 
                      equipment_type: str = "", 
                      section: str = "", 
                      status: str = "Terminé",
                      detection_date: Optional[str] = None) -> Dict:
        """
        Predict priority for a single anomaly using the enhanced model
        
        Args:
            description: Anomaly description (French text)
            equipment_type: Type of equipment affected
            section: Department/section responsible
            status: Current status of the anomaly
            detection_date: Date when anomaly was detected (optional)
            
        Returns:
            Dictionary with prediction results
        """
        if self.pipeline is None:
            raise ValueError("No enhanced model loaded. Please load a model first.")
        
        # Use current date if not provided
        if detection_date is None:
            detection_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create enhanced sample data
        sample_data = self._create_enhanced_sample_data(
            description, equipment_type, section, status, detection_date
        )
        
        # Make prediction
        try:
            prediction = self.pipeline.predict(sample_data)[0]
            prediction = float(np.clip(prediction, 1.0, 5.0))
            
            # Get confidence/explanation
            explanation = self._get_prediction_explanation(description, prediction)
            
            return {
                'priority_score': round(prediction, 2),
                'priority_range': self.model_info.get('priority_range', [1.0, 5.0]),
                'explanation': explanation,
                'confidence': self._calculate_confidence(description, prediction),
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
                'error': f"Enhanced prediction failed: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def _create_enhanced_sample_data(self, description: str, equipment_type: str, section: str, 
                                   status: str, detection_date: str) -> np.ndarray:
        """
        Create enhanced sample data for prediction
        
        Args:
            description: Anomaly description
            equipment_type: Equipment type
            section: Section/department
            status: Status
            detection_date: Detection date
            
        Returns:
            Numpy array with enhanced sample data
        """
        # Parse detection date
        try:
            detection_dt = pd.to_datetime(detection_date)
        except:
            detection_dt = pd.Timestamp.now()
        
        # Extract enhanced text features
        text_features = self._improved_text_features(description)
        
        # Numerical features
        numerical_features = [
            detection_dt.year,
            detection_dt.month,
            detection_dt.dayofweek,
            detection_dt.quarter,
            text_features['length'],
            text_features['safety_keywords'],
            text_features['urgent_keywords'],
            text_features['maintenance_keywords'],
            text_features['improvement_keywords'],
            text_features['technical_keywords'],
            text_features['critical_keywords']
        ]
        
        # Equipment categorization
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
        
        equipment_lower = equipment_type.lower()
        for category, keywords in equipment_categories.items():
            has_equipment = any(keyword in equipment_lower for keyword in keywords)
            numerical_features.append(1 if has_equipment else 0)
        
        # Section categorization
        section_categories = {
            'mechanical': ['MC', 'mechanical'],
            'electrical': ['EL', 'electrical'],
            'control': ['CT', 'control'],
            'maintenance': ['maintenance', 'MAINT']
        }
        
        for category, keywords in section_categories.items():
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
        
        # Confidence based on keyword detection
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
    
    def predict_batch(self, anomalies: List[Dict]) -> List[Dict]:
        """
        Predict priorities for multiple anomalies using enhanced model
        
        Args:
            anomalies: List of dictionaries, each containing anomaly data
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, anomaly in enumerate(anomalies):
            try:
                result = self.predict_single(
                    description=anomaly.get('description', ''),
                    equipment_type=anomaly.get('equipment_type', ''),
                    section=anomaly.get('section', ''),
                    status=anomaly.get('status', 'Terminé'),
                    detection_date=anomaly.get('detection_date')
                )
                result['anomaly_id'] = anomaly.get('id', i)
                results.append(result)
                
            except Exception as e:
                results.append({
                    'anomaly_id': anomaly.get('id', i),
                    'error': f"Enhanced prediction failed: {str(e)}",
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded enhanced model
        
        Returns:
            Dictionary with model information
        """
        if self.model_info is None:
            return {'error': 'No enhanced model loaded'}
        
        return {
            'model_info': self.model_info,
            'model_loaded': self.pipeline is not None,
            'timestamp': datetime.now().isoformat()
        }
    
    def health_check(self) -> Dict:
        """
        Perform a health check on the enhanced model
        
        Returns:
            Health status dictionary
        """
        try:
            # Test prediction with sample data
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


# Example usage and testing
def main():
    """Example usage of the enhanced API wrapper"""
    
    # Initialize enhanced API
    api = EnhancedAnomalyRegressorAPI()
    
    # Try to load the enhanced model
    try:
        api.load_model('enhanced_anomaly_priority_regressor.joblib')
    except FileNotFoundError:
        print("Enhanced model file not found. Please train the enhanced model first.")
        return
    
    # Test single prediction
    print("=== Enhanced Single Prediction Test ===")
    result = api.predict_single(
        description="SAFETY: Fuite importante de vapeur par tresse ramoneur IK 14",
        equipment_type="RAMONEUR LONG RETRACTABLE N°14 TYPE IK",
        section="34MC"
    )
    
    print(f"Prediction Result: {result}")
    
    # Test batch prediction
    print("\n=== Enhanced Batch Prediction Test ===")
    test_anomalies = [
        {
            'id': 1,
            'description': 'Prévoir contrôle mesure de température moteur broyeur A',
            'equipment_type': 'MOTEUR BROYEUR A',
            'section': '34CT'
        },
        {
            'id': 2,
            'description': 'Amélioration éclairage salle technique',
            'equipment_type': 'ECLAIRAGE',
            'section': '34EL'
        },
        {
            'id': 3,
            'description': 'SAFETY: Défaut critique alternateur - arrêt immédiat requis',
            'equipment_type': 'ALTERNATEUR',
            'section': '34EL'
        },
        {
            'id': 4,
            'description': 'urgent: fuite importante niveau turbine',
            'equipment_type': 'TURBINE',
            'section': '34MC'
        }
    ]
    
    batch_results = api.predict_batch(test_anomalies)
    
    for result in batch_results:
        score = result.get('priority_score', 'Error')
        explanation = result.get('explanation', '')
        confidence = result.get('confidence', '')
        print(f"Anomaly {result['anomaly_id']}: Score {score} ({confidence} confidence) {explanation}")
    
    # Health check
    print("\n=== Enhanced Health Check ===")
    health = api.health_check()
    print(f"Health Status: {health}")
    
    # Model info
    print("\n=== Enhanced Model Information ===")
    info = api.get_model_info()
    print(f"Model Info: {info}")


if __name__ == "__main__":
    main() 