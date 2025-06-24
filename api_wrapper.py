import joblib
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

class AnomalyRegressorAPI:
    """
    Simple API wrapper for the TAQA Anomaly Priority Regressor
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the API wrapper
        
        Args:
            model_path: Path to the trained model file (.joblib)
        """
        self.model = None
        self.model_info = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load a trained model from file
        
        Args:
            model_path: Path to the .joblib model file
        """
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")
            
            # Try to load model info
            try:
                with open('regressor_model_info.json', 'r') as f:
                    self.model_info = json.load(f)
            except FileNotFoundError:
                print("Model info file not found, using default settings")
                self.model_info = {'priority_range': [1.0, 5.0]}
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict_single(self, 
                      description: str, 
                      equipment_type: str = "", 
                      section: str = "", 
                      status: str = "Terminé",
                      detection_date: Optional[str] = None) -> Dict:
        """
        Predict priority for a single anomaly
        
        Args:
            description: Anomaly description (French text)
            equipment_type: Type of equipment affected
            section: Department/section responsible
            status: Current status of the anomaly
            detection_date: Date when anomaly was detected (optional)
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Use current date if not provided
        if detection_date is None:
            detection_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create sample data
        sample_data = self._create_sample_data(
            description, equipment_type, section, status, detection_date
        )
        
        # Make prediction
        try:
            prediction = self.model.predict(sample_data)[0]
            prediction = float(max(1.0, min(5.0, prediction)))
            
            return {
                'priority_score': prediction,
                'priority_range': self.model_info.get('priority_range', [1.0, 5.0]),
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
                'error': f"Prediction failed: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch(self, anomalies: List[Dict]) -> List[Dict]:
        """
        Predict priorities for multiple anomalies
        
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
                    'error': f"Prediction failed: {str(e)}",
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def _create_sample_data(self, description: str, equipment_type: str, section: str, 
                           status: str, detection_date: str) -> pd.DataFrame:
        """
        Create sample data for prediction
        
        Args:
            description: Anomaly description
            equipment_type: Equipment type
            section: Section/department
            status: Status
            detection_date: Detection date
            
        Returns:
            DataFrame with sample data
        """
        # Parse detection date
        try:
            detection_dt = pd.to_datetime(detection_date)
        except:
            detection_dt = pd.Timestamp.now()
        
        # Create base sample
        sample_data = pd.DataFrame({
            'Description': [description],
            'Description_clean': [description],
            'Equipment_Type': [equipment_type],
            'Section': [section],
            'Status': [status],
            'Date de detection de l\'anomalie': [detection_dt],
            'Year': [detection_dt.year],
            'Month': [detection_dt.month],
            'DayOfWeek': [detection_dt.dayofweek],
            'Description_Length': [len(description)],
            'Equipment_Length': [len(equipment_type)],
            'High_Priority_Keywords': [description.lower().count('urgent') + description.lower().count('safety')],
            'Medium_Priority_Keywords': [description.lower().count('prévoir') + description.lower().count('contrôle')],
            'Low_Priority_Keywords': [description.lower().count('amélioration')]
        })
        
        # Add equipment category features
        equipment_categories = ['pump', 'motor', 'valve', 'mill', 'fan', 'boiler', 'turbine', 'generator', 'transformer', 'electrical', 'sootblower', 'cleaner']
        for category in equipment_categories:
            sample_data[f'Equipment_{category}'] = [1 if category in equipment_type.lower() else 0]
        
        # Add section category features
        section_categories = ['mechanical', 'electrical', 'control', 'maintenance', 'mechanical_maintenance', 'mechanical_general']
        for category in section_categories:
            sample_data[f'Section_{category}'] = [1 if category in section.lower() else 0]
        
        return sample_data
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if self.model_info is None:
            return {'error': 'No model loaded'}
        
        return {
            'model_info': self.model_info,
            'model_loaded': self.model is not None,
            'timestamp': datetime.now().isoformat()
        }
    
    def health_check(self) -> Dict:
        """
        Perform a health check on the model
        
        Returns:
            Health status dictionary
        """
        try:
            # Test prediction with sample data
            test_result = self.predict_single(
                description="Test anomaly description",
                equipment_type="Test Equipment",
                section="34MC"
            )
            
            return {
                'status': 'healthy',
                'model_loaded': self.model is not None,
                'test_prediction_successful': 'error' not in test_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'model_loaded': self.model is not None,
                'timestamp': datetime.now().isoformat()
            }


# Example usage and testing
def main():
    """Example usage of the API wrapper"""
    
    # Initialize API (assuming model is already trained)
    api = AnomalyRegressorAPI()
    
    # Try to load the best model
    try:
        api.load_model('anomaly_priority_regressor_random_forest.joblib')
    except FileNotFoundError:
        print("Model file not found. Please train the model first using anomaly_classifier.py")
        return
    
    # Test single prediction
    print("=== Single Prediction Test ===")
    result = api.predict_single(
        description="SAFETY: Fuite importante de vapeur par tresse ramoneur IK 14",
        equipment_type="RAMONEUR LONG RETRACTABLE N°14 TYPE IK",
        section="34MC"
    )
    
    print(f"Prediction Result: {result}")
    
    # Test batch prediction
    print("\n=== Batch Prediction Test ===")
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
        }
    ]
    
    batch_results = api.predict_batch(test_anomalies)
    
    for result in batch_results:
        print(f"Anomaly {result['anomaly_id']}: Score {result.get('priority_score', 'Error')}")
    
    # Health check
    print("\n=== Health Check ===")
    health = api.health_check()
    print(f"Health Status: {health}")
    
    # Model info
    print("\n=== Model Information ===")
    info = api.get_model_info()
    print(f"Model Info: {info}")


if __name__ == "__main__":
    main() 