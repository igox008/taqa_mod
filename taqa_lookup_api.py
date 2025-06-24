
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import re

class TAQALookupAPI:
    """
    TAQA Priority API using lookup tables based on actual historical patterns
    Much more accurate than ML for TAQA's equipment-based priority system
    """
    
    def __init__(self, lookup_file: str = 'taqa_priority_lookup.json'):
        with open(lookup_file, 'r', encoding='utf-8') as f:
            self.lookup_data = json.load(f)
        print("TAQA Lookup-based priority system loaded")
    
    def predict_single(self, 
                      description: str, 
                      equipment_type: str = "", 
                      section: str = "", 
                      status: str = "Terminé",
                      detection_date: Optional[str] = None) -> Dict:
        """Predict priority using lookup tables"""
        
        if detection_date is None:
            detection_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Try equipment-based lookup first (most accurate)
        if equipment_type and equipment_type in self.lookup_data['equipment_priorities']:
            equipment_data = self.lookup_data['equipment_priorities'][equipment_type]
            predicted_score = equipment_data['mean']
            confidence = "High"
            explanation = f"Based on {int(equipment_data['count'])} historical cases for this equipment type"
            method = "Equipment Lookup"
            
        # Fall back to section-based lookup
        elif section and section in self.lookup_data['section_priorities']:
            section_data = self.lookup_data['section_priorities'][section]
            predicted_score = section_data['mean'] 
            confidence = "Medium"
            explanation = f"Based on {int(section_data['count'])} historical cases for section {section}"
            method = "Section Lookup"
            
        # Last resort: keyword analysis + default
        else:
            predicted_score = self._analyze_keywords(description)
            confidence = "Low"
            explanation = "Based on keyword analysis and historical averages"
            method = "Keyword Analysis"
        
        # Ensure valid range
        predicted_score = max(1.0, min(4.0, predicted_score))
        
        return {
            'priority_score': round(predicted_score, 2),
            'priority_range': [1.0, 4.0],
            'explanation': explanation,
            'confidence': confidence,
            'method': method,
            'timestamp': datetime.now().isoformat(),
            'input_data': {
                'description': description,
                'equipment_type': equipment_type,
                'section': section,
                'status': status,
                'detection_date': detection_date
            }
        }
    
    def _analyze_keywords(self, description: str) -> float:
        """Analyze keywords to estimate priority"""
        if not description:
            return self.lookup_data['default_priority']
        
        desc_lower = description.lower()
        scores = []
        
        # Check against each priority level's common words
        for priority_level in ['priority_1.0', 'priority_2.0', 'priority_3.0', 'priority_4.0']:
            if priority_level in self.lookup_data['keyword_analysis']:
                keywords = self.lookup_data['keyword_analysis'][priority_level]
                matches = sum(1 for word in keywords if word in desc_lower)
                if matches > 0:
                    priority_value = float(priority_level.split('_')[1])
                    scores.append((priority_value, matches))
        
        if scores:
            # Weight by number of matches
            weighted_score = sum(score * matches for score, matches in scores) / sum(matches for _, matches in scores)
            return weighted_score
        else:
            return self.lookup_data['default_priority']
    
    def health_check(self) -> Dict:
        """Health check for lookup system"""
        return {
            'status': 'healthy',
            'method': 'lookup_based',
            'equipment_types_loaded': len(self.lookup_data['equipment_priorities']),
            'sections_loaded': len(self.lookup_data['section_priorities']),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_model_info(self) -> Dict:
        """Get lookup system info"""
        return {
            'model_info': {
                'type': 'Lookup-based System',
                'equipment_types': len(self.lookup_data['equipment_priorities']),
                'sections': len(self.lookup_data['section_priorities']),
                'priority_distribution': self.lookup_data['priority_distribution'],
                'accuracy': 'High for known equipment types'
            },
            'timestamp': datetime.now().isoformat()
        }

# Test the lookup API
if __name__ == "__main__":
    api = TAQALookupAPI()
    
    test_cases = [
        ("Test description", "Moteur pompe alimentaire 10", "34MC"),
        ("Another test", "éclairage et prise de courant", "34EL"),
        ("Steam leak", "unknown equipment", "34MC"),
        ("General maintenance", "", "34CT")
    ]
    
    print("=== TESTING LOOKUP API ===")
    for desc, equip, section in test_cases:
        result = api.predict_single(desc, equip, section)
        print(f"Equipment: {equip[:30]:<30} | Score: {result['priority_score']:<4} | Method: {result['method']}")
        print(f"Explanation: {result['explanation']}")
        print("-" * 80)
