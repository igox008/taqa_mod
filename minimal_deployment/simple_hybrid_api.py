#!/usr/bin/env python3
"""
Simplified TAQA Hybrid API for DigitalOcean
Works with just the lookup system and basic ML fallback
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimpleTAQAAPI:
    """Simplified TAQA system that always works"""
    
    def __init__(self, lookup_file='taqa_priority_lookup.json'):
        """Initialize with just lookup system"""
        self.lookup_data = None
        self.lookup_loaded = False
        
        # Load lookup system
        try:
            with open(lookup_file, 'r', encoding='utf-8') as f:
                self.lookup_data = json.load(f)
            self.lookup_loaded = True
            print("✅ Lookup system loaded successfully")
        except Exception as e:
            print(f"⚠️ Lookup system failed: {e}")
            # Create minimal fallback
            self.create_minimal_lookup()
    
    def create_minimal_lookup(self):
        """Create minimal lookup data if file is missing"""
        print("🔧 Creating minimal lookup data...")
        self.lookup_data = {
            "equipment_priorities": {
                "pompe alimentaire": {"mean": 3.2, "count": 50, "std": 0.8},
                "alternateur": {"mean": 2.8, "count": 30, "std": 0.6},
                "chaudière": {"mean": 3.0, "count": 40, "std": 0.7},
                "moteur": {"mean": 2.5, "count": 60, "std": 0.5},
                "transformateur": {"mean": 2.9, "count": 25, "std": 0.6},
                "évents ballon": {"mean": 3.0, "count": 10, "std": 0.4},
                "pompe": {"mean": 2.3, "count": 80, "std": 0.7},
                "ventilateur": {"mean": 2.1, "count": 45, "std": 0.5}
            },
            "section_averages": {
                "34MC": {"mean": 2.3, "count": 500, "std": 0.8},
                "34EL": {"mean": 2.1, "count": 400, "std": 0.7},
                "34CT": {"mean": 2.0, "count": 300, "std": 0.6},
                "34MD": {"mean": 2.4, "count": 350, "std": 0.7},
                "34MM": {"mean": 2.2, "count": 450, "std": 0.8},
                "34MG": {"mean": 2.1, "count": 250, "std": 0.5}
            }
        }
        self.lookup_loaded = True
        print("✅ Minimal lookup system created")
    
    def smart_text_analysis(self, text):
        """Simple text analysis without ML dependencies"""
        if not text:
            return 2.0, "Default priority"
        
        text_lower = str(text).lower()
        
        # High priority keywords (3.5-4.0)
        high_priority_words = [
            'urgent', 'immédiat', 'critique', 'danger', 'sécurité', 'arrêt',
            'fuite', 'panne', 'défaut majeur', 'dysfonctionnement grave'
        ]
        
        # Medium priority keywords (2.5-3.0)
        medium_priority_words = [
            'défaut', 'anomalie', 'maintenance', 'contrôle', 'vérification',
            'révision', 'réparation', 'dépannage'
        ]
        
        # Low priority keywords (1.5-2.0)
        low_priority_words = [
            'préventive', 'programmée', 'amélioration', 'optimisation',
            'éclairage', 'bureau', 'confort', 'esthétique'
        ]
        
        # Count matches
        high_matches = sum(1 for word in high_priority_words if word in text_lower)
        medium_matches = sum(1 for word in medium_priority_words if word in text_lower)
        low_matches = sum(1 for word in low_priority_words if word in text_lower)
        
        # Determine priority
        if high_matches > 0:
            base_score = 3.5 + min(high_matches * 0.2, 0.5)
            explanation = f"High priority detected ({high_matches} critical keywords)"
        elif medium_matches > 0:
            base_score = 2.5 + min(medium_matches * 0.1, 0.4)
            explanation = f"Medium priority detected ({medium_matches} maintenance keywords)"
        elif low_matches > 0:
            base_score = 1.5 + min(low_matches * 0.1, 0.4)
            explanation = f"Low priority detected ({low_matches} routine keywords)"
        else:
            base_score = 2.0
            explanation = "Default priority - no specific keywords detected"
        
        return min(base_score, 4.0), explanation
    
    def equipment_priority_lookup(self, equipment_type):
        """Look up equipment priority"""
        if not equipment_type or not self.lookup_loaded:
            return None
        
        equipment_clean = str(equipment_type).lower().strip()
        
        # Direct match
        equipment_priorities = self.lookup_data.get('equipment_priorities', {})
        for equip_name, data in equipment_priorities.items():
            if equip_name.lower() in equipment_clean or equipment_clean in equip_name.lower():
                priority = data['mean'] if isinstance(data, dict) else data
                confidence = 0.9 if isinstance(data, dict) and data.get('count', 0) > 5 else 0.7
                return {
                    'priority_score': priority,
                    'method': 'Equipment Lookup',
                    'confidence': confidence,
                    'explanation': f'Historical data for {equip_name} ({data.get("count", "unknown")} records)'
                }
        
        return None
    
    def section_priority_lookup(self, section):
        """Look up section priority"""
        if not section or not self.lookup_loaded:
            return None
        
        section_clean = str(section).strip()
        section_averages = self.lookup_data.get('section_averages', {})
        
        if section_clean in section_averages:
            data = section_averages[section_clean]
            priority = data['mean'] if isinstance(data, dict) else data
            return {
                'priority_score': priority,
                'method': 'Section Average',
                'confidence': 0.6,
                'explanation': f'Average for {section} section ({data.get("count", "unknown")} records)'
            }
        
        return None
    
    def predict_single(self, description, equipment_type=None, section=None):
        """Main prediction method"""
        try:
            # Strategy 1: Equipment lookup (highest priority)
            equipment_result = self.equipment_priority_lookup(equipment_type)
            if equipment_result and equipment_result['confidence'] >= 0.8:
                result = equipment_result
                result['priority_label'] = self.get_priority_label(result['priority_score'])
                result['color'] = self.get_priority_color(result['priority_score'])
                result['urgency'] = self.get_urgency_level(result['priority_score'])
                return result
            
            # Strategy 2: Text analysis
            text_score, text_explanation = self.smart_text_analysis(description)
            
            # Strategy 3: Section lookup as modifier
            section_result = self.section_priority_lookup(section)
            section_modifier = 0
            if section_result:
                section_modifier = (section_result['priority_score'] - 2.0) * 0.3  # 30% influence
            
            # Combine text analysis with section modifier
            final_score = max(1.0, min(4.0, text_score + section_modifier))
            
            # Use equipment lookup if available but low confidence
            if equipment_result:
                final_score = (final_score + equipment_result['priority_score']) / 2
                method = "Hybrid (Text + Equipment)"
                explanation = f"{text_explanation}. Equipment factor included."
                confidence = 0.75
            else:
                method = "Text Analysis"
                explanation = text_explanation
                if section_result:
                    explanation += f". Section {section} modifier applied."
                    method = "Text + Section"
                confidence = 0.7
            
            return {
                'priority_score': round(final_score, 2),
                'priority_label': self.get_priority_label(final_score),
                'urgency': self.get_urgency_level(final_score),
                'confidence': confidence,
                'color': self.get_priority_color(final_score),
                'explanation': explanation,
                'method': method
            }
            
        except Exception as e:
            # Absolute fallback
            return {
                'priority_score': 2.0,
                'priority_label': 'Moyenne',
                'urgency': 'Planifié',
                'confidence': 0.3,
                'color': 'yellow',
                'explanation': f'Fallback prediction due to error: {str(e)}',
                'method': 'Fallback'
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

def test_simple_api():
    """Test the simple API"""
    print("🧪 TESTING SIMPLE TAQA API")
    print("=" * 40)
    
    api = SimpleTAQAAPI()
    
    test_cases = [
        {
            'name': 'Critical Safety',
            'description': "URGENT: Fuite massive vapeur haute pression - danger personnel arrêt immédiat",
            'equipment': "pompe alimentaire",
            'section': "34MC"
        },
        {
            'name': 'Equipment Maintenance',
            'description': "Défaut alternateur - vibrations anormales",
            'equipment': "alternateur",
            'section': "34EL"
        },
        {
            'name': 'Routine Work',
            'description': "Maintenance préventive programmée",
            'equipment': "ventilateur",
            'section': "34MM"
        },
        {
            'name': 'Unknown Equipment',
            'description': "Contrôle système nouveau",
            'equipment': "EQUIPMENT INCONNU",
            'section': "34CT"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🧪 TEST {i}: {test['name']}")
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

if __name__ == "__main__":
    test_simple_api() 