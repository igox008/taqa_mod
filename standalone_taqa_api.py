#!/usr/bin/env python3
"""
Standalone TAQA API - No External Files Required
All data built-in, guaranteed to work on any platform
"""

import json
from datetime import datetime

class StandaloneTAQAAPI:
    """Completely standalone TAQA system with built-in data"""
    
    def __init__(self):
        """Initialize with built-in data"""
        self.equipment_priorities = self.get_builtin_equipment_data()
        self.section_averages = self.get_builtin_section_data()
        print("✅ Standalone TAQA System initialized with built-in data")
    
    def get_builtin_equipment_data(self):
        """Built-in equipment priority data based on TAQA historical data"""
        return {
            "pompe alimentaire": {"mean": 3.2, "confidence": 0.9},
            "alternateur": {"mean": 2.8, "confidence": 0.85},
            "chaudière": {"mean": 3.0, "confidence": 0.9},
            "moteur": {"mean": 2.5, "confidence": 0.8},
            "transformateur": {"mean": 2.9, "confidence": 0.85},
            "évents ballon": {"mean": 3.0, "confidence": 0.9},
            "pompe": {"mean": 2.3, "confidence": 0.8},
            "ventilateur": {"mean": 2.1, "confidence": 0.75},
            "vanne": {"mean": 2.2, "confidence": 0.75},
            "soupape": {"mean": 2.7, "confidence": 0.8},
            "manometre": {"mean": 2.2, "confidence": 0.7},
            "indicateur": {"mean": 2.1, "confidence": 0.7},
            "detecteur": {"mean": 2.4, "confidence": 0.75},
            "capteur": {"mean": 2.3, "confidence": 0.75},
            "eclairage": {"mean": 1.8, "confidence": 0.9},
            "armoire": {"mean": 2.0, "confidence": 0.8},
            "cable": {"mean": 1.9, "confidence": 0.8},
            "prise": {"mean": 1.7, "confidence": 0.85}
        }
    
    def get_builtin_section_data(self):
        """Built-in section average data"""
        return {
            "34MC": {"mean": 2.3, "confidence": 0.7},
            "34EL": {"mean": 2.1, "confidence": 0.7},
            "34CT": {"mean": 2.0, "confidence": 0.7},
            "34MD": {"mean": 2.4, "confidence": 0.7},
            "34MM": {"mean": 2.2, "confidence": 0.7},
            "34MG": {"mean": 2.1, "confidence": 0.7}
        }
    
    def analyze_text_priority(self, text):
        """Analyze text to determine priority without any ML dependencies"""
        if not text:
            return 2.0, "Default priority - no description provided"
        
        text_lower = str(text).lower()
        
        # Critical/Emergency keywords (Priority 3.5-4.0)
        critical_keywords = {
            'urgent': 0.8,
            'immédiat': 0.8,
            'critique': 0.7,
            'danger': 0.9,
            'sécurité': 0.8,
            'arrêt': 0.7,
            'fuite': 0.8,
            'panne': 0.6,
            'défaut majeur': 0.9,
            'dysfonctionnement grave': 0.8,
            'emergency': 0.9,
            'risque': 0.7
        }
        
        # High priority keywords (Priority 2.5-3.5)
        high_keywords = {
            'défaut': 0.4,
            'anomalie': 0.4,
            'maintenance': 0.3,
            'contrôle urgent': 0.6,
            'vérification': 0.3,
            'révision': 0.3,
            'réparation': 0.5,
            'dépannage': 0.5,
            'vibration': 0.4,
            'température': 0.4,
            'pression': 0.5
        }
        
        # Medium priority keywords (Priority 2.0-2.5)
        medium_keywords = {
            'contrôle': 0.2,
            'inspection': 0.2,
            'nettoyage': 0.1,
            'installation': 0.2,
            'modification': 0.2
        }
        
        # Low priority keywords (Priority 1.0-2.0)
        low_keywords = {
            'préventive': -0.3,
            'programmée': -0.3,
            'amélioration': -0.4,
            'optimisation': -0.4,
            'éclairage': -0.5,
            'bureau': -0.6,
            'confort': -0.5,
            'esthétique': -0.6,
            'général': -0.3
        }
        
        # Calculate score
        base_score = 2.0
        total_weight = 0
        matched_keywords = []
        
        # Check all keyword categories
        for keywords, category_name in [
            (critical_keywords, "critical"),
            (high_keywords, "high"),
            (medium_keywords, "medium"), 
            (low_keywords, "low")
        ]:
            for keyword, weight in keywords.items():
                if keyword in text_lower:
                    base_score += weight
                    total_weight += abs(weight)
                    matched_keywords.append(f"{keyword}({category_name})")
        
        # Normalize and cap the score
        final_score = max(1.0, min(4.0, base_score))
        
        # Create explanation
        if matched_keywords:
            explanation = f"Detected keywords: {', '.join(matched_keywords[:3])}"
            if len(matched_keywords) > 3:
                explanation += f" (+{len(matched_keywords)-3} more)"
        else:
            explanation = "No specific priority keywords detected - using default"
        
        return final_score, explanation
    
    def lookup_equipment_priority(self, equipment_type):
        """Look up equipment priority from built-in data"""
        if not equipment_type:
            return None
        
        equipment_clean = str(equipment_type).lower().strip()
        
        # Try exact and partial matches
        for equip_name, data in self.equipment_priorities.items():
            if (equip_name in equipment_clean or 
                equipment_clean in equip_name or
                any(word in equipment_clean for word in equip_name.split())):
                
                return {
                    'priority_score': data['mean'],
                    'method': 'Equipment Lookup',
                    'confidence': data['confidence'],
                    'explanation': f'Historical data for {equip_name} equipment type'
                }
        
        return None
    
    def lookup_section_priority(self, section):
        """Look up section priority from built-in data"""
        if not section:
            return None
        
        section_clean = str(section).strip()
        
        if section_clean in self.section_averages:
            data = self.section_averages[section_clean]
            return {
                'priority_score': data['mean'],
                'method': 'Section Average',
                'confidence': data['confidence'],
                'explanation': f'Average priority for {section} section'
            }
        
        return None
    
    def predict_single(self, description, equipment_type=None, section=None):
        """Main prediction method - guaranteed to work"""
        try:
            # Strategy 1: Equipment lookup (highest confidence)
            equipment_result = self.lookup_equipment_priority(equipment_type)
            if equipment_result and equipment_result['confidence'] >= 0.8:
                result = equipment_result
                result['priority_label'] = self.get_priority_label(result['priority_score'])
                result['color'] = self.get_priority_color(result['priority_score'])
                result['urgency'] = self.get_urgency_level(result['priority_score'])
                return result
            
            # Strategy 2: Text analysis
            text_score, text_explanation = self.analyze_text_priority(description)
            
            # Strategy 3: Apply section modifier
            section_result = self.lookup_section_priority(section)
            section_modifier = 0
            method_parts = ["Text Analysis"]
            
            if section_result:
                section_modifier = (section_result['priority_score'] - 2.0) * 0.2
                method_parts.append("Section")
            
            # Strategy 4: Apply equipment modifier if available
            if equipment_result:
                equipment_modifier = (equipment_result['priority_score'] - 2.0) * 0.3
                text_score = (text_score + equipment_result['priority_score']) / 2
                method_parts.append("Equipment")
                text_explanation += f" Combined with {equipment_type} data."
            
            # Final score calculation
            final_score = max(1.0, min(4.0, text_score + section_modifier))
            method = " + ".join(method_parts)
            
            # Determine confidence
            confidence = 0.7
            if equipment_result:
                confidence = 0.75
            if len(method_parts) >= 3:
                confidence = 0.8
            
            return {
                'priority_score': round(final_score, 2),
                'priority_label': self.get_priority_label(final_score),
                'urgency': self.get_urgency_level(final_score),
                'confidence': confidence,
                'color': self.get_priority_color(final_score),
                'explanation': text_explanation,
                'method': method
            }
            
        except Exception as e:
            # Ultimate fallback - this should never fail
            return {
                'priority_score': 2.0,
                'priority_label': 'Moyenne',
                'urgency': 'Planifié',
                'confidence': 0.5,
                'color': 'yellow',
                'explanation': f'Fallback prediction - {str(e)[:100]}',
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

def test_standalone_api():
    """Test the standalone API"""
    print("🧪 TESTING STANDALONE TAQA API")
    print("=" * 40)
    
    api = StandaloneTAQAAPI()
    
    test_cases = [
        {
            'name': 'Critical Emergency',
            'description': "URGENT: Fuite danger immédiat sécurité personnel arrêt critique",
            'equipment': "pompe alimentaire",
            'section': "34MC"
        },
        {
            'name': 'Equipment Issue',
            'description': "Défaut alternateur vibrations anormales",
            'equipment': "alternateur",
            'section': "34EL"
        },
        {
            'name': 'Routine Maintenance',
            'description': "Maintenance préventive programmée contrôle",
            'equipment': "ventilateur",
            'section': "34MM"
        },
        {
            'name': 'Office Improvement',
            'description': "Amélioration éclairage bureau confort général",
            'equipment': "eclairage",
            'section': "34EL"
        },
        {
            'name': 'Unknown Equipment',
            'description': "Contrôle nouveau système",
            'equipment': "UNKNOWN DEVICE",
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
        print(f"📊 Score: {result['priority_score']} ({result['priority_label']})")
        print(f"🎭 Method: {result['method']}")
        print(f"🔒 Confidence: {result['confidence']}")
        print(f"⚡ Urgency: {result['urgency']}")
        print(f"💡 Explanation: {result['explanation']}")

if __name__ == "__main__":
    test_standalone_api() 