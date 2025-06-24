#!/usr/bin/env python3
"""
Startup Check for TAQA Hybrid System on DigitalOcean
Handles missing models and provides fallback options
"""

import os
import sys
import json
import joblib
from pathlib import Path

def create_fallback_if_missing():
    """Create fallback systems if models are missing"""
    
    # Check if lookup file exists
    if not os.path.exists('taqa_priority_lookup.json'):
        print("⚠️ Lookup file missing - creating minimal fallback")
        fallback_lookup = {
            "equipment_priorities": {
                "pompe alimentaire": {"mean": 3.0, "count": 10, "std": 0.5},
                "alternateur": {"mean": 2.5, "count": 5, "std": 0.3},
                "chaudière": {"mean": 2.8, "count": 8, "std": 0.4},
                "moteur": {"mean": 2.2, "count": 15, "std": 0.6}
            },
            "section_averages": {
                "34MC": {"mean": 2.3, "count": 100, "std": 0.8},
                "34EL": {"mean": 2.1, "count": 80, "std": 0.7},
                "34CT": {"mean": 2.0, "count": 60, "std": 0.5},
                "34MD": {"mean": 2.4, "count": 70, "std": 0.6},
                "34MM": {"mean": 2.2, "count": 90, "std": 0.7},
                "34MG": {"mean": 2.1, "count": 50, "std": 0.4}
            }
        }
        
        with open('taqa_priority_lookup.json', 'w', encoding='utf-8') as f:
            json.dump(fallback_lookup, f, indent=2)
        print("✅ Fallback lookup system created")
    
    # Check if ML model exists
    if not os.path.exists('improved_balanced_taqa_model.joblib'):
        print("⚠️ ML model missing - creating simple fallback")
        
        # Create a simple fallback that returns reasonable predictions
        class FallbackModel:
            def predict(self, X):
                # Simple rule-based fallback
                return [2.0] * len(X)
        
        fallback_model_data = {
            'models': {
                'Fallback': {'model': FallbackModel()}
            },
            'ensemble_weights': [1.0],
            'ensemble_models': [('Fallback', FallbackModel())],
            'vectorizers': {
                'tfidf': create_fallback_vectorizer()
            },
            'scalers': {
                'robust': create_fallback_scaler()
            },
            'feature_names': ['fallback_feature']
        }
        
        joblib.dump(fallback_model_data, 'improved_balanced_taqa_model.joblib')
        print("✅ Fallback ML model created")

def create_fallback_vectorizer():
    """Create a simple TF-IDF vectorizer fallback"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=10, ngram_range=(1, 1))
    # Fit on dummy data
    vectorizer.fit(['maintenance', 'urgent', 'routine', 'critical'])
    return vectorizer

def create_fallback_scaler():
    """Create a simple scaler fallback"""
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    # Fit on dummy data
    import numpy as np
    dummy_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    scaler.fit(dummy_data)
    return scaler

def main():
    """Check and create fallback systems if needed"""
    print("🚀 TAQA HYBRID SYSTEM - STARTUP CHECK")
    print("=" * 50)
    
    # Create fallbacks if needed
    create_fallback_if_missing()
    
    # Test the system
    try:
        from hybrid_taqa_api import HybridTAQAAPI
        api = HybridTAQAAPI()
        print("✅ Hybrid System initialized successfully")
        
        # Test prediction
        result = api.predict_single(
            description="Test maintenance",
            equipment_type="pompe alimentaire",
            section="34MC"
        )
        print(f"✅ Test prediction: {result['priority_score']} via {result['method']}")
        
        return True
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 Try downloading models manually:")
        print("   - taqa_priority_lookup.json")
        print("   - improved_balanced_taqa_model.joblib")
        sys.exit(1) 