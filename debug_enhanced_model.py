from enhanced_api_wrapper import EnhancedAnomalyRegressorAPI
import numpy as np
import pandas as pd

def debug_enhanced_model():
    print("=== DEBUGGING ENHANCED MODEL ===")
    
    # Load the model
    api = EnhancedAnomalyRegressorAPI('enhanced_anomaly_priority_regressor.joblib')
    
    # Test different cases
    test_cases = [
        ('SAFETY: Fuite importante de vapeur - arret immediat requis', 'TURBINE', '34MC'),
        ('urgent: defaut critique alternateur', 'ALTERNATEUR', '34EL'), 
        ('Prevoir controle mesure de temperature', 'MOTEUR', '34CT'),
        ('Amelioration eclairage salle technique', 'ECLAIRAGE', '34EL'),
        ('inspection generale equipement', 'DIVERS', '34MG')
    ]
    
    print("Testing different descriptions:")
    for desc, equip, section in test_cases:
        result = api.predict_single(desc, equip, section)
        score = result.get('priority_score', 'ERROR')
        print(f"Description: {desc[:50]}...")
        print(f"Score: {score}")
        print(f"Explanation: {result.get('explanation', 'None')}")
        print("-" * 60)
    
    # Let's also check what features are being generated
    print("\n=== FEATURE ANALYSIS ===")
    desc = "SAFETY: Fuite importante de vapeur - arret immediat requis"
    
    # Check text features
    text_features = api._improved_text_features(desc)
    print("Text features for safety description:")
    for key, value in text_features.items():
        print(f"  {key}: {value}")
    
    # Check if TF-IDF is working
    tfidf_features = api.tfidf.transform([desc]).toarray()
    print(f"\nTF-IDF features shape: {tfidf_features.shape}")
    print(f"TF-IDF non-zero features: {np.count_nonzero(tfidf_features)}")
    
    # Create sample data and check shape
    sample_data = api._create_enhanced_sample_data(desc, "TURBINE", "34MC", "En cours", "2024-06-24")
    print(f"\nSample data shape: {sample_data.shape}")
    print(f"Sample data non-zero features: {np.count_nonzero(sample_data)}")
    
    # Check if model pipeline exists
    print(f"\nModel pipeline loaded: {api.pipeline is not None}")
    if api.pipeline:
        print(f"Pipeline steps: {api.pipeline.steps}")

if __name__ == "__main__":
    debug_enhanced_model()
