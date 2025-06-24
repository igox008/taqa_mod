from taqa_lookup_api import TAQALookupAPI
import pandas as pd

def test_lookup_accuracy():
    """Test the lookup system accuracy against actual TAQA data"""
    
    print("=" * 60)
    print("TESTING LOOKUP SYSTEM ACCURACY")
    print("=" * 60)
    
    # Load the lookup API
    api = TAQALookupAPI()
    
    # Load actual data for testing
    df = pd.read_csv('data_set.csv')
    df_test = df.dropna(subset=['Priorité', 'Description', 'Description equipement']).sample(100, random_state=42)
    
    print(f"Testing on {len(df_test)} random real TAQA cases...")
    
    correct_predictions = 0
    total_predictions = 0
    errors = []
    
    for _, row in df_test.iterrows():
        actual_priority = row['Priorité']
        
        result = api.predict_single(
            description=row['Description'],
            equipment_type=row['Description equipement'],
            section=row['Section proprietaire'] if pd.notna(row['Section proprietaire']) else '',
            status=row['Statut'] if pd.notna(row['Statut']) else 'Terminé'
        )
        
        predicted_priority = result['priority_score']
        error = abs(predicted_priority - actual_priority)
        method = result['method']
        
        # Consider prediction correct if within 0.5 points
        if error <= 0.5:
            correct_predictions += 1
            status = ""
        else:
            status = ""
        
        errors.append(error)
        total_predictions += 1
        
        if total_predictions <= 10:  # Show first 10 for detail
            print(f"{status} Predicted: {predicted_priority:.2f} | Actual: {actual_priority:.2f} | Error: {error:.2f} | Method: {method}")
            print(f"   Equipment: {row['Description equipement'][:50]}...")
            print(f"   Section: {row['Section proprietaire']}")
            print()
    
    # Calculate metrics
    accuracy = (correct_predictions / total_predictions) * 100
    mean_error = sum(errors) / len(errors)
    max_error = max(errors)
    
    print("=" * 60)
    print("LOOKUP SYSTEM RESULTS")
    print("=" * 60)
    
    print(f" Accuracy (0.5 points): {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")
    print(f" Mean Absolute Error: {mean_error:.3f}")
    print(f" Maximum Error: {max_error:.3f}")
    
    # Compare with previous models
    print(f"\n COMPARISON WITH PREVIOUS MODELS:")
    print(f"    Original model: Always predicted 3.0 (0% accuracy)")
    print(f"    Enhanced model: 66.7% accuracy")
    print(f"    Lookup system: {accuracy:.1f}% accuracy")
    
    if accuracy >= 80:
        rating = " EXCELLENT - Production Ready!"
    elif accuracy >= 70:
        rating = " GOOD - Acceptable for production"
    elif accuracy >= 60:
        rating = " FAIR - Needs improvement"
    else:
        rating = " POOR - Requires rework"
    
    print(f"\n Overall Rating: {rating}")
    
    # Test specific high-value cases
    print(f"\n=== TESTING HIGH-PRIORITY EQUIPMENT ===")
    
    high_priority_equipment = [
        "évents ballon chaudière",
        "POMPES DE DESCHARGE", 
        "SERVOMOTEUR CLAPET DE NON RETOUR"
    ]
    
    for equipment in high_priority_equipment:
        result = api.predict_single(
            description=f"Maintenance required for {equipment}",
            equipment_type=equipment,
            section="34MC"
        )
        print(f" {equipment}: {result['priority_score']:.2f} ({result['method']})")
    
    return accuracy, mean_error

if __name__ == "__main__":
    accuracy, error = test_lookup_accuracy()
    print(f"\nFinal Accuracy: {accuracy:.1f}% with MAE: {error:.3f}")
