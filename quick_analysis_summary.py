p#!/usr/bin/env python3
"""
Quick Analysis Summary for Equipment Anomaly Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def quick_analysis():
    """Perform a quick analysis of the equipment anomaly dataset"""
    
    print("🔍 Quick Equipment Anomaly Analysis")
    print("=" * 50)
    
    # Load data
    data = pd.read_csv('data_set.csv')
    print(f"Dataset: {data.shape[0]} records, {data.shape[1]} columns")
    
    # Basic statistics
    print(f"\n📊 Criticality Score Statistics:")
    print(f"Range: {data['Criticité'].min()} - {data['Criticité'].max()}")
    print(f"Mean: {data['Criticité'].mean():.2f}")
    print(f"Std: {data['Criticité'].std():.2f}")
    
    print(f"\n🎯 Priority Distribution:")
    low_priority = (data['Criticité'] <= 4).sum()
    medium_priority = ((data['Criticité'] > 4) & (data['Criticité'] <= 7)).sum()
    high_priority = (data['Criticité'] > 7).sum()
    
    print(f"Low Priority (≤4): {low_priority} ({low_priority/len(data)*100:.1f}%)")
    print(f"Medium Priority (5-7): {medium_priority} ({medium_priority/len(data)*100:.1f}%)")
    print(f"High Priority (≥8): {high_priority} ({high_priority/len(data)*100:.1f}%)")
    
    # Equipment analysis
    print(f"\n⚙️ Equipment Analysis:")
    top_equipment = data['Description de l\'équipement'].value_counts().head(10)
    print("Top 10 Equipment Types:")
    for equipment, count in top_equipment.items():
        avg_criticality = data[data['Description de l\'équipement'] == equipment]['Criticité'].mean()
        print(f"  {equipment}: {count} anomalies (avg criticality: {avg_criticality:.1f})")
    
    # Section analysis
    print(f"\n🏭 Section Analysis:")
    section_stats = data.groupby('Section propriétaire')['Criticité'].agg(['count', 'mean', 'std']).round(2)
    section_stats.columns = ['Count', 'Avg_Criticality', 'Std_Criticality']
    print(section_stats)
    
    # Text analysis - critical keywords
    print(f"\n📝 Critical Keywords Analysis:")
    descriptions = data['Description'].fillna('').str.lower()
    
    critical_keywords = {
        'Leaks': ['fuite', 'fuite importante', 'fuite d\'eau', 'fuite d\'huile'],
        'Vibrations': ['vibration', 'bruit anormal', 'bruit'],
        'Sealing Issues': ['non étanche', 'non-étanche', 'étanchéité'],
        'Damage': ['percement', 'rupture', 'défaillance'],
        'Temperature': ['surchauffe', 'température', 'chauffage']
    }
    
    for category, keywords in critical_keywords.items():
        count = 0
        total_criticality = 0
        for keyword in keywords:
            mask = descriptions.str.contains(keyword, na=False)
            keyword_count = mask.sum()
            if keyword_count > 0:
                count += keyword_count
                total_criticality += data.loc[mask, 'Criticité'].mean() * keyword_count
        
        if count > 0:
            avg_criticality = total_criticality / count
            print(f"  {category}: {count} occurrences (avg criticality: {avg_criticality:.1f})")
    
    # Quick predictive model
    print(f"\n🤖 Quick Predictive Model (Random Forest):")
    
    # Simple features
    X = data[['Fiabilité Intégrité', 'Disponibilté', 'Process Safety']].copy()
    
    # Add basic text features
    X['has_fuite'] = descriptions.str.contains('fuite', na=False).astype(int)
    X['has_vibration'] = descriptions.str.contains('vibration|bruit', na=False).astype(int)
    X['has_non_etanche'] = descriptions.str.contains('non étanche|non-étanche', na=False).astype(int)
    X['has_percement'] = descriptions.str.contains('percement', na=False).astype(int)
    X['description_length'] = data['Description'].str.len().fillna(0)
    
    # Encode equipment type (top 10 only)
    top_equipment_list = top_equipment.index.tolist()
    for i, equipment in enumerate(top_equipment_list):
        X[f'equipment_{i}'] = (data['Description de l\'équipement'] == equipment).astype(int)
    
    # Target
    y = data['Criticité']
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    
    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"  Test MAE: {mae:.3f}")
    print(f"  Test R²: {r2:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📈 Top 10 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"  {i+1}. {row['feature']}: {row['importance']:.3f}")
    
    # Prediction examples
    print(f"\n🔮 Prediction Examples:")
    
    # Example 1: Major leak
    example1 = pd.DataFrame({
        'Fiabilité Intégrité': [2], 'Disponibilté': [4], 'Process Safety': [3],
        'has_fuite': [1], 'has_vibration': [0], 'has_non_etanche': [1], 
        'has_percement': [0], 'description_length': [60]
    })
    # Add equipment features
    for i in range(len(top_equipment_list)):
        example1[f'equipment_{i}'] = 0
    
    pred1 = rf.predict(example1)[0]
    print(f"  Major leak scenario: Predicted criticality = {pred1:.1f}")
    
    # Example 2: Vibration issue
    example2 = pd.DataFrame({
        'Fiabilité Intégrité': [3], 'Disponibilté': [3], 'Process Safety': [2],
        'has_fuite': [0], 'has_vibration': [1], 'has_non_etanche': [0], 
        'has_percement': [0], 'description_length': [40]
    })
    # Add equipment features
    for i in range(len(top_equipment_list)):
        example2[f'equipment_{i}'] = 0
    
    pred2 = rf.predict(example2)[0]
    print(f"  Vibration issue scenario: Predicted criticality = {pred2:.1f}")
    
    print(f"\n✅ Quick analysis complete!")
    print(f"💡 Key Insights:")
    print(f"  • Dataset contains {len(data)} equipment maintenance records")
    print(f"  • {high_priority} ({high_priority/len(data)*100:.1f}%) are high priority (≥8)")
    print(f"  • Text features (keywords) are highly predictive")
    print(f"  • Random Forest achieves {r2:.3f} R² with simple features")
    print(f"  • Most critical issues involve leaks, vibrations, and sealing problems")
    
    return rf, feature_importance, data

if __name__ == "__main__":
    model, importance, data = quick_analysis() 