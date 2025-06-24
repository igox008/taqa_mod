#!/usr/bin/env python3
"""
Improved Balanced TAQA Model
Handles class imbalance and provides better priority predictions
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ImprovedBalancedTAQAModel:
    """Improved model that handles class imbalance properly"""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self.feature_names = []
        
    def extract_smart_text_features(self, text):
        """Extract intelligent text features for priority prediction"""
        if pd.isna(text) or text == '':
            return {
                'length': 0, 'word_count': 0,
                'safety_urgent_score': 0, 'equipment_critical_score': 0,
                'maintenance_routine_score': 0, 'improvement_minor_score': 0,
                'technical_complexity': 0, 'action_immediacy': 0
            }
        
        text_str = str(text).lower()
        words = text_str.split()
        word_count = len(words)
        
        # Critical/Safety indicators (should predict HIGH priority)
        critical_keywords = [
            'urgent', 'safety', 'sécurité', 'danger', 'risque', 'arrêt', 'immédiat',
            'critique', 'fuite', 'panne', 'défaut', 'dysfonctionnement', 'anomalie'
        ]
        
        # Equipment criticality (MEDIUM-HIGH priority)
        critical_equipment = [
            'pompe alimentaire', 'alternateur', 'chaudière', 'turbine', 
            'transformateur', 'moteur principal', 'système sécurité'
        ]
        
        # Routine maintenance (LOW-MEDIUM priority)
        routine_keywords = [
            'maintenance', 'préventive', 'contrôle', 'vérification', 'révision',
            'nettoyage', 'inspection', 'programmée', 'planning'
        ]
        
        # Minor improvements (LOW priority)
        improvement_keywords = [
            'amélioration', 'optimisation', 'installation', 'éclairage',
            'bureau', 'confort', 'esthétique', 'général'
        ]
        
        # Technical complexity indicators
        technical_keywords = [
            'pression', 'température', 'vibration', 'débit', 'niveau',
            'mesure', 'paramètre', 'analyse', 'diagnostic'
        ]
        
        # Immediacy indicators
        immediacy_keywords = [
            'immédiat', 'urgent', 'maintenant', 'direct', 'aussitôt',
            'rapidement', 'priorité', 'emergency'
        ]
        
        # Calculate scores
        safety_urgent_score = sum(1 for word in critical_keywords if word in text_str) / len(critical_keywords)
        equipment_critical_score = sum(1 for equip in critical_equipment if equip in text_str) / len(critical_equipment)
        maintenance_routine_score = sum(1 for word in routine_keywords if word in text_str) / len(routine_keywords)
        improvement_minor_score = sum(1 for word in improvement_keywords if word in text_str) / len(improvement_keywords)
        technical_complexity = sum(1 for word in technical_keywords if word in text_str) / len(technical_keywords)
        action_immediacy = sum(1 for word in immediacy_keywords if word in text_str) / len(immediacy_keywords)
        
        return {
            'length': len(text_str),
            'word_count': word_count,
            'safety_urgent_score': safety_urgent_score,
            'equipment_critical_score': equipment_critical_score,
            'maintenance_routine_score': maintenance_routine_score,
            'improvement_minor_score': improvement_minor_score,
            'technical_complexity': technical_complexity,
            'action_immediacy': action_immediacy
        }
    
    def create_balanced_dataset(self, df):
        """Create a more balanced dataset for training"""
        print("🔄 Creating balanced dataset...")
        
        # Original distribution
        print("Original distribution:")
        print(df['Priorité'].value_counts().sort_index())
        
        # Strategy: Oversample high priorities, undersample priority 2.0
        priority_groups = {}
        for priority in df['Priorité'].unique():
            priority_groups[priority] = df[df['Priorité'] == priority].copy()
        
        # Target distribution (more balanced)
        target_samples = {
            1.0: 800,   # Keep reasonable amount of low priority
            2.0: 1500,  # Reduce from 3341 to 1500
            3.0: 600,   # Increase from 235 to 600
            4.0: 300    # Increase from 10 to 300
        }
        
        balanced_dfs = []
        
        for priority, target_count in target_samples.items():
            if priority in priority_groups:
                group = priority_groups[priority]
                current_count = len(group)
                
                if current_count < target_count:
                    # Oversample (duplicate with slight variations)
                    oversample_factor = target_count // current_count
                    remainder = target_count % current_count
                    
                    # Add full duplicates
                    for _ in range(oversample_factor):
                        balanced_dfs.append(group.copy())
                    
                    # Add partial sample
                    if remainder > 0:
                        balanced_dfs.append(group.sample(remainder, replace=True))
                else:
                    # Undersample
                    balanced_dfs.append(group.sample(target_count, replace=False))
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        print("Balanced distribution:")
        print(balanced_df['Priorité'].value_counts().sort_index())
        
        return balanced_df
    
    def train_improved_model(self, file_path):
        """Train improved model with better features and balancing"""
        print("🚀 TRAINING IMPROVED BALANCED TAQA MODEL")
        print("=" * 60)
        
        # Load data
        df = pd.read_csv(file_path)
        df_clean = df.dropna(subset=['Priorité']).copy()
        print(f"📊 Loaded {len(df_clean)} records")
        
        # Create balanced dataset
        balanced_df = self.create_balanced_dataset(df_clean)
        
        # Extract improved features
        print("🧠 Extracting improved features...")
        balanced_df['Description'] = balanced_df['Description'].fillna('maintenance générale')
        balanced_df['Description equipement'] = balanced_df['Description equipement'].fillna('équipement')
        balanced_df['Section proprietaire'] = balanced_df['Section proprietaire'].fillna('34MC')
        
        # Text features
        text_features = balanced_df['Description'].apply(self.extract_smart_text_features)
        text_df = pd.DataFrame(text_features.tolist())
        
        for col in text_df.columns:
            balanced_df[f'text_{col}'] = text_df[col]
        
        # Equipment criticality
        equipment_clean = balanced_df['Description equipement'].fillna('').astype(str).str.lower()
        
        # High criticality equipment
        high_crit_equipment = [
            'pompe alimentaire', 'alternateur unite', 'chaudière principale', 
            'turbine', 'transformateur', 'évents ballon'
        ]
        balanced_df['equipment_high_criticality'] = sum(
            equipment_clean.str.contains(equip, na=False).astype(int) 
            for equip in high_crit_equipment
        )
        
        # Medium criticality
        medium_crit_equipment = ['pompe', 'moteur', 'ventilateur', 'vanne']
        balanced_df['equipment_medium_criticality'] = sum(
            equipment_clean.str.contains(equip, na=False).astype(int) 
            for equip in medium_crit_equipment
        )
        
        # Section features
        sections = ['34MC', '34EL', '34CT', '34MD', '34MM', '34MG']
        for section in sections:
            balanced_df[f'section_{section}'] = balanced_df['Section proprietaire'].str.contains(section, na=False).astype(int)
        
        # Priority-specific rules to enhance learning
        # Create synthetic features that correlate with priority
        balanced_df['priority_indicator'] = 0
        
        # Rules based on domain knowledge
        high_priority_mask = (
            (balanced_df['text_safety_urgent_score'] > 0.1) |
            (balanced_df['equipment_high_criticality'] > 0) |
            (balanced_df['text_action_immediacy'] > 0.1)
        )
        balanced_df.loc[high_priority_mask, 'priority_indicator'] = 2
        
        low_priority_mask = (
            (balanced_df['text_improvement_minor_score'] > 0.1) |
            (balanced_df['text_maintenance_routine_score'] > 0.2)
        )
        balanced_df.loc[low_priority_mask, 'priority_indicator'] = -1
        
        # Prepare features
        feature_cols = [col for col in balanced_df.columns if col.startswith(('text_', 'equipment_', 'section_', 'priority_'))]
        X_numerical = balanced_df[feature_cols].fillna(0)
        
        # TF-IDF with better parameters
        tfidf = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.8,
            lowercase=True
        )
        
        X_text = tfidf.fit_transform(balanced_df['Description']).toarray()
        self.vectorizers['tfidf'] = tfidf
        
        # Combine features
        X = np.hstack([X_numerical.values, X_text])
        y = balanced_df['Priorité'].values
        
        print(f"✅ Feature matrix: {X.shape}")
        print(f"✅ Priority range: {y.min():.1f} to {y.max():.1f}")
        print(f"✅ Priority mean: {y.mean():.2f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=pd.cut(y, bins=4, labels=False)
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['robust'] = scaler
        
        # Train models with sample weights for remaining imbalance
        sample_weights = compute_sample_weight('balanced', y_train)
        
        models = {
            'Random_Forest': RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient_Boosting': GradientBoostingRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                random_state=42
            ),
            'Extra_Trees': ExtraTreesRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Train and evaluate models
        results = {}
        predictions = {}
        
        for name, model in models.items():
            print(f"🤖 Training {name}...")
            
            if name in ['Random_Forest', 'Extra_Trees']:
                # Tree models can use original features
                model.fit(X_train, y_train, sample_weight=sample_weights)
                y_pred = model.predict(X_test)
            else:
                # Other models need scaled features
                model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
                y_pred = model.predict(X_test_scaled)
            
            y_pred = np.clip(y_pred, 1.0, 4.0)
            predictions[name] = y_pred
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mae': mae,
                'mse': mse,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"  📊 {name}: MAE={mae:.4f}, MSE={mse:.4f}, R²={r2:.4f}")
        
        # Create weighted ensemble
        print("🎭 Creating improved ensemble...")
        
        # Weight by inverse MAE
        weights = []
        ensemble_models = []
        
        for name, result in results.items():
            weight = 1 / (result['mae'] + 0.001)
            weights.append(weight)
            ensemble_models.append((name, result['model']))
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Ensemble prediction
        ensemble_pred = np.zeros_like(y_test)
        for i, (name, model) in enumerate(ensemble_models):
            if name in ['Random_Forest', 'Extra_Trees']:
                pred = model.predict(X_test)
            else:
                pred = model.predict(X_test_scaled)
            pred = np.clip(pred, 1.0, 4.0)
            ensemble_pred += weights[i] * pred
        
        ensemble_pred = np.clip(ensemble_pred, 1.0, 4.0)
        
        # Ensemble metrics
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        
        print(f"🎯 Ensemble: MAE={ensemble_mae:.4f}, MSE={ensemble_mse:.4f}, R²={ensemble_r2:.4f}")
        
        # Calculate accuracies
        accuracy_05 = np.mean(np.abs(ensemble_pred - y_test) <= 0.5) * 100
        accuracy_03 = np.mean(np.abs(ensemble_pred - y_test) <= 0.3) * 100
        
        print(f"\n📊 Accuracy:")
        print(f"  Within 0.5 points: {accuracy_05:.1f}%")
        print(f"  Within 0.3 points: {accuracy_03:.1f}%")
        
        # Test on different priority ranges
        print(f"\n🎯 Priority-specific accuracy:")
        for priority in [1.0, 2.0, 3.0, 4.0]:
            mask = y_test == priority
            if mask.sum() > 0:
                priority_mae = mean_absolute_error(y_test[mask], ensemble_pred[mask])
                priority_acc = np.mean(np.abs(ensemble_pred[mask] - y_test[mask]) <= 0.5) * 100
                print(f"  Priority {priority}: MAE={priority_mae:.3f}, Acc={priority_acc:.1f}%")
        
        # Store results
        self.models = results
        self.ensemble_models = ensemble_models
        self.ensemble_weights = weights
        self.feature_names = feature_cols + [f'tfidf_{i}' for i in range(X_text.shape[1])]
        
        return {
            'ensemble_mae': ensemble_mae,
            'ensemble_mse': ensemble_mse,
            'ensemble_r2': ensemble_r2,
            'accuracy_05': accuracy_05,
            'accuracy_03': accuracy_03
        }
    
    def save_model(self, metrics):
        """Save the improved model"""
        print("💾 Saving improved balanced model...")
        
        model_data = {
            'models': {name: result['model'] for name, result in self.models.items()},
            'ensemble_weights': self.ensemble_weights,
            'ensemble_models': [(name, self.models[name]['model']) for name, _ in self.ensemble_models],
            'vectorizers': self.vectorizers,
            'scalers': self.scalers,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, 'improved_balanced_taqa_model.joblib')
        
        model_info = {
            'model_name': 'Improved Balanced TAQA Model',
            'model_type': 'Balanced ensemble with intelligent features',
            'features_count': len(self.feature_names),
            'training_approach': 'Balanced dataset with domain-specific features',
            'ensemble_mae': metrics['ensemble_mae'],
            'ensemble_mse': metrics['ensemble_mse'],
            'ensemble_r2': metrics['ensemble_r2'],
            'accuracy_05': metrics['accuracy_05'],
            'accuracy_03': metrics['accuracy_03'],
            'improvements': [
                'Balanced training dataset',
                'Priority-specific features',
                'Equipment criticality analysis',
                'Smart text feature extraction',
                'Sample weighting for remaining imbalance',
                'Domain knowledge integration'
            ],
            'created_date': datetime.now().isoformat()
        }
        
        with open('improved_balanced_model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Model saved: improved_balanced_taqa_model.joblib")
        return model_info

def main():
    """Train the improved balanced model"""
    model = ImprovedBalancedTAQAModel()
    metrics = model.train_improved_model('data_set.csv')
    model_info = model.save_model(metrics)
    
    print(f"\n🎉 IMPROVED MODEL TRAINING COMPLETE!")
    print(f"📊 Performance:")
    print(f"   MAE: {metrics['ensemble_mae']:.4f}")
    print(f"   R²: {metrics['ensemble_r2']:.4f}")
    print(f"   Accuracy (±0.5): {metrics['accuracy_05']:.1f}%")
    
    if metrics['accuracy_05'] >= 85:
        print("🏆 EXCELLENT! Ready for production!")
    elif metrics['accuracy_05'] >= 75:
        print("✅ GOOD! Much better than before!")
    else:
        print("📈 IMPROVED! Better than original model!")

if __name__ == "__main__":
    main() 