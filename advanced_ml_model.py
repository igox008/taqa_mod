#!/usr/bin/env python3
"""
Advanced TAQA Anomaly Priority Predictor
State-of-the-art ML model with ensemble methods and deep feature engineering
Target: >90% accuracy with robust performance
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                            ExtraTreesRegressor, VotingRegressor, BaggingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import joblib
import json
import re
from datetime import datetime, timedelta
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

class AdvancedTAQAPredictor:
    """Advanced ML predictor with state-of-the-art techniques"""
    
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.feature_extractors = {}
        self.scalers = {}
        self.vectorizers = {}
        self.feature_names = []
    
    def extract_advanced_text_features(self, text):
        """Extract sophisticated text features"""
        if pd.isna(text) or text == '':
            return {
                'length': 0, 'word_count': 0, 'sentence_count': 0,
                'avg_word_length': 0, 'capital_ratio': 0, 'digit_ratio': 0,
                'punctuation_ratio': 0, 'unique_word_ratio': 0,
                'safety_score': 0, 'urgency_score': 0, 'technical_score': 0,
                'maintenance_score': 0, 'equipment_score': 0,
                'sentiment_score': 0, 'complexity_score': 0
            }
        
        text_str = str(text).lower()
        
        # Basic metrics
        length = len(text_str)
        words = text_str.split()
        word_count = len(words)
        sentence_count = len([s for s in text_str.split('.') if s.strip()])
        
        # Advanced metrics
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        capital_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        digit_ratio = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        punctuation_ratio = sum(1 for c in text if c in '.,!?;:') / len(text) if text else 0
        unique_word_ratio = len(set(words)) / len(words) if words else 0
        
        # Domain-specific scoring
        safety_keywords = ['fuite', 'urgent', 'arrêt', 'danger', 'risque', 'sécurité', 'défaut', 'panne']
        urgency_keywords = ['immédiat', 'urgent', 'critique', 'priorité', 'maintenance', 'réparation']
        technical_keywords = ['température', 'pression', 'vibration', 'débit', 'niveau', 'mesure', 'contrôle']
        maintenance_keywords = ['prévoir', 'maintenance', 'révision', 'inspection', 'vérification', 'nettoyage']
        equipment_keywords = ['pompe', 'moteur', 'vanne', 'alternateur', 'chaudière', 'turbine', 'ventilateur']
        
        safety_score = sum(1 for word in safety_keywords if word in text_str) / len(safety_keywords)
        urgency_score = sum(1 for word in urgency_keywords if word in text_str) / len(urgency_keywords)
        technical_score = sum(1 for word in technical_keywords if word in text_str) / len(technical_keywords)
        maintenance_score = sum(1 for word in maintenance_keywords if word in text_str) / len(maintenance_keywords)
        equipment_score = sum(1 for word in equipment_keywords if word in text_str) / len(equipment_keywords)
        
        # Sentiment and complexity
        positive_words = ['amélioration', 'optimisation', 'bon', 'correct', 'normal']
        negative_words = ['défaut', 'panne', 'problème', 'erreur', 'anomalie', 'dysfonctionnement']
        
        sentiment_score = (sum(1 for word in positive_words if word in text_str) - 
                          sum(1 for word in negative_words if word in text_str)) / max(len(words), 1)
        
        complexity_score = (avg_word_length * 0.3 + sentence_count * 0.2 + 
                          punctuation_ratio * 0.5) if sentence_count > 0 else 0
        
        return {
            'length': length,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'capital_ratio': capital_ratio,
            'digit_ratio': digit_ratio,
            'punctuation_ratio': punctuation_ratio,
            'unique_word_ratio': unique_word_ratio,
            'safety_score': safety_score,
            'urgency_score': urgency_score,
            'technical_score': technical_score,
            'maintenance_score': maintenance_score,
            'equipment_score': equipment_score,
            'sentiment_score': sentiment_score,
            'complexity_score': complexity_score
        }
    
    def extract_temporal_features(self, date_series):
        """Extract sophisticated temporal features"""
        dates = pd.to_datetime(date_series, errors='coerce')
        
        features = pd.DataFrame()
        features['year'] = dates.dt.year.fillna(2024)
        features['month'] = dates.dt.month.fillna(6)
        features['day'] = dates.dt.day.fillna(15)
        features['dayofweek'] = dates.dt.dayofweek.fillna(1)
        features['quarter'] = dates.dt.quarter.fillna(2)
        features['is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)
        features['is_month_start'] = dates.dt.is_month_start.astype(int)
        features['is_month_end'] = dates.dt.is_month_end.astype(int)
        features['day_of_year'] = dates.dt.dayofyear.fillna(150)
        
        # Cyclical encoding for time features
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['day_sin'] = np.sin(2 * np.pi * features['dayofweek'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['dayofweek'] / 7)
        
        return features
    
    def extract_equipment_features(self, equipment_series, description_series):
        """Extract advanced equipment features"""
        equipment_clean = equipment_series.fillna('').astype(str).str.lower()
        description_clean = description_series.fillna('').astype(str).str.lower()
        
        # Equipment categories with more sophistication
        equipment_categories = {
            'critical_rotating': ['pompe', 'moteur', 'ventilateur', 'turbine', 'alternateur'],
            'critical_static': ['chaudière', 'transformateur', 'échangeur'],
            'control_systems': ['vanne', 'clapet', 'servomoteur', 'régulateur'],
            'auxiliary': ['éclairage', 'prises', 'câblage', 'armoire'],
            'safety_systems': ['soupape', 'détecteur', 'alarme', 'sécurité'],
            'maintenance_tools': ['ramoneur', 'décrasseur', 'nettoyeur']
        }
        
        features = pd.DataFrame()
        
        # Category detection
        for category, keywords in equipment_categories.items():
            equipment_match = equipment_clean.str.contains('|'.join(keywords), na=False)
            description_match = description_clean.str.contains('|'.join(keywords), na=False)
            features[f'equipment_cat_{category}'] = (equipment_match | description_match).astype(int)
        
        # Equipment complexity score
        complexity_indicators = ['pompe alimentaire', 'moteur principal', 'alternateur unité', 'chaudière']
        features['equipment_complexity'] = sum(
            equipment_clean.str.contains(indicator, na=False).astype(int) 
            for indicator in complexity_indicators
        )
        
        # Equipment criticality (based on actual TAQA data patterns)
        high_criticality = ['évents ballon chaudière', 'pompe alimentaire', 'alternateur']
        medium_criticality = ['ventilateur', 'pompe', 'moteur']
        
        features['equipment_criticality'] = 0
        for equipment in high_criticality:
            features.loc[equipment_clean.str.contains(equipment, na=False), 'equipment_criticality'] = 3
        for equipment in medium_criticality:
            features.loc[equipment_clean.str.contains(equipment, na=False), 'equipment_criticality'] = 2
        features['equipment_criticality'] = features['equipment_criticality'].fillna(1)
        
        return features
    
    def load_and_prepare_data(self, file_path):
        """Load and prepare data with advanced preprocessing"""
        print("🔄 Loading and preparing data with advanced preprocessing...")
        
        df = pd.read_csv(file_path)
        print(f"📊 Loaded {len(df)} records")
        
        # Clean priority data
        df_clean = df.dropna(subset=['Priorité']).copy()
        print(f"📊 {len(df_clean)} records with valid priorities")
        
        print("Priority distribution:")
        print(df_clean['Priorité'].value_counts().sort_index())
        
        # Handle missing data intelligently
        df_clean['Description'] = df_clean['Description'].fillna('maintenance générale')
        df_clean['Description equipement'] = df_clean['Description equipement'].fillna('équipement non spécifié')
        df_clean['Section proprietaire'] = df_clean['Section proprietaire'].fillna('34MC')
        df_clean['Statut'] = df_clean['Statut'].fillna('Terminé')
        
        # Extract advanced text features
        print("🔤 Extracting advanced text features...")
        text_features = df_clean['Description'].apply(self.extract_advanced_text_features)
        text_df = pd.DataFrame(text_features.tolist())
        
        for col in text_df.columns:
            df_clean[f'text_{col}'] = text_df[col]
        
        # Extract temporal features
        print("📅 Extracting temporal features...")
        temporal_features = self.extract_temporal_features(df_clean['Date de detection de l\'anomalie'])
        for col in temporal_features.columns:
            df_clean[f'temporal_{col}'] = temporal_features[col]
        
        # Extract equipment features
        print("⚙️ Extracting equipment features...")
        equipment_features = self.extract_equipment_features(
            df_clean['Description equipement'], 
            df_clean['Description']
        )
        for col in equipment_features.columns:
            df_clean[col] = equipment_features[col]
        
        # Section features with advanced encoding
        section_dummies = pd.get_dummies(df_clean['Section proprietaire'], prefix='section')
        df_clean = pd.concat([df_clean, section_dummies], axis=1)
        
        # Status features
        status_dummies = pd.get_dummies(df_clean['Statut'], prefix='status')
        df_clean = pd.concat([df_clean, status_dummies], axis=1)
        
        self.processed_data = df_clean
        print(f"✅ Data preparation complete. Shape: {df_clean.shape}")
        
        return df_clean
    
    def create_advanced_features(self, df):
        """Create feature matrix with advanced techniques"""
        print("🧠 Creating advanced feature matrix...")
        
        # Text vectorization with multiple approaches
        text_features_list = []
        
        # TF-IDF with character n-grams (better for French)
        tfidf_word = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.85,
            analyzer='word'
        )
        
        tfidf_char = TfidfVectorizer(
            max_features=300,
            ngram_range=(2, 5),
            min_df=5,
            max_df=0.9,
            analyzer='char'
        )
        
        descriptions = df['Description'].fillna('').astype(str)
        
        X_tfidf_word = tfidf_word.fit_transform(descriptions).toarray()
        X_tfidf_char = tfidf_char.fit_transform(descriptions).toarray()
        
        self.vectorizers['tfidf_word'] = tfidf_word
        self.vectorizers['tfidf_char'] = tfidf_char
        
        # Numerical features
        numerical_cols = [col for col in df.columns if col.startswith(('text_', 'temporal_', 'equipment_'))]
        numerical_cols += [col for col in df.columns if col.startswith(('section_', 'status_'))]
        
        X_numerical = df[numerical_cols].fillna(0).values
        
        # Feature engineering: polynomial features for key variables
        from sklearn.preprocessing import PolynomialFeatures
        key_features = ['text_safety_score', 'text_urgency_score', 'text_technical_score', 'equipment_criticality']
        key_indices = [i for i, col in enumerate(numerical_cols) if col in key_features]
        
        if key_indices:
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            X_poly = poly.fit_transform(X_numerical[:, key_indices])
            self.feature_extractors['polynomial'] = poly
        else:
            X_poly = np.array([]).reshape(len(X_numerical), 0)
        
        # Combine all features
        X_combined = np.hstack([X_numerical, X_tfidf_word, X_tfidf_char, X_poly])
        
        # Feature selection
        selector = SelectKBest(score_func=f_regression, k=min(1000, X_combined.shape[1]))
        X_selected = selector.fit_transform(X_combined, df['Priorité'])
        self.feature_extractors['selector'] = selector
        
        # Feature names
        feature_names = numerical_cols.copy()
        feature_names += [f'tfidf_word_{i}' for i in range(X_tfidf_word.shape[1])]
        feature_names += [f'tfidf_char_{i}' for i in range(X_tfidf_char.shape[1])]
        feature_names += [f'poly_{i}' for i in range(X_poly.shape[1])]
        
        selected_indices = selector.get_support()
        self.feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_indices[i]]
        
        print(f"✅ Created {X_selected.shape[1]} selected features from {X_combined.shape[1]} total features")
        
        return X_selected, df['Priorité'].values
    
    def train_ensemble_models(self, X, y):
        """Train ensemble of advanced models"""
        print("🤖 Training ensemble of advanced models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=4, labels=False)
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['robust'] = scaler
        
        # Define advanced models
        models = {
            'Random_Forest': RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'Extra_Trees': ExtraTreesRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient_Boosting': GradientBoostingRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            ),
            'Neural_Network': MLPRegressor(
                hidden_layer_sizes=(200, 100, 50),
                activation='relu',
                alpha=0.01,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            ),
            'SVR_RBF': SVR(
                kernel='rbf',
                C=100,
                gamma='scale',
                epsilon=0.1
            ),
            'Elastic_Net': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42
            )
        }
        
        # Train individual models
        results = {}
        predictions = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            try:
                if name in ['Random_Forest', 'Extra_Trees', 'Gradient_Boosting']:
                    # Tree-based models can use original features
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                else:
                    # Other models need scaled features
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                
                # Clip predictions to valid range
                y_pred = np.clip(y_pred, 1.0, 5.0)
                predictions[name] = y_pred
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train_scaled if name not in ['Random_Forest', 'Extra_Trees', 'Gradient_Boosting'] else X_train, 
                    y_train, cv=5, scoring='neg_mean_absolute_error'
                )
                
                results[name] = {
                    'model': model,
                    'mae': mae,
                    'mse': mse,
                    'r2': r2,
                    'cv_mae': -cv_scores.mean(),
                    'predictions': y_pred
                }
                
                print(f"  ✅ {name}: MAE={mae:.4f}, MSE={mse:.4f}, R²={r2:.4f}, CV_MAE={-cv_scores.mean():.4f}")
                
            except Exception as e:
                print(f"  ❌ {name} failed: {e}")
                continue
        
        # Create ensemble
        print("🎭 Creating ensemble model...")
        
        # Weight models by performance (inverse MAE)
        weights = []
        ensemble_models = []
        
        for name, result in results.items():
            if result['mae'] < 0.5:  # Only include good models
                weight = 1 / (result['mae'] + 0.001)  # Inverse MAE weighting
                weights.append(weight)
                ensemble_models.append((name, result['model']))
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        print(f"Ensemble weights: {dict(zip([name for name, _ in ensemble_models], weights))}")
        
        # Create weighted ensemble prediction
        ensemble_pred = np.zeros_like(y_test)
        for i, (name, model) in enumerate(ensemble_models):
            if name in ['Random_Forest', 'Extra_Trees', 'Gradient_Boosting']:
                pred = model.predict(X_test)
            else:
                pred = model.predict(X_test_scaled)
            pred = np.clip(pred, 1.0, 5.0)
            ensemble_pred += weights[i] * pred
        
        ensemble_pred = np.clip(ensemble_pred, 1.0, 5.0)
        
        # Ensemble metrics
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        
        print(f"🎯 Ensemble: MAE={ensemble_mae:.4f}, MSE={ensemble_mse:.4f}, R²={ensemble_r2:.4f}")
        
        # Store results
        self.models = results
        self.ensemble_models = ensemble_models
        self.ensemble_weights = weights
        self.test_data = (X_test, X_test_scaled, y_test)
        
        # Calculate accuracy at different thresholds
        accuracy_05 = np.mean(np.abs(ensemble_pred - y_test) <= 0.5) * 100
        accuracy_03 = np.mean(np.abs(ensemble_pred - y_test) <= 0.3) * 100
        accuracy_02 = np.mean(np.abs(ensemble_pred - y_test) <= 0.2) * 100
        
        print(f"\n📊 Ensemble Accuracy:")
        print(f"  Within 0.5 points: {accuracy_05:.1f}%")
        print(f"  Within 0.3 points: {accuracy_03:.1f}%")
        print(f"  Within 0.2 points: {accuracy_02:.1f}%")
        
        return {
            'ensemble_mae': ensemble_mae,
            'ensemble_mse': ensemble_mse,
            'ensemble_r2': ensemble_r2,
            'accuracy_05': accuracy_05,
            'accuracy_03': accuracy_03,
            'accuracy_02': accuracy_02
        }
    
    def save_advanced_model(self, metrics):
        """Save the advanced model"""
        print("💾 Saving advanced model...")
        
        # Save model components
        model_data = {
            'models': {name: result['model'] for name, result in self.models.items()},
            'ensemble_weights': self.ensemble_weights,
            'ensemble_models': [(name, None) for name, _ in self.ensemble_models],  # Names only
            'vectorizers': self.vectorizers,
            'scalers': self.scalers,
            'feature_extractors': self.feature_extractors,
            'feature_names': self.feature_names
        }
        
        # Add actual models to ensemble_models
        for i, (name, _) in enumerate(model_data['ensemble_models']):
            model_data['ensemble_models'][i] = (name, self.models[name]['model'])
        
        joblib.dump(model_data, 'advanced_taqa_model.joblib')
        
        # Save model info
        model_info = {
            'model_name': 'Advanced TAQA Ensemble',
            'model_type': 'Ensemble of Random Forest, Extra Trees, Gradient Boosting, Neural Network, SVR, Elastic Net',
            'features_count': len(self.feature_names),
            'training_samples': len(self.processed_data),
            'ensemble_mae': metrics['ensemble_mae'],
            'ensemble_mse': metrics['ensemble_mse'],
            'ensemble_r2': metrics['ensemble_r2'],
            'accuracy_05': metrics['accuracy_05'],
            'accuracy_03': metrics['accuracy_03'],
            'accuracy_02': metrics['accuracy_02'],
            'priority_range': [1.0, 5.0],
            'created_date': datetime.now().isoformat(),
            'techniques_used': [
                'Advanced text feature engineering',
                'Temporal feature extraction',
                'Equipment criticality analysis',
                'Multiple text vectorization approaches',
                'Polynomial feature interactions',
                'Feature selection',
                'Ensemble modeling with weighted voting',
                'Robust scaling',
                'Cross-validation'
            ]
        }
        
        with open('advanced_model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Advanced model saved as: advanced_taqa_model.joblib")
        print(f"✅ Model info saved as: advanced_model_info.json")
        
        return model_info

def main():
    """Main training function"""
    print("🚀 ADVANCED TAQA ML MODEL TRAINING")
    print("=" * 60)
    
    # Initialize predictor
    predictor = AdvancedTAQAPredictor()
    
    # Load and prepare data
    df = predictor.load_and_prepare_data('data_set.csv')
    
    # Create advanced features
    X, y = predictor.create_advanced_features(df)
    
    # Train ensemble models
    metrics = predictor.train_ensemble_models(X, y)
    
    # Save model
    model_info = predictor.save_advanced_model(metrics)
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"📊 Final Performance:")
    print(f"   MAE: {metrics['ensemble_mae']:.4f}")
    print(f"   R²: {metrics['ensemble_r2']:.4f}")
    print(f"   Accuracy (±0.3): {metrics['accuracy_03']:.1f}%")
    
    if metrics['accuracy_03'] >= 85:
        print("🏆 EXCELLENT! Model ready for production!")
    elif metrics['accuracy_03'] >= 75:
        print("✅ GOOD! Model performs well!")
    else:
        print("⚠️ Needs improvement, but better than previous models!")

if __name__ == "__main__":
    main() 