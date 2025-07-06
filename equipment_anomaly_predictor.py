#!/usr/bin/env python3


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
import re
from datetime import datetime
import os
import openai
from typing import List, Dict, Tuple, Optional

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, using alternatives")

from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                           classification_report, confusion_matrix, accuracy_score,
                           precision_recall_fscore_support)

from scipy import stats
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    FRENCH_STOPWORDS = stopwords.words('french')
except:
    FRENCH_STOPWORDS = ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour']

class EquipmentAnomalyPredictor:
    def __init__(self):
        self.data = None
        self.features = None
        self.target = None
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.tfidf_vectorizers = {}
        self.feature_names = []
        self.evaluation_results = {}
        
    def load_and_explore_data(self, filepath):
        print("Loading and exploring dataset...")
        
        self.data = pd.read_csv(filepath)
        print(f"Dataset shape: {self.data.shape}")
        
        print("\nDataset Overview:")
        print(self.data.info())
        print(f"\nTarget variable (Criticité) distribution:")
        print(self.data['Criticité'].value_counts().sort_index())
        
        print(f"\nMissing values:")
        print(self.data.isnull().sum())
        
        self._create_exploratory_plots()
        
        return self.data
    
    def _create_exploratory_plots(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        self.data['Criticité'].hist(bins=20, ax=axes[0,0], alpha=0.7, color='skyblue')
        axes[0,0].set_title('Distribution of Criticality Scores')
        axes[0,0].set_xlabel('Criticité')
        axes[0,0].set_ylabel('Frequency')
        
        equipment_counts = self.data['Description de l\'équipement'].value_counts().head(10)
        equipment_counts.plot(kind='bar', ax=axes[0,1], color='lightgreen')
        axes[0,1].set_title('Top 10 Equipment Types')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        self.data['Section propriétaire'].value_counts().plot(kind='bar', ax=axes[0,2], color='coral')
        axes[0,2].set_title('Equipment by Section')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        section_criticality = self.data.groupby('Section propriétaire')['Criticité'].mean().sort_values(ascending=False)
        section_criticality.plot(kind='bar', ax=axes[1,0], color='gold')
        axes[1,0].set_title('Average Criticality by Section')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        score_cols = ['Fiabilité Intégrité', 'Disponibilté', 'Process Safety', 'Criticité']
        corr_matrix = self.data[score_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
        axes[1,1].set_title('Correlation Matrix - Scoring System')
        
        self.data['Date de détéction de l\'anomalie'] = pd.to_datetime(self.data['Date de détéction de l\'anomalie'])
        monthly_counts = self.data.set_index('Date de détéction de l\'anomalie').resample('M').size()
        monthly_counts.plot(ax=axes[1,2], color='purple')
        axes[1,2].set_title('Anomalies Over Time')
        axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('equipment_anomaly_eda.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def extract_text_features(self, description_col):
        print("Extracting text features from anomaly descriptions...")
        
        critical_keywords = {
            'leak_keywords': ['fuite', 'fuite importante', 'fuite d\'eau', 'fuite d\'huile', 'fuite de vapeur'],
            'vibration_keywords': ['vibration', 'bruit anormal', 'bruit', 'vibrant'],
            'seal_keywords': ['non étanche', 'non-étanche', 'étanche', 'étanchéité'],
            'damage_keywords': ['percement', 'rupture', 'cassé', 'défaillance', 'panne'],
            'temperature_keywords': ['surchauffe', 'température', 'chauffage', 'refroidissement'],
            'pressure_keywords': ['pression', 'surpression', 'dépression'],
            'wear_keywords': ['usure', 'encrassement', 'corrosion', 'érosion'],
            'mechanical_keywords': ['coincement', 'blocage', 'grippé', 'difficile']
        }
        
        text_features = pd.DataFrame()
        
        for category, keywords in critical_keywords.items():
            for keyword in keywords:
                col_name = f'has_{keyword.replace(" ", "_").replace("\'", "")}'
                text_features[col_name] = description_col.str.lower().str.contains(keyword, na=False).astype(int)
        
        severity_patterns = {
            'severity_high': ['importante?', 'grave', 'urgent', 'critique'],
            'severity_medium': ['prévoir', 'contrôle', 'vérification'],
            'severity_maintenance': ['nettoyage', 'entretien', 'graissage', 'remplacement']
        }
        
        for severity, patterns in severity_patterns.items():
            pattern = '|'.join(patterns)
            text_features[severity] = description_col.str.lower().str.contains(pattern, na=False).astype(int)
        
        text_features['description_length'] = description_col.str.len()
        text_features['word_count'] = description_col.str.split().str.len()
        text_features['capital_ratio'] = description_col.str.count(r'[A-Z]') / text_features['description_length']
        
        cleaned_descriptions = description_col.str.lower()
        cleaned_descriptions = cleaned_descriptions.str.replace(r'[^\w\s]', ' ', regex=True)
        cleaned_descriptions = cleaned_descriptions.str.replace(r'\s+', ' ', regex=True)
        
        if len(description_col) == 1:
            if 'description' not in self.tfidf_vectorizers:
                self.tfidf_vectorizers['description'] = TfidfVectorizer(
                    max_features=50,
                    stop_words=FRENCH_STOPWORDS,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=1.0
                )
                tfidf_features = self.tfidf_vectorizers['description'].fit_transform(cleaned_descriptions.fillna(''))
            else:
                tfidf_features = self.tfidf_vectorizers['description'].transform(cleaned_descriptions.fillna(''))
        else:
        self.tfidf_vectorizers['description'] = TfidfVectorizer(
            max_features=50,
            stop_words=FRENCH_STOPWORDS,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        tfidf_features = self.tfidf_vectorizers['description'].fit_transform(cleaned_descriptions.fillna(''))
        
        feature_names = [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=feature_names
        )
        
        if len(description_col) == 1 and 'description' in self.tfidf_vectorizers:
            expected_features = [f'tfidf_{i}' for i in range(50)]
            for feature in expected_features:
                if feature not in tfidf_df.columns:
                    tfidf_df[feature] = 0
            
            # Ensure correct column order
            tfidf_df = tfidf_df[expected_features]
        
        return pd.concat([text_features, tfidf_df], axis=1)
    
    def extract_temporal_features(self, date_col):
        """Extract temporal features from anomaly detection dates"""
        print("Extracting temporal features...")
        
        dates = pd.to_datetime(date_col)
        temporal_features = pd.DataFrame()
        
        # Basic temporal features
        temporal_features['year'] = dates.dt.year
        temporal_features['month'] = dates.dt.month
        temporal_features['day_of_week'] = dates.dt.dayofweek
        temporal_features['day_of_month'] = dates.dt.day
        temporal_features['hour'] = dates.dt.hour
        temporal_features['quarter'] = dates.dt.quarter
        
        # Cyclical encoding
        temporal_features['month_sin'] = np.sin(2 * np.pi * temporal_features['month'] / 12)
        temporal_features['month_cos'] = np.cos(2 * np.pi * temporal_features['month'] / 12)
        temporal_features['dow_sin'] = np.sin(2 * np.pi * temporal_features['day_of_week'] / 7)
        temporal_features['dow_cos'] = np.cos(2 * np.pi * temporal_features['day_of_week'] / 7)
        temporal_features['hour_sin'] = np.sin(2 * np.pi * temporal_features['hour'] / 24)
        temporal_features['hour_cos'] = np.cos(2 * np.pi * temporal_features['hour'] / 24)
        
        # Operational patterns (assuming 3 shifts: 6-14, 14-22, 22-6)
        shift_mapping = []
        for hour in temporal_features['hour']:
            if 6 <= hour < 14:
                shift_mapping.append(0)  # morning
            elif 14 <= hour < 22:
                shift_mapping.append(1)  # afternoon
            else:
                shift_mapping.append(2)  # night
        temporal_features['shift'] = shift_mapping
        
        # Weekend indicator
        temporal_features['is_weekend'] = (temporal_features['day_of_week'] >= 5).astype(int)
        
        return temporal_features
    
    def extract_equipment_features(self):
        """Extract and engineer equipment-based features"""
        print("Extracting equipment and system features...")
        
        equipment_features = pd.DataFrame()
        
        # Encode equipment types
        self.encoders['equipment'] = LabelEncoder()
        equipment_features['equipment_type_encoded'] = self.encoders['equipment'].fit_transform(
            self.data['Description de l\'équipement'].fillna('Unknown')
        )
        
        # Encode sections
        self.encoders['section'] = LabelEncoder()
        equipment_features['section_encoded'] = self.encoders['section'].fit_transform(
            self.data['Section propriétaire'].fillna('Unknown')
        )
        
        # Equipment frequency (how often this equipment type has anomalies)
        equipment_counts = self.data['Description de l\'équipement'].value_counts()
        equipment_features['equipment_frequency'] = self.data['Description de l\'équipement'].map(equipment_counts)
        
        # System frequency
        system_counts = self.data['Systeme'].value_counts()
        equipment_features['system_frequency'] = self.data['Systeme'].map(system_counts)
        
        # Historical criticality for equipment type
        equipment_criticality = self.data.groupby('Description de l\'équipement')['Criticité'].agg(['mean', 'std', 'count'])
        equipment_features['equipment_avg_criticality'] = self.data['Description de l\'équipement'].map(equipment_criticality['mean'])
        equipment_features['equipment_std_criticality'] = self.data['Description de l\'équipement'].map(equipment_criticality['std']).fillna(0)
        
        # Section criticality patterns
        section_criticality = self.data.groupby('Section propriétaire')['Criticité'].agg(['mean', 'std'])
        equipment_features['section_avg_criticality'] = self.data['Section propriétaire'].map(section_criticality['mean'])
        equipment_features['section_std_criticality'] = self.data['Section propriétaire'].map(section_criticality['std']).fillna(0)
        
        return equipment_features
    
    def prepare_features(self):
        """Comprehensive feature engineering pipeline"""
        print("Preparing comprehensive feature set...")
        
        # Extract all feature types
        text_features = self.extract_text_features(self.data['Description'])
        temporal_features = self.extract_temporal_features(self.data['Date de détéction de l\'anomalie'])
        equipment_features = self.extract_equipment_features()
        
        # Numerical features from existing scoring system
        numerical_features = self.data[['Fiabilité Intégrité', 'Disponibilté', 'Process Safety']].copy()
        
        # Combine all features
        self.features = pd.concat([
            numerical_features,
            text_features,
            temporal_features,
            equipment_features
        ], axis=1)
        
        # Handle categorical variables in temporal features
        categorical_cols = self.features.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col in self.features.columns:
                self.encoders[col] = LabelEncoder()
                self.features[col] = self.encoders[col].fit_transform(self.features[col].fillna('Unknown'))
        
        # Fill any remaining NaN values
        self.features = self.features.fillna(0)
        
        # Target variable
        self.target = self.data['Criticité'].copy()
        
        # Store feature names
        self.feature_names = list(self.features.columns)
        
        print(f"Feature engineering complete. Shape: {self.features.shape}")
        print(f"Features created: {len(self.feature_names)}")
        
        return self.features, self.target
    
    def create_priority_tiers(self):
        """Create priority tier classification as suggested"""
        # Create priority tiers: 1-4: Low, 5-7: Medium, 8-12: High
        priority_tiers = pd.cut(
            self.target,
            bins=[0, 4, 7, 12],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        return priority_tiers
    
    def train_models(self, test_size=0.15, validation_size=0.15, random_state=42):
        """Train Gradient Boosting Classifier for 15-class prediction"""
        print("Training Gradient Boosting Classifier...")
        
        # Prepare data splits
        X, y = self.features, self.target
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=pd.cut(y, bins=3)
        )
        
        # Second split: train and validation
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state,
            stratify=pd.cut(y_temp, bins=3)
        )
        
        print(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")
        
        # Convert target variables to integer class labels
        y_train_class = y_train.astype(int)
        y_val_class = y_val.astype(int)
        y_test_class = y_test.astype(int)
        
        # Initialize Gradient Boosting Classifier with optimized parameters
        gb_classifier = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state
        )
        
        # Train the model
        gb_classifier.fit(X_train, y_train_class)
        
        # Make predictions
        val_pred = gb_classifier.predict(X_val)
        test_pred = gb_classifier.predict(X_test)
        
        # Calculate metrics
            val_acc = accuracy_score(y_val_class, val_pred)
            test_acc = accuracy_score(y_test_class, test_pred)
            
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(y_val_class, val_pred, average='weighted')
            test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test_class, test_pred, average='weighted')
            
        # Store model and results
        self.models['Gradient Boosting Classifier'] = gb_classifier
        self.evaluation_results['Gradient Boosting Classifier'] = {
            'val_accuracy': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'val_predictions': val_pred,
            'test_predictions': test_pred
        }
        
        print(f"\nModel Performance:")
        print(f"Validation - Accuracy: {val_acc:.3f}, F1: {val_f1:.3f}")
        print(f"Test - Accuracy: {test_acc:.3f}, F1: {test_f1:.3f}")
        
        # Store data splits for later use
        self.data_splits = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'y_train_class': y_train_class, 'y_val_class': y_val_class, 'y_test_class': y_test_class
        }
        
        return self.models, self.evaluation_results
    
    def analyze_feature_importance(self):
        """Analyze feature importance across models"""
        print("Analyzing feature importance...")
        
        # Get feature importance from tree-based models
        importance_data = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_data[model_name] = model.feature_importances_
        
        if importance_data:
            importance_df = pd.DataFrame(importance_data, index=self.feature_names)
            
            # Plot top features
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            
            for i, (model_name, importances) in enumerate(list(importance_data.items())[:4]):
                ax = axes[i//2, i%2]
                top_features = importance_df[model_name].nlargest(20)
                top_features.plot(kind='barh', ax=ax)
                ax.set_title(f'Top 20 Features - {model_name}')
                ax.set_xlabel('Feature Importance')
            
            plt.tight_layout()
            plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return importance_df
        
        return None
    
    def create_ensemble_prediction(self):
        """Create ensemble predictions combining multiple models"""
        print("Creating ensemble predictions...")
        
        # Get predictions from best performing models
        regression_preds = []
        classification_preds = []
        
        # Select top 3 regression models by test R²
        reg_scores = {name: results['test_r2'] for name, results in self.evaluation_results.items() 
                     if 'regression' in name}
        top_reg_models = sorted(reg_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for model_name, _ in top_reg_models:
            pred = self.evaluation_results[model_name]['test_predictions']
            regression_preds.append(pred)
        
        # Select top 3 classification models by test F1
        class_scores = {name: results['test_f1'] for name, results in self.evaluation_results.items() 
                       if 'classification' in name}
        top_class_models = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for model_name, _ in top_class_models:
            pred = self.evaluation_results[model_name]['test_predictions']
            classification_preds.append(pred)
        
        # Ensemble predictions
        ensemble_reg_pred = np.mean(regression_preds, axis=0)
        ensemble_class_pred = stats.mode(classification_preds, axis=0)[0].flatten()
        
        # Evaluate ensemble
        y_test = self.data_splits['y_test']
        y_test_class = self.data_splits['y_test_class']
        
        ensemble_mae = mean_absolute_error(y_test, ensemble_reg_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_reg_pred))
        ensemble_r2 = r2_score(y_test, ensemble_reg_pred)
        
        ensemble_acc = accuracy_score(y_test_class, ensemble_class_pred)
        ensemble_f1 = precision_recall_fscore_support(y_test_class, ensemble_class_pred, average='weighted')[2]
        
        print(f"\nEnsemble Results:")
        print(f"Regression - MAE: {ensemble_mae:.3f}, RMSE: {ensemble_rmse:.3f}, R²: {ensemble_r2:.3f}")
        print(f"Classification - Accuracy: {ensemble_acc:.3f}, F1: {ensemble_f1:.3f}")
        
        return {
            'ensemble_regression': ensemble_reg_pred,
            'ensemble_classification': ensemble_class_pred,
            'metrics': {
                'mae': ensemble_mae, 'rmse': ensemble_rmse, 'r2': ensemble_r2,
                'accuracy': ensemble_acc, 'f1': ensemble_f1
            }
        }
    
    def generate_comprehensive_report(self):
        """Generate comprehensive model evaluation report"""
        print("Generating comprehensive evaluation report...")
        
        # Create results summary
        results_summary = []
        
        for model_name, results in self.evaluation_results.items():
            if 'regression' in model_name:
                results_summary.append({
                    'Model': model_name,
                    'Type': 'Regression',
                    'Test_MAE': results['test_mae'],
                    'Test_RMSE': results['test_rmse'],
                    'Test_R2': results['test_r2']
                })
            else:
                results_summary.append({
                    'Model': model_name,
                    'Type': 'Classification',
                    'Test_Accuracy': results['test_accuracy'],
                    'Test_F1': results['test_f1'],
                    'Test_Precision': results['test_precision']
                })
        
        results_df = pd.DataFrame(results_summary)
        
        # Save results
        results_df.to_csv('model_evaluation_results.csv', index=False)
        
        # Create visualizations
        self._create_evaluation_plots()
        
        print("Comprehensive report generated!")
        print("\nTop Performing Models:")
        print("Regression (by R²):")
        reg_results = results_df[results_df['Type'] == 'Regression'].sort_values('Test_R2', ascending=False)
        print(reg_results.head(3)[['Model', 'Test_MAE', 'Test_R2']])
        
        print("\nClassification (by F1):")
        class_results = results_df[results_df['Type'] == 'Classification'].sort_values('Test_F1', ascending=False)
        print(class_results.head(3)[['Model', 'Test_Accuracy', 'Test_F1']])
        
        return results_df
    
    def _create_evaluation_plots(self):
        """Create comprehensive evaluation plots"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Model performance comparison - Regression
        reg_models = [name for name in self.evaluation_results.keys() if 'regression' in name]
        reg_mae = [self.evaluation_results[name]['test_mae'] for name in reg_models]
        reg_r2 = [self.evaluation_results[name]['test_r2'] for name in reg_models]
        
        x_pos = np.arange(len(reg_models))
        axes[0,0].bar(x_pos, reg_mae, color='skyblue', alpha=0.7)
        axes[0,0].set_title('Test MAE by Model (Regression)')
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels([name.replace('_regression', '') for name in reg_models], rotation=45)
        
        # 2. R² scores
        axes[0,1].bar(x_pos, reg_r2, color='lightgreen', alpha=0.7)
        axes[0,1].set_title('Test R² by Model (Regression)')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels([name.replace('_regression', '') for name in reg_models], rotation=45)
        
        # 3. Classification performance
        class_models = [name for name in self.evaluation_results.keys() if 'classification' in name]
        class_acc = [self.evaluation_results[name]['test_accuracy'] for name in class_models]
        class_f1 = [self.evaluation_results[name]['test_f1'] for name in class_models]
        
        x_pos_class = np.arange(len(class_models))
        axes[0,2].bar(x_pos_class, class_acc, color='coral', alpha=0.7)
        axes[0,2].set_title('Test Accuracy by Model (Classification)')
        axes[0,2].set_xticks(x_pos_class)
        axes[0,2].set_xticklabels([name.replace('_classification', '') for name in class_models], rotation=45)
        
        # 4. F1 scores
        axes[1,0].bar(x_pos_class, class_f1, color='gold', alpha=0.7)
        axes[1,0].set_title('Test F1 Score by Model (Classification)')
        axes[1,0].set_xticks(x_pos_class)
        axes[1,0].set_xticklabels([name.replace('_classification', '') for name in class_models], rotation=45)
        
        # 5. Prediction vs Actual (best regression model)
        best_reg_model = max(reg_models, key=lambda x: self.evaluation_results[x]['test_r2'])
        test_pred = self.evaluation_results[best_reg_model]['test_predictions']
        y_test = self.data_splits['y_test']
        
        axes[1,1].scatter(y_test, test_pred, alpha=0.6, color='purple')
        axes[1,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1,1].set_xlabel('Actual Criticality')
        axes[1,1].set_ylabel('Predicted Criticality')
        axes[1,1].set_title(f'Predicted vs Actual - {best_reg_model.replace("_regression", "")}')
        
        # 6. Confusion matrix (best classification model)
        best_class_model = max(class_models, key=lambda x: self.evaluation_results[x]['test_f1'])
        test_pred_class = self.evaluation_results[best_class_model]['test_predictions']
        y_test_class = self.data_splits['y_test_class']
        
        cm = confusion_matrix(y_test_class, test_pred_class)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,2])
        axes[1,2].set_title(f'Confusion Matrix - {best_class_model.replace("_classification", "")}')
        axes[1,2].set_xlabel('Predicted')
        axes[1,2].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('model_evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_new_anomaly(self, description, equipment_type, section, 
                           fiabilite=1, disponibilite=1, process_safety=1, 
                           detection_date=None):
        """Predict criticality for a new anomaly"""
        if detection_date is None:
            detection_date = datetime.now()
        
        # Create a single-row dataframe with the new data
        new_data = pd.DataFrame({
            'Description': [description],
            'Description de l\'équipement': [equipment_type],
            'Section propriétaire': [section],
            'Fiabilité Intégrité': [fiabilite],
            'Disponibilté': [disponibilite],
            'Process Safety': [process_safety],
            'Date de détéction de l\'anomalie': [detection_date]
        })
        
        # Extract features using same pipeline
        text_features = self.extract_text_features(new_data['Description'])
        temporal_features = self.extract_temporal_features(new_data['Date de détéction de l\'anomalie'])
        
        # Equipment features (handle unknown equipment/section)
        equipment_features = pd.DataFrame()
        
        # Handle equipment encoding
        if equipment_type in self.encoders['equipment'].classes_:
            equipment_features['equipment_type_encoded'] = [self.encoders['equipment'].transform([equipment_type])[0]]
        else:
            equipment_features['equipment_type_encoded'] = [0]  # Unknown
        
        # Handle section encoding
        if section in self.encoders['section'].classes_:
            equipment_features['section_encoded'] = [self.encoders['section'].transform([section])[0]]
        else:
            equipment_features['section_encoded'] = [0]  # Unknown
        
        # Set default values for frequency and historical features
        equipment_features['equipment_frequency'] = [1]
        equipment_features['system_frequency'] = [1]
        equipment_features['equipment_avg_criticality'] = [6]  # Average criticality
        equipment_features['equipment_std_criticality'] = [2]
        equipment_features['section_avg_criticality'] = [6]
        equipment_features['section_std_criticality'] = [2]
        
        # Combine features
        new_features = pd.concat([
            new_data[['Fiabilité Intégrité', 'Disponibilté', 'Process Safety']],
            text_features,
            temporal_features,
            equipment_features
        ], axis=1)
        
        # Ensure all features are present and in correct order
        for col in self.feature_names:
            if col not in new_features.columns:
                new_features[col] = 0
        
        new_features = new_features[self.feature_names]
        
        # Get predictions from Gradient Boosting Classifier
        model = self.models['Gradient Boosting Classifier']
        criticality_pred = model.predict(new_features)[0]
        
        # Calculate confidence score
        proba = model.predict_proba(new_features)[0]
        max_proba = np.max(proba)  # Get the highest probability
        
        # Calculate weighted confidence based on neighboring classes
        pred_idx = int(criticality_pred) - min(model.classes_)  # Adjust for class indexing
        confidence = proba[pred_idx]
        
        # Add partial confidence from neighboring classes
        if pred_idx > 0:  # Add confidence from lower class
            confidence += 0.3 * proba[pred_idx - 1]
        if pred_idx < len(proba) - 1:  # Add confidence from higher class
            confidence += 0.3 * proba[pred_idx + 1]
        
        # Normalize confidence to be between 0 and 1
        confidence = min(confidence, 1.0)
        
        return {
            'predicted_criticality': int(criticality_pred),
            'confidence_score': float(confidence),
            'max_probability': float(max_proba),
            'probability_distribution': {i: float(p) for i, p in enumerate(proba, start=min(model.classes_))},
            'recommendations': self._generate_recommendations(description, criticality_pred)
        }
    
    def _generate_recommendations(self, description, criticality):
        """Generate maintenance recommendations based on prediction"""
        recommendations = []
        
        desc_lower = description.lower()
        
        # Detailed recommendations based on exact criticality score
        if criticality >= 12:
            recommendations.append("CRITICAL EMERGENCY: Immediate shutdown and intervention required")
            recommendations.append("Alert senior maintenance team and management")
        elif criticality >= 9:
            recommendations.append("URGENT: Immediate intervention required")
            recommendations.append("Schedule emergency maintenance within 24h")
        elif criticality >= 7:
            recommendations.append("HIGH PRIORITY: Schedule maintenance within 1 week")
            recommendations.append("Prepare maintenance team and spare parts")
        elif criticality >= 5:
            recommendations.append("MEDIUM PRIORITY: Schedule maintenance within 1 month")
            recommendations.append("Monitor condition closely")
        else:
            recommendations.append("LOW PRIORITY: Include in next planned maintenance")
            recommendations.append("Continue routine monitoring")
        
        # Specific recommendations based on description
        if 'fuite' in desc_lower:
            recommendations.append("Check seals and gaskets")
            recommendations.append("Verify tightening torque of connections")
        
        if 'vibration' in desc_lower or 'bruit' in desc_lower:
            recommendations.append("Check alignment and balancing")
            recommendations.append("Inspect bearings and mountings")
        
        if 'température' in desc_lower or 'surchauffe' in desc_lower:
            recommendations.append("Verify cooling system operation")
            recommendations.append("Check for obstructions in ventilation")
        
        return recommendations


class SmartParameterPredictor:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """Initialize the SmartParameterPredictor with OpenAI integration"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either directly or via OPENAI_API_KEY environment variable")
        
        openai.api_key = self.api_key
        self.model = model
        self.df = pd.read_csv("data_set.csv")
        
        # Initialize TF-IDF vectorizer for similarity matching
        self.vectorizer = TfidfVectorizer(
            stop_words=FRENCH_STOPWORDS,
            ngram_range=(1, 2),
            max_features=100
        )
        
    def _find_similar_incidents(self, new_description: str, equipment_data: pd.DataFrame, top_k: int = 3) -> List[Dict]:
        """Find most similar past incidents using TF-IDF and cosine similarity"""
        if len(equipment_data) == 0:
            return []
            
        # Combine all descriptions
        all_descriptions = list(equipment_data['Description']) + [new_description]
        
        # Fit and transform descriptions to TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(all_descriptions)
        
        # Calculate cosine similarity between new description and all past incidents
        new_vector = tfidf_matrix[-1]
        past_vectors = tfidf_matrix[:-1]
        similarities = past_vectors.dot(new_vector.T).toarray().flatten()
        
        # Get indices of top-k similar incidents
        similar_indices = similarities.argsort()[-top_k:][::-1]
        
        # Format similar incidents
        similar_incidents = []
        for idx in similar_indices:
            if similarities[idx] > 0.1:  # Only include if similarity is above threshold
                incident = equipment_data.iloc[idx]
                similar_incidents.append({
                    'date': pd.to_datetime(incident['Date de détéction de l\'anomalie']).strftime('%Y-%m-%d'),
                    'description': incident['Description'],
                    'ratings': {
                        'Fiabilité': incident['Fiabilité Intégrité'],
                        'Disponibilité': incident['Disponibilté'],
                        'Process Safety': incident['Process Safety'],
                        'Criticité': incident['Criticité']
                    },
                    'similarity': similarities[idx]
                })
        
        return similar_incidents

    def _get_equipment_history(self, equipment_type: str, max_examples: int = 20) -> Tuple[List[Dict], Dict[str, float]]:
        """Get historical examples for specific equipment type"""
        # Filter for the specific equipment
        equipment_data = self.df[self.df['Description de l\'équipement'] == equipment_type].copy()
        
        # Sort by date (newest first) and take last max_examples
        equipment_data['Date'] = pd.to_datetime(equipment_data['Date de détéction de l\'anomalie'])
        equipment_data = equipment_data.sort_values('Date', ascending=False).head(max_examples)
        
        # Calculate average ratings for this equipment
        avg_ratings = {
            'Fiabilité': equipment_data['Fiabilité Intégrité'].mean(),
            'Disponibilité': equipment_data['Disponibilté'].mean(),
            'Process Safety': equipment_data['Process Safety'].mean(),
            'Criticité': equipment_data['Criticité'].mean()
        }
        
        # Format examples
        examples = []
        for _, row in equipment_data.iterrows():
            examples.append({
                'date': row['Date'].strftime('%Y-%m-%d'),
                'description': row['Description'],
                'ratings': {
                    'Fiabilité': row['Fiabilité Intégrité'],
                    'Disponibilité': row['Disponibilté'],
                    'Process Safety': row['Process Safety'],
                    'Criticité': row['Criticité']
                }
            })
        
        return examples, avg_ratings, equipment_data

    def _create_prompt(self, description: str, equipment_type: str, section: str) -> str:
        """Create a focused prompt with equipment history and similar incidents"""
        examples, avg_ratings, equipment_data = self._get_equipment_history(equipment_type)
        similar_incidents = self._find_similar_incidents(description, equipment_data)
        
        prompt = f"""You are an expert industrial equipment anomaly analyzer. Analyze this new anomaly based on historical data:

EQUIPMENT TYPE: {equipment_type}
SECTION: {section}

NEW ANOMALY TO ANALYZE:
Description: {description}

SIMILAR PAST INCIDENTS:"""

        if similar_incidents:
            prompt += "\nMost relevant historical cases for this type of anomaly:"
            for idx, incident in enumerate(similar_incidents, 1):
                prompt += f"""
{idx}. [Similarity: {incident['similarity']:.1%}]
Date: {incident['date']}
Description: {incident['description']}
Ratings: Fiabilité={incident['ratings']['Fiabilité']}, Disponibilité={incident['ratings']['Disponibilité']}, Safety={incident['ratings']['Process Safety']}, Criticité={incident['ratings']['Criticité']}
"""
        
        prompt += f"""
EQUIPMENT HISTORY STATISTICS:
Average ratings for this equipment type:
- Fiabilité Intégrité: {avg_ratings['Fiabilité']:.1f}
- Disponibilité: {avg_ratings['Disponibilité']:.1f}
- Process Safety: {avg_ratings['Process Safety']:.1f}
- Average Criticité: {avg_ratings['Criticité']:.1f}

Additional recent anomalies for context:"""

        # Add some recent examples that weren't in the similar incidents
        shown_dates = {inc['date'] for inc in similar_incidents}
        count = 0
        for ex in examples:
            if ex['date'] not in shown_dates and count < 5:  # Show up to 5 additional recent examples
                prompt += f"""
Date: {ex['date']}
Description: {ex['description']}
Ratings: Fiabilité={ex['ratings']['Fiabilité']}, Disponibilité={ex['ratings']['Disponibilité']}, Safety={ex['ratings']['Process Safety']}, Criticité={ex['ratings']['Criticité']}
---"""
                count += 1

        prompt += """

Based on the similar past incidents and historical patterns of this equipment, rate this new anomaly on three parameters (1-5 scale):
Fiabilité Intégrité: [1=Minor, 5=Critical]
Disponibilité: [1=Minor, 5=Critical]
Process Safety: [1=Minor, 5=Critical]

RESPOND WITH ONLY THREE NUMBERS IN THIS EXACT FORMAT:
Fiabilité=X,Disponibilité=Y,Safety=Z"""

        return prompt

    def predict(self, description: str, equipment_type: str, section: str) -> Dict[str, int]:
        """Predict parameters for a new anomaly"""
        try:
            prompt = self._create_prompt(description, equipment_type, section)
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an industrial equipment anomaly analysis expert. Respond only with the exact format requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Low temperature for more consistent predictions
                max_tokens=10     # We only need a short response
            )
            
            # Parse response
            result = response.choices[0].message.content.strip()
            
            # Extract numbers using regex
            pattern = r"Fiabilité=(\d),Disponibilité=(\d),Safety=(\d)"
            match = re.match(pattern, result)
            
            if match:
                return {
                    'Fiabilité': int(match.group(1)),
                    'Disponibilité': int(match.group(2)),
                    'Process Safety': int(match.group(3))
                }
            else:
                # Fallback values if parsing fails
                return {'Fiabilité': 3, 'Disponibilité': 3, 'Process Safety': 3}
                
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            return {'Fiabilité': 3, 'Disponibilité': 3, 'Process Safety': 3}


def main():
    """Main execution function"""
    print("Equipment Anomaly Criticality Prediction System")
    print("=" * 60)
    
    # Initialize predictors
    smart_predictor = SmartParameterPredictor()
    
    # Get list of equipment types (this would be shown in web UI dropdown)
    equipment_types = smart_predictor.get_equipment_list()
    print("\nAvailable Equipment Types:")
    print("-" * 30)
    for i, eq in enumerate(equipment_types[:10], 1):  # Show first 10 for demo
        print(f"{i}. {eq}")
    print("... and more ...")
    
    # Simulate web app workflow with a test case
    def test_equipment_specific_prediction():
        # Example: User selects "Pompe alimentaire" from dropdown
        test_equipment = "Pompe alimentaire"
        print(f"\nSelected Equipment: {test_equipment}")
        
        # Example: User enters new anomaly description
        test_description = "Vibrations anormales détectées sur le palier côté accouplement"
        test_section = "34MM"
        
        print("\nNew Anomaly Details:")
        print(f"Description: {test_description}")
        print(f"Section: {test_section}")
        
        # Get prediction with equipment-specific context
        fi, di, ps = smart_predictor.predict(
            description=test_description,
            equipment_type=test_equipment,
            section=test_section
        )
        
        print("\nPredicted Severity Scores:")
        print(f"Fiabilité Intégrité: {fi}")
        print(f"Disponibilité: {di}")
        print(f"Process Safety: {ps}")
        print(f"Overall Criticality: {fi + di + ps}")
    
    test_equipment_specific_prediction()

if __name__ == '__main__':
    main() 