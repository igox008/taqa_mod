import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class AnomalyPriorityRegressor:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.text_vectorizer = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """Load and preprocess the anomaly dataset"""
        print("Loading dataset...")
        self.data = pd.read_csv(file_path)
        print(f"Dataset loaded with {len(self.data)} records")
        print(f"Columns: {list(self.data.columns)}")
        return self.data
    
    def explore_data(self):
        """Explore the dataset structure and statistics"""
        print("\n=== DATASET EXPLORATION ===")
        print(f"Dataset shape: {self.data.shape}")
        print("\nFirst few records:")
        print(self.data.head())
        
        print("\n=== COLUMN INFO ===")
        print(self.data.info())
        
        print("\n=== MISSING VALUES ===")
        print(self.data.isnull().sum())
        
        print("\n=== PRIORITY DISTRIBUTION ===")
        priority_counts = self.data['Priorité'].value_counts().sort_index()
        print(priority_counts)
        
        print("\n=== STATUS DISTRIBUTION ===")
        print(self.data['Statut'].value_counts())
        
        print("\n=== SECTION DISTRIBUTION ===")
        print(self.data['Section proprietaire'].value_counts().head(10))
        
        # Visualize priority distribution
        plt.figure(figsize=(10, 6))
        priority_counts.plot(kind='bar')
        plt.title('Distribution of Anomaly Priorities')
        plt.xlabel('Priority Level')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('priority_distribution.png')
        plt.show()
        
        return priority_counts
    
    def preprocess_data(self):
        """Preprocess the data for machine learning"""
        print("\n=== PREPROCESSING DATA ===")
        
        # Create a copy for preprocessing
        df = self.data.copy()
        
        # Handle missing priorities - fill with median (or 3.0)
        df['Priorité'] = df['Priorité'].fillna(3.0)
        
        # Convert priority to float
        df['Priorité'] = df['Priorité'].astype(float)
        
        # Create text features from description
        df['Description_clean'] = df['Description'].fillna('')
        
        # Create equipment type features
        df['Equipment_Type'] = df['Description equipement'].fillna('')
        
        # Create section features
        df['Section'] = df['Section proprietaire'].fillna('')
        
        # Extract date features
        df['Date de detection de l\'anomalie'] = pd.to_datetime(df['Date de detection de l\'anomalie'], errors='coerce')
        df['Year'] = df['Date de detection de l\'anomalie'].dt.year
        df['Month'] = df['Date de detection de l\'anomalie'].dt.month
        df['DayOfWeek'] = df['Date de detection de l\'anomalie'].dt.dayofweek
        
        # Create status features
        df['Status'] = df['Statut'].fillna('Unknown')
        
        # Create text length features
        df['Description_Length'] = df['Description_clean'].str.len()
        df['Equipment_Length'] = df['Equipment_Type'].str.len()
        
        # Create keyword features for priority indicators
        high_priority_keywords = ['urgent', 'safety', 'critique', 'dangereux', 'arrêt', 'défaut', 'fuite importante']
        medium_priority_keywords = ['prévoir', 'contrôle', 'maintenance', 'nettoyage', 'réparation']
        low_priority_keywords = ['amélioration', 'optimisation', 'aménagement']
        
        df['High_Priority_Keywords'] = df['Description_clean'].str.lower().str.count('|'.join(high_priority_keywords))
        df['Medium_Priority_Keywords'] = df['Description_clean'].str.lower().str.count('|'.join(medium_priority_keywords))
        df['Low_Priority_Keywords'] = df['Description_clean'].str.lower().str.count('|'.join(low_priority_keywords))
        
        # Create equipment category features
        equipment_categories = {
            'pompe': 'pump',
            'moteur': 'motor', 
            'vanne': 'valve',
            'broyeur': 'mill',
            'ventilateur': 'fan',
            'chaudière': 'boiler',
            'turbine': 'turbine',
            'alternateur': 'generator',
            'transfo': 'transformer',
            'électro': 'electrical',
            'ramoneur': 'sootblower',
            'décrasseur': 'cleaner'
        }
        
        for fr_keyword, en_category in equipment_categories.items():
            df[f'Equipment_{en_category}'] = df['Equipment_Type'].str.lower().str.contains(fr_keyword, na=False).astype(int)
        
        # Create section category features
        section_categories = {
            '34MC': 'mechanical',
            '34EL': 'electrical', 
            '34CT': 'control',
            '34MD': 'maintenance',
            '34MM': 'mechanical_maintenance',
            '34MG': 'mechanical_general'
        }
        
        for section_code, section_category in section_categories.items():
            df[f'Section_{section_category}'] = (df['Section'] == section_code).astype(int)
        
        print(f"Preprocessed data shape: {df.shape}")
        print(f"Priority distribution after preprocessing:")
        print(df['Priorité'].value_counts().sort_index())
        
        self.processed_data = df
        return df
    
    def prepare_features(self):
        """Prepare features for machine learning"""
        print("\n=== PREPARING FEATURES ===")
        
        df = self.processed_data.copy()
        
        # Select features for modeling
        text_features = ['Description_clean']
        categorical_features = ['Equipment_Type', 'Section', 'Status']
        numerical_features = [
            'Year', 'Month', 'DayOfWeek', 'Description_Length', 'Equipment_Length',
            'High_Priority_Keywords', 'Medium_Priority_Keywords', 'Low_Priority_Keywords'
        ]
        
        # Add equipment category features
        equipment_cat_features = [col for col in df.columns if col.startswith('Equipment_')]
        numerical_features.extend(equipment_cat_features)
        
        # Add section category features  
        section_cat_features = [col for col in df.columns if col.startswith('Section_')]
        numerical_features.extend(section_cat_features)
        
        # Prepare target variable
        y = df['Priorité']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            df, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        print(f"Features used: {len(numerical_features)} numerical + {len(categorical_features)} categorical + {len(text_features)} text")
        
        self.text_features = text_features
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def create_pipeline(self):
        """Create a machine learning pipeline"""
        print("\n=== CREATING REGRESSION PIPELINE ===")
        text_transformer = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words=None,
                min_df=2,
                max_df=0.95
            ))
        ])
        numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Define exact numerical columns to avoid string columns
        numerical_columns = [
            'Year', 'Month', 'DayOfWeek', 'Description_Length', 'Equipment_Length',
            'High_Priority_Keywords', 'Medium_Priority_Keywords', 'Low_Priority_Keywords',
            'Equipment_pump', 'Equipment_motor', 'Equipment_valve', 'Equipment_mill',
            'Equipment_fan', 'Equipment_boiler', 'Equipment_turbine', 'Equipment_generator',
            'Equipment_transformer', 'Equipment_electrical', 'Equipment_sootblower', 'Equipment_cleaner',
            'Section_mechanical', 'Section_electrical', 'Section_control', 'Section_maintenance',
            'Section_mechanical_maintenance', 'Section_mechanical_general'
        ]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', text_transformer, 'Description_clean'),
                ('num', numerical_transformer, numerical_columns)
            ],
            remainder='drop'
        )
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Linear Regression': LinearRegression(),
            'SVR': SVR()
        }
        self.pipelines = {}
        for name, model in models.items():
            self.pipelines[name] = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
        print(f"Created {len(self.pipelines)} regression pipelines")
        return self.pipelines
    
    def train_models(self):
        """Train all models and evaluate performance"""
        print("\n=== TRAINING REGRESSION MODELS ===")
        
        results = {}
        
        for name, pipeline in self.pipelines.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            pipeline.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = pipeline.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            cv_scores = cross_val_score(pipeline, self.X_train, self.y_train, cv=5, scoring='neg_mean_absolute_error')
            
            results[name] = {
                'pipeline': pipeline,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'cv_mean_mae': -cv_scores.mean(),
                'cv_std_mae': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"{name} - MAE: {mae:.3f}, MSE: {mse:.3f}, R2: {r2:.3f}, CV MAE: {-cv_scores.mean():.3f}")
        
        self.results = results
        return results
    
    def evaluate_models(self):
        """Evaluate and compare model performance"""
        print("\n=== MODEL EVALUATION ===")
        
        # Compare MAE
        maes = {name: result['mae'] for name, result in self.results.items()}
        
        plt.figure(figsize=(12, 6))
        
        # Plot MAE comparison
        plt.subplot(1, 2, 1)
        names = list(maes.keys())
        scores = list(maes.values())
        bars = plt.bar(names, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.title('Model MAE Comparison')
        plt.ylabel('Mean Absolute Error')
        plt.xticks(rotation=45)
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.2f}', ha='center', va='bottom')
        
        # Plot actual vs predicted for best model
        best_model_name = min(maes, key=maes.get)
        best_result = self.results[best_model_name]
        
        plt.subplot(1, 2, 2)
        plt.scatter(self.y_test, best_result['predictions'], alpha=0.5)
        plt.xlabel('Actual Priority')
        plt.ylabel('Predicted Priority')
        plt.title(f'Actual vs Predicted (Best: {best_model_name})')
        plt.plot([1, 5], [1, 5], 'r--')
        
        plt.tight_layout()
        plt.savefig('regression_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed evaluation for best model
        print(f"\n=== DETAILED EVALUATION FOR {best_model_name} ===")
        print(f"MAE: {best_result['mae']:.3f}, MSE: {best_result['mse']:.3f}, R2: {best_result['r2']:.3f}")
        
        return best_model_name
    
    def save_best_model(self, best_model_name):
        """Save the best performing model"""
        import joblib
        
        best_pipeline = self.results[best_model_name]['pipeline']
        
        # Save the model
        model_filename = f'anomaly_priority_regressor_{best_model_name.lower().replace(" ", "_")}.joblib'
        joblib.dump(best_pipeline, model_filename)
        
        print(f"\nBest model ({best_model_name}) saved as: {model_filename}")
        
        # Save model info
        model_info = {
            'model_name': best_model_name,
            'mae': self.results[best_model_name]['mae'],
            'mse': self.results[best_model_name]['mse'],
            'r2': self.results[best_model_name]['r2'],
            'features_used': best_pipeline.named_steps['preprocessor'].get_feature_names_out().tolist(),
            'priority_range': [1.0, 5.0]
        }
        
        import json
        with open('regressor_model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("Model information saved as: regressor_model_info.json")
        
        return model_filename
    
    def predict_priority(self, description, equipment_type="", section="", status="Terminé"):
        """Predict priority for a new anomaly"""
        if not hasattr(self, 'best_model'):
            print("No trained model available. Please train the model first.")
            return None
        
        # Create a sample record
        sample_data = pd.DataFrame({
            'Description': [description],
            'Description_clean': [description],
            'Equipment_Type': [equipment_type],
            'Section': [section],
            'Status': [status],
            'Date de detection de l\'anomalie': [pd.Timestamp.now()],
            'Year': [pd.Timestamp.now().year],
            'Month': [pd.Timestamp.now().month],
            'DayOfWeek': [pd.Timestamp.now().dayofweek],
            'Description_Length': [len(description)],
            'Equipment_Length': [len(equipment_type)],
            'High_Priority_Keywords': [description.lower().count('urgent') + description.lower().count('safety')],
            'Medium_Priority_Keywords': [description.lower().count('prévoir') + description.lower().count('contrôle')],
            'Low_Priority_Keywords': [description.lower().count('amélioration')]
        })
        
        # Add equipment category features
        equipment_categories = ['pump', 'motor', 'valve', 'mill', 'fan', 'boiler', 'turbine', 'generator', 'transformer', 'electrical', 'sootblower', 'cleaner']
        for category in equipment_categories:
            sample_data[f'Equipment_{category}'] = [1 if category in equipment_type.lower() else 0]
        
        # Add section category features
        section_categories = ['mechanical', 'electrical', 'control', 'maintenance', 'mechanical_maintenance', 'mechanical_general']
        for category in section_categories:
            sample_data[f'Section_{category}'] = [1 if category in section.lower() else 0]
        
        # Make prediction
        prediction = self.best_model.predict(sample_data)[0]
        prediction = float(np.clip(prediction, 1.0, 5.0))
        
        return {
            'priority_score': prediction,
            'priority_range': [1.0, 5.0]
        }

def main():
    print("=== TAQA ANOMALY PRIORITY REGRESSOR ===")
    regressor = AnomalyPriorityRegressor()
    data = regressor.load_data('data_set.csv')
    processed_data = regressor.preprocess_data()
    regressor.prepare_features()
    regressor.create_pipeline()
    results = regressor.train_models()
    best_model_name = regressor.evaluate_models()
    model_filename = regressor.save_best_model(best_model_name)
    regressor.best_model = regressor.results[best_model_name]['pipeline']
    print("\n=== EXAMPLE PREDICTIONS ===")
    test_cases = [
        {
            'description': 'SAFETY: Fuite importante de vapeur par tresse ramoneur',
            'equipment': 'RAMONEUR LONG RETRACTABLE',
            'section': '34MC'
        },
        {
            'description': 'Prévoir contrôle mesure de température moteur broyeur',
            'equipment': 'MOTEUR BROYEUR A',
            'section': '34CT'
        },
        {
            'description': 'Amélioration éclairage salle technique',
            'equipment': 'ECLAIRAGE',
            'section': '34EL'
        }
    ]
    for i, case in enumerate(test_cases, 1):
        result = regressor.predict_priority(
            case['description'], 
            case['equipment'], 
            case['section']
        )
        print(f"\nTest Case {i}:")
        print(f"Description: {case['description']}")
        print(f"Equipment: {case['equipment']}")
        print(f"Section: {case['section']}")
        print(f"Predicted Priority Score: {result['priority_score']:.2f} (Range: 1.0–5.0)")
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Best model: {best_model_name}")
    print(f"Model saved as: {model_filename}")
    print("You can now use the model to predict priority scores for new anomalies!")

if __name__ == "__main__":
    main() 