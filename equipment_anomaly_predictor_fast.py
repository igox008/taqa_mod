#!/usr/bin/env python3
"""
Equipment Parameter Prediction - Smart Implementation with OpenAI Integration
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
import os
import openai
from dotenv import load_dotenv
import traceback
import logging
import time
import re # Added for regex validation
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure OpenAI
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
logger.info(f"OpenAI API key loaded: {api_key[:8]}...")

openai.api_key = api_key

class SmartParameterPredictor:
    def __init__(self):
        """Initialize the predictor with system prompt and load historical data"""
        self.system_prompt = """You are an expert industrial equipment analyst. Your task is to:
        1. Learn from historical equipment anomaly data
        2. Use this knowledge to predict three parameters (1-5) for new cases
        3. Output only three numbers separated by commas (e.g., "4,3,5")

        The numbers represent:
        1. Fiabilité Intégrité (F): Equipment Reliability
        2. Disponibilité (D): Availability
        3. Process Safety (S): Process Safety

        Scale interpretation (sum of F+D+S):
        - Sum 1-7: Minor issues (routine maintenance)
        - Sum 8-9: Moderate issues (planned intervention)
        - Sum 10-15: Critical issues (immediate attention)"""

        # Load historical data
        try:
            self.historical_data = pd.read_csv("data_set.csv")
            logger.info(f"Loaded {len(self.historical_data)} historical records")
            logger.info(f"Dataset columns: {self.historical_data.columns.tolist()}")
            
            # Initialize conversation memory
            self.initialize_model_memory()
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            self.historical_data = pd.DataFrame()

    def chunk_data(self, chunk_size=50):
        """Split historical data into manageable chunks"""
        for i in range(0, len(self.historical_data), chunk_size):
            yield self.historical_data.iloc[i:i + chunk_size]

    def format_chunk_data(self, chunk):
        """Format a chunk of historical data"""
        chunk_context = "Learning data batch:\n"
        for _, case in chunk.iterrows():
            chunk_context += f"""Case:
            Description: {case['Description'][:100]}
            Equipment: {case['Description de l\'équipement']}
            System: {case['Systeme']}
            Section: {case['Section propriétaire']}
            Ratings(F/D/S): {int(case['Fiabilité Intégrité'])}/{int(case['Disponibilté'])}/{int(case['Process Safety'])}\n"""
        return chunk_context

    def initialize_model_memory(self):
        """Initialize the model's memory with historical data in chunks"""
        logger.info("Starting model memory initialization...")
        client = openai.OpenAI()
        
        # Calculate and store equipment type summaries
        self.equipment_stats = self.historical_data.groupby('Description de l\'équipement').agg({
            'Fiabilité Intégrité': ['mean', 'count'],
            'Disponibilté': 'mean',
            'Process Safety': 'mean',
            'Criticité': 'mean'
        }).reset_index()
        
        # Store processed chunks for each equipment type
        self.equipment_memories = {}
        
        logger.info("Model memory initialization completed")

    def get_relevant_history(self, equipment_type, max_cases=100):
        """Get relevant historical cases for a specific equipment type"""
        # Start with equipment statistics
        stats = self.equipment_stats[
            self.equipment_stats['Description de l\'équipement'].str.contains(equipment_type, case=False, na=False)
        ]
        
        context = "Equipment Statistics:\n"
        if not stats.empty:
            for _, stat in stats.iterrows():
                context += f"""Equipment: {stat['Description de l\'équipement']}
                Cases: {int(stat['Fiabilité Intégrité']['count'])}
                Avg(F/D/S): {stat['Fiabilité Intégrité']['mean']:.1f}/{stat['Disponibilté']['mean']:.1f}/{stat['Process Safety']['mean']:.1f}
                Avg Criticité: {stat['Criticité']['mean']:.1f}\n"""
        
        # Get relevant cases
        relevant_cases = self.historical_data[
            self.historical_data['Description de l\'équipement'].str.contains(equipment_type, case=False, na=False)
        ]
        
        if len(relevant_cases) == 0:
            # If no exact matches, get cases from the same system
            system = self.historical_data[
                self.historical_data['Description de l\'équipement'].str.contains(equipment_type, case=False, na=False)
            ]['Systeme'].iloc[0] if not self.historical_data.empty else None
            
            if system:
                relevant_cases = self.historical_data[
                    self.historical_data['Systeme'] == system
                ]
        
        if len(relevant_cases) > 0:
            # Get most recent cases
            recent_cases = relevant_cases.tail(min(20, len(relevant_cases)))
            context += "\nRecent Cases:\n"
            for _, case in recent_cases.iterrows():
                context += f"""Description: {case['Description'][:100]}...
                Equipment: {case['Description de l\'équipement']}
                F/D/S: {int(case['Fiabilité Intégrité'])}/{int(case['Disponibilté'])}/{int(case['Process Safety'])}
                Criticité: {int(case['Criticité'])}\n"""
            
            # Get highest criticality cases
            critical_cases = relevant_cases.nlargest(10, 'Criticité')
            context += "\nMost Critical Cases:\n"
            for _, case in critical_cases.iterrows():
                context += f"""Description: {case['Description'][:100]}...
                Equipment: {case['Description de l\'équipement']}
                F/D/S: {int(case['Fiabilité Intégrité'])}/{int(case['Disponibilté'])}/{int(case['Process Safety'])}
                Criticité: {int(case['Criticité'])}\n"""
        
        return context

    def predict(self, description, equipment_type, section):
        """Predict parameters using OpenAI's reasoning capabilities with relevant historical data"""
        try:
            # Get relevant historical context
            historical_context = self.get_relevant_history(equipment_type)
            
            # Construct the analysis prompt
            analysis_prompt = f"""You must respond with exactly three numbers between 1 and 5, separated by commas. Nothing else.
            Example valid response: "3,4,2"

            Analyze this new case and predict three severity ratings (1-5) for Fiabilité, Disponibilité, and Process Safety:

            New Case:
            Description: {description}
            Equipment: {equipment_type}
            Section: {section}

            Historical Context:
            {historical_context}

            Rules:
            1. ONLY output three numbers between 1-5 separated by commas
            2. No text, no explanations
            3. Format must be exactly: X,Y,Z where X,Y,Z are single digits 1-5

            Response (NUMBERS ONLY):"""

            logger.info("Sending prediction request to OpenAI")
            
            # Get prediction from OpenAI
            client = openai.OpenAI()
            messages = [
                {"role": "system", "content": "You are a precise prediction system. You MUST ONLY output three numbers between 1-5 separated by commas. Example: '3,4,2'. ANY OTHER FORMAT IS FORBIDDEN."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0,
                max_tokens=10
            )

            # Extract the three numbers from the response
            result = response.choices[0].message.content.strip()
            logger.debug(f"Raw result from OpenAI: {result}")
            
            # Validate the response format
            if not re.match(r'^[1-5],[1-5],[1-5]$', result):
                logger.error(f"Invalid response format: {result}")
                raise ValueError(f"Invalid response format from OpenAI: {result}")
            
            # Clean and parse the result
            fiabilite, disponibilite, safety = map(int, result.split(','))
            logger.info(f"Parsed values: F={fiabilite}, D={disponibilite}, S={safety}")

            # Calculate total score and get severity
            total_score = fiabilite + disponibilite + safety
            severity = self.get_severity_level(total_score)
            logger.info(f"Total score: {total_score}, Severity: {severity}")

            return {
                'Fiabilité': fiabilite,
                'Disponibilité': disponibilite,
                'Process Safety': safety,
                'total_score': total_score,
                'severity': severity
            }

        except Exception as e:
            logger.error(f"Error during OpenAI prediction: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Full error details: {traceback.format_exc()}")
            
            # Fallback to moderate predictions
            return {
                'Fiabilité': 3,
                'Disponibilité': 3,
                'Process Safety': 3,
                'total_score': 9,
                'severity': 'MODÉRÉ - Planifier intervention'
            }

    def get_severity_level(self, total_score):
        """Determine severity level based on total score"""
        if total_score <= 7:
            return "MINEUR - Maintenance de routine"
        elif total_score <= 9:
            return "MODÉRÉ - Planifier intervention"
        else:
            return "CRITIQUE - Intervention immédiate requise"

class FastParameterPredictor:
    def __init__(self):
        self.model = None
        self.tfidf = None
        self.scaler = None
        self.label_encoders = {}
        
    def save_model(self, filepath):
        """Save the trained model and all necessary components"""
        import joblib
        model_components = {
            'model': self.model,
            'tfidf': self.tfidf,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        joblib.dump(model_components, filepath)
        
    def load_model(self, filepath):
        """Load a trained model and all necessary components"""
        import joblib
        components = joblib.load(filepath)
        self.model = components['model']
        self.tfidf = components['tfidf']
        self.scaler = components['scaler']
        self.label_encoders = components['label_encoders']
        
    def prepare_data(self, data):
        """Prepare features from input data"""
        # Text features with enhanced severity indicators
        severity_terms = {
            'critical': ['fuite importante', 'fuite majeure', 'rupture', 'casse', 'urgent', 'critique',
                        'défaillance majeure', 'arrêt immédiat', 'danger', 'risque majeur'],
            'safety': ['sécurité', 'danger', 'risque', 'protection', 'alarme', 'incendie', 'explosion',
                      'toxique', 'contamination', 'fuite dangereuse', 'haute pression'],
            'availability': ['arrêt', 'blocage', 'indisponible', 'panne', 'hors service', 'défaut',
                           'non fonctionnel', 'maintenance requise'],
            'reliability': ['usure', 'dégradation', 'vibration', 'bruit anormal', 'surchauffe',
                          'corrosion', 'fissure', 'fuite mineure']
        }
        
        # Create TF-IDF with enhanced features
        self.tfidf = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 3),  # Capture phrases up to 3 words
            stop_words=['le', 'la', 'les', 'de', 'du', 'des', 'un', 'une']
        )
        text_features = self.tfidf.fit_transform(data['Description'].fillna(''))
        
        # Add severity scores based on keyword matches
        severity_scores = pd.DataFrame(index=data.index)
        for category, terms in severity_terms.items():
            pattern = '|'.join(terms)
            severity_scores[f'{category}_score'] = data['Description'].fillna('').str.lower().str.contains(pattern, regex=True, na=False).astype(int)
        
        # Equipment description encoding
        self.label_encoders['Description de l\'équipement'] = LabelEncoder()
        equipment_encoded = self.label_encoders['Description de l\'équipement'].fit_transform(
            data['Description de l\'équipement'].fillna('UNKNOWN')
        )
        
        # Section encoding
        self.label_encoders['Section propriétaire'] = LabelEncoder()
        section_encoded = self.label_encoders['Section propriétaire'].fit_transform(
            data['Section propriétaire'].fillna('UNKNOWN')
        )
        
        # Combine all features
        features = np.hstack([
            text_features.toarray(),
            equipment_encoded.reshape(-1, 1),
            section_encoded.reshape(-1, 1),
            severity_scores.values
        ])
        
        # Scale features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Prepare targets
        targets = data[['Fiabilité Intégrité', 'Disponibilté', 'Process Safety']]
        
        return features_scaled, targets
    
    def train(self, features, targets):
        """Train the multi-output classifier"""
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model = MultiOutputClassifier(base_model)
        self.model.fit(features, targets)
        
    def predict(self, description, equipment_type, section):
        """Predict parameters for new input"""
        # Prepare text features
        text_features = self.tfidf.transform([description]).toarray()
        
        # Encode equipment description
        try:
            equipment_encoded = self.label_encoders['Description de l\'équipement'].transform([equipment_type])
        except (KeyError, ValueError):
            print(f"Warning: Unknown equipment description '{equipment_type}'. Using default value.")
            equipment_encoded = np.array([0])
        
        # Encode section
        try:
            section_encoded = self.label_encoders['Section propriétaire'].transform([section])
        except (KeyError, ValueError):
            print(f"Warning: Unknown section '{section}'. Using default value.")
            section_encoded = np.array([0])
        
        # Calculate severity scores
        severity_scores = []
        severity_terms = {
            'critical': ['fuite importante', 'fuite majeure', 'rupture', 'casse', 'urgent'],
            'safety': ['sécurité', 'danger', 'risque', 'protection', 'alarme'],
            'availability': ['arrêt', 'blocage', 'indisponible', 'panne', 'hors service'],
            'reliability': ['usure', 'dégradation', 'vibration', 'bruit anormal', 'surchauffe']
        }
        
        desc_lower = description.lower()
        for category, terms in severity_terms.items():
            score = any(term in desc_lower for term in terms)
            severity_scores.append(1 if score else 0)
        
        # Combine features
        features = np.hstack([
            text_features,
            equipment_encoded.reshape(-1, 1),
            section_encoded.reshape(-1, 1),
            np.array(severity_scores).reshape(1, -1)
        ])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        predictions = self.model.predict(features_scaled)
        
        # Apply business rules for critical situations
        if any(term in desc_lower for term in severity_terms['critical']):
            predictions[0] = np.maximum(predictions[0], [4, 4, 4])  # Ensure high scores for critical issues
        
        if any(term in desc_lower for term in severity_terms['safety']):
            predictions[0][2] = max(predictions[0][2], 4)  # Ensure high Process Safety for safety issues
        
        return {
            'Fiabilité': int(predictions[0][0]),
            'Disponibilité': int(predictions[0][1]),
            'Process Safety': int(predictions[0][2])
        }

def main():
    """Train and save the parameter prediction models"""
    print("Loading data...")
    data = pd.read_csv('data_set.csv')
    
    print("\n=== Training Fast Predictor ===")
    predictor = FastParameterPredictor()
    
    print("Preparing data...")
    features, targets = predictor.prepare_data(data)
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    
    print("Training model...")
    predictor.train(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = predictor.model.predict(X_test)
    params = ['Fiabilité', 'Disponibilité', 'Process Safety']
    for i, param in enumerate(params):
        acc = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
        print(f"{param}: {acc:.3f} accuracy")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    print("\nSaving model...")
    model_path = 'models/parameter_predictor.joblib'
    predictor.save_model(model_path)
    print(f"Model saved to {model_path}")

    print("\n=== Testing Smart Predictor ===")
    smart_predictor = SmartParameterPredictor()
    
    # Test case
    test_description = "Fuite importante d'huile sur le joint principal de la pompe P201, vibrations anormales et température élevée"
    test_equipment = "POMPE DE CIRCULATION P201"
    test_section = "34MC"
    
    print("\nTest Case:")
    print(f"Description: {test_description}")
    print(f"Equipment: {test_equipment}")
    print(f"Section: {test_section}")
    
    print("\nPredictions:")
    print("Fast Predictor:", predictor.predict(test_description, test_equipment, test_section))
    print("Smart Predictor:", smart_predictor.predict(test_description, test_equipment, test_section))

if __name__ == "__main__":
    main() 