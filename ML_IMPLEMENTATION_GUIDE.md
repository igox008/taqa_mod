# 🚀 ML Implementation Guide: Availability Prediction System

## 📋 Overview

This system predicts equipment availability (disponibilité) based on:
- **Equipment historical performance** 
- **Anomaly description analysis**
- **92 severe keywords** with severity scores
- **23 engineered features**

## 🧠 ML Logic & Feature Engineering

### 🔧 Core Architecture

```python
# Main Pipeline
User Input → Feature Extraction → ML Model → Availability Prediction (1-5 scale)
```

### 📊 Feature Categories (23 Features Total)

#### **1. Equipment-Based Features (6 features)**
- `equipment_historical_avg`: Historical average score from your data
- `equipment_risk_level`: Risk category (1=Low, 2=Medium, 3=High, 4=Critical)
- `is_turbine`, `is_pump`, `is_motor`, `is_valve`, `is_sensor`, `is_electrical`: Equipment type indicators

#### **2. Description Analysis Features (8 features)**
- `severe_words_count`: Number of severe keywords found
- `severe_words_avg_score`: Average severity score of found words
- `severe_words_min_score` / `severe_words_max_score`: Min/max severity scores
- `weighted_severity_score`: Importance-weighted severity calculation
- `description_length` / `description_word_count`: Text characteristics
- `urgency_score`: Count of urgency indicators

#### **3. Categorical Issue Features (4 features)**
- `has_temperature_issue`: Temperature-related problems (surchauffe, température, etc.)
- `has_mechanical_issue`: Mechanical problems (vibration, bruit, usure, etc.)
- `has_electrical_issue`: Electrical problems (alarme, court-circuit, etc.)
- `has_leakage_issue`: Leakage problems (fuite, étanchéité, etc.)

#### **4. Interaction Features (5 features)**
- `equipment_severity_interaction`: Equipment score × Severity score
- `high_risk_amplifier`: Binary flag for high-risk combinations
- `combined_risk_score`: Weighted risk calculation

## 🎯 How the Prediction Works

### Step 1: Input Processing
```python
def predict_availability(description, equipment_name, equipment_id):
    # Extract 23 features from input
    features = extract_features(description, equipment_name, equipment_id)
    
    # Feed to trained Random Forest model
    prediction = model.predict([features])
    
    return prediction  # 1-5 scale
```

### Step 2: Feature Extraction Logic

#### **Equipment Risk Assessment**
```python
# Look up historical performance
historical_avg = equipment_scores.get(equipment_id, 2.5)

# Categorize risk level
if historical_avg >= 4.0: risk_level = 1  # Low risk
elif historical_avg >= 3.0: risk_level = 2  # Medium risk  
elif historical_avg >= 2.0: risk_level = 3  # High risk
else: risk_level = 4  # Critical risk
```

#### **Severity Analysis**
```python
# Find severe keywords in description
found_words = []
for word in severe_keywords:
    if word in description.lower():
        found_words.append(word)
        
# Calculate weighted severity
severity_weights = {'panne': 5, 'urgent': 4, 'surchauffe': 3, 'alarme': 2}
weighted_score = sum(word_score * weight for word, weight in found_words)
```

#### **Combined Risk Calculation**
```python
combined_risk = (
    (5 - equipment_historical_avg) * 0.4 +  # Equipment risk
    (5 - severe_words_avg_score) * 0.4 +    # Description risk  
    severe_words_count * 0.2                # Severity count
)
```

## 🛠️ Implementation Steps

### 1. **Setup and Training**
```python
# Initialize system
predictor = AvailabilityPredictor()

# Load your processed data
predictor.load_historical_data()  # Uses equipment_simple.csv + severe_words_simple.csv

# Train model on historical data
X, y = predictor.prepare_training_data('disponibilite.csv')
predictor.train_model(X, y)

# Save for production use
predictor.save_model('availability_model.pkl')
```

### 2. **Production Usage**
```python
# Load trained model
predictor = AvailabilityPredictor()
predictor.load_model('availability_model.pkl')

# Make predictions
availability, features, explanation = predictor.predict_availability(
    description="Fuite importante avec surchauffe du moteur",
    equipment_name="POMPE HYDRAULIQUE", 
    equipment_id="876b208c-a21b-49d8-9a44-9fbe67dbfd5c"
)

# Results
print(f"Predicted Availability: {availability:.2f}/5")
```

## 📈 Model Performance

- **Test MAE**: 0.209 (Very accurate predictions)
- **Test RMSE**: 0.481 (Low prediction error)
- **Algorithm**: Random Forest Regressor (100 trees)
- **Training Data**: 6,344 historical records

### 🔝 Most Important Features
1. `description_length` (30.1%)
2. `equipment_historical_avg` (16.7%)  
3. `equipment_risk_level` (13.8%)
4. `description_word_count` (10.2%)
5. `combined_risk_score` (2.1%)

## 🎨 Integration Examples

### **API Integration**
```python
@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.json
    
    prediction, _, explanation = predictor.predict_availability(
        data['description'],
        data['equipment_name'],
        data['equipment_id']
    )
    
    return {
        'availability': round(prediction, 2),
        'risk_level': explanation['equipment_risk'],
        'severe_words': explanation['severe_words_found'],
        'recommendation': get_recommendation(prediction)
    }
```

### **Batch Processing**
```python
def process_batch(equipment_list):
    results = []
    for equipment in equipment_list:
        prediction, _, explanation = predictor.predict_availability(
            equipment['description'],
            equipment['name'],
            equipment['id']
        )
        results.append({
            'equipment_id': equipment['id'],
            'predicted_availability': prediction,
            'risk_assessment': explanation
        })
    return results
```

## 🚦 Risk Interpretation

| Availability Score | Status | Action Required |
|-------------------|--------|-----------------|
| 4.0 - 5.0 | 🟢 LOW RISK | Normal monitoring |
| 3.0 - 3.9 | 🟡 MEDIUM RISK | Schedule maintenance |
| 2.0 - 2.9 | 🟠 HIGH RISK | Priority maintenance |
| 1.0 - 1.9 | 🔴 CRITICAL RISK | Immediate action |

## 🔄 Continuous Improvement

### **Model Retraining**
```python
# Add new data and retrain periodically
new_data = load_new_anomaly_data()
X_new, y_new = predictor.prepare_training_data(new_data)
predictor.train_model(X_new, y_new)
predictor.save_model('availability_model_v2.pkl')
```

### **Feature Enhancement**
- Add seasonal patterns
- Include maintenance history
- Incorporate external factors (weather, load, etc.)
- Add equipment age/usage metrics

## ⚡ Quick Start Code

```python
from ml_feature_engine import AvailabilityPredictor

# One-time setup
predictor = AvailabilityPredictor()
predictor.load_historical_data()
X, y = predictor.prepare_training_data()
predictor.train_model(X, y)
predictor.save_model()

# Production usage
predictor = AvailabilityPredictor()
predictor.load_model()

# Predict availability
availability = predictor.predict_availability(
    "Your anomaly description here",
    "Equipment name",
    "equipment-id-123"
)[0]

print(f"Predicted Availability: {availability:.2f}/5")
```

## 📝 Files Created

1. **`ml_feature_engine.py`** - Complete ML pipeline
2. **`simple_usage_example.py`** - Usage examples  
3. **`equipment_simple.csv`** - Equipment data (ID, description, avg score)
4. **`severe_words_simple.csv`** - Severe words (word, avg score, occurrences)
5. **`availability_model.pkl`** - Trained model (auto-generated)

Your system is now ready for production use! 🚀 