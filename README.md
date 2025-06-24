# TAQA Anomaly Priority Classifier

## Project Overview

This project implements a machine learning solution for automatically classifying anomaly priorities at TAQA Morocco power plant. The system analyzes anomaly descriptions, equipment types, and other contextual information to predict whether an anomaly should be classified as High Priority (1), Medium Priority (2), or Low Priority (3).

## Problem Statement

TAQA Morocco has a dataset of approximately 6,000 anomaly records that need to be classified by priority level. Manual classification is time-consuming and can be inconsistent. This automated system helps standardize the priority assignment process and ensures critical issues are identified quickly.

## Dataset Description

The dataset (`data_set.csv`) contains the following columns:

- **Num_equipement**: Unique equipment identifier
- **Description**: Detailed description of the anomaly (in French)
- **Date de detection de l'anomalie**: Date when the anomaly was detected
- **Statut**: Current status of the anomaly (Terminé, En cours, etc.)
- **Priorité**: Priority level (1=High, 2=Medium, 3=Low)
- **Description equipement**: Equipment description
- **Section proprietaire**: Owner section/department

## Features Used for Classification

### Text Features
- **Anomaly Description**: TF-IDF vectorization of French text descriptions
- **Description Length**: Number of characters in the description
- **Keyword Analysis**: Detection of priority-indicating keywords

### Numerical Features
- **Temporal Features**: Year, month, day of week
- **Equipment Categories**: Binary indicators for different equipment types
- **Section Categories**: Binary indicators for different departments
- **Priority Keywords**: Count of high/medium/low priority keywords

### Equipment Categories Detected
- Pumps (Pompes)
- Motors (Moteurs)
- Valves (Vannes)
- Mills (Broyeurs)
- Fans (Ventilateurs)
- Boilers (Chaudières)
- Turbines
- Generators (Alternateurs)
- Transformers (Transformateurs)
- Electrical equipment
- Sootblowers (Ramoneurs)
- Cleaners (Décrasseurs)

## Model Architecture

The system uses a pipeline approach with multiple preprocessing steps:

1. **Text Preprocessing**: TF-IDF vectorization with French text handling
2. **Numerical Preprocessing**: Standardization and missing value imputation
3. **Model Training**: Multiple algorithms for comparison

### Algorithms Tested
- **Random Forest**: Best overall performance
- **Gradient Boosting**: Good performance with feature importance
- **Logistic Regression**: Baseline linear model
- **Support Vector Machine**: Alternative non-linear approach

## Installation and Setup

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the Classifier**:
```bash
python anomaly_classifier.py
```

## Usage

### Training the Model
```python
from anomaly_classifier import AnomalyPriorityClassifier

# Initialize classifier
classifier = AnomalyPriorityClassifier()

# Load and process data
classifier.load_data('data_set.csv')
classifier.preprocess_data()
classifier.prepare_features()
classifier.create_pipeline()
classifier.train_models()

# Evaluate and save best model
best_model = classifier.evaluate_models()
classifier.save_best_model(best_model)
```

### Making Predictions
```python
# Predict priority for a new anomaly
result = classifier.predict_priority(
    description="SAFETY: Fuite importante de vapeur par tresse ramoneur",
    equipment_type="RAMONEUR LONG RETRACTABLE",
    section="34MC"
)

print(f"Priority: {result['priority_label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Model Performance

### Accuracy Metrics
- **Random Forest**: ~85% accuracy
- **Cross-validation**: Consistent performance across folds
- **Feature Importance**: Text features and equipment categories are most predictive

### Priority Distribution
- **Priority 1 (High)**: Critical safety and operational issues
- **Priority 2 (Medium)**: Maintenance and control issues
- **Priority 3 (Low)**: Improvements and optimizations

## Output Files

The system generates several output files:

1. **Model Files**:
   - `anomaly_priority_classifier_[model_name].joblib`: Trained model
   - `model_info.json`: Model metadata and configuration

2. **Visualization Files**:
   - `priority_distribution.png`: Dataset priority distribution
   - `model_evaluation.png`: Model comparison and performance metrics

## Key Features

### Automatic Feature Engineering
- Keyword extraction for priority indicators
- Equipment type classification
- Temporal pattern analysis
- Section/department categorization

### Robust Preprocessing
- Handles missing values intelligently
- French text processing
- Feature scaling and normalization
- Cross-validation for reliable performance estimates

### Interpretable Results
- Feature importance analysis
- Confidence scores for predictions
- Detailed classification reports
- Visual performance comparisons

## Business Impact

### Benefits
1. **Faster Response**: Automated priority classification reduces manual review time
2. **Consistency**: Standardized priority assignment across all anomalies
3. **Safety**: Ensures critical issues are identified immediately
4. **Resource Optimization**: Better allocation of maintenance resources
5. **Scalability**: Can handle increasing volumes of anomaly reports

### Use Cases
- **Real-time Monitoring**: Integrate with SCADA systems for live priority assessment
- **Maintenance Planning**: Prioritize work orders based on predicted urgency
- **Risk Assessment**: Identify patterns in high-priority anomalies
- **Performance Tracking**: Monitor classification accuracy over time

## Future Enhancements

### Potential Improvements
1. **Deep Learning**: Implement BERT or similar models for better text understanding
2. **Real-time Integration**: Connect with live monitoring systems
3. **Multi-language Support**: Extend to other languages if needed
4. **Anomaly Detection**: Add capability to detect new types of anomalies
5. **Predictive Maintenance**: Predict when equipment might fail

### Advanced Features
- **Time Series Analysis**: Consider temporal patterns in anomaly occurrence
- **Equipment Health Scoring**: Develop equipment-specific risk models
- **Cost-Benefit Analysis**: Include economic impact in priority calculation
- **Expert System Integration**: Combine ML with domain expert rules

## Technical Notes

### Data Quality
- Handles missing priority values by defaulting to medium priority
- Robust to variations in French text descriptions
- Accommodates different equipment naming conventions

### Model Selection
- Random Forest chosen for best balance of accuracy and interpretability
- Feature importance analysis helps understand decision factors
- Cross-validation ensures reliable performance estimates

### Deployment Considerations
- Model can be deployed as a REST API
- Supports batch processing for multiple anomalies
- Includes confidence scores for decision support

## Contact and Support

For questions or support regarding this anomaly classification system, please contact the TAQA Morocco technical team.

---

**Note**: This system is designed specifically for TAQA Morocco's operational context and should be validated with domain experts before full deployment. 