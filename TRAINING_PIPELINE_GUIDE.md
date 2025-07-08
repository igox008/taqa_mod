# 🎯 Equipment Anomaly Training Pipeline Guide

## Overview

Your ML system includes a **fully operational training pipeline** that allows continuous model improvement through incremental learning with new anomaly data. The system supports training three models simultaneously:

- **Availability Model** - Equipment uptime and operational readiness (1-5 scale)
- **Fiability Model** - Equipment integrity and dependability (1-5 scale)  
- **Process Safety Model** - Safety risk assessment and hazard identification (1-5 scale)

## 🚀 Quick Start

### 1. Start the API Server
```bash
python app.py
```

### 2. Test the Training Pipeline
```bash
python test_training_pipeline.py
```

### 3. Run Full Training Example
```bash
python training_example.py
```

## 📡 API Endpoints

### POST `/train`
**Train models with new anomaly data**

**Request Format:**
```json
[
  {
    "anomaly_id": "unique_identifier",
    "description": "detailed anomaly description (max 2000 chars)",
    "equipment_name": "equipment name/description", 
    "equipment_id": "equipment UUID",
    "availability_score": 3,      // 1-5 scale
    "fiability_score": 2,         // 1-5 scale  
    "process_safety_score": 4     // 1-5 scale
  }
  // ... up to 1000 records per training session
]
```

**Response Format:**
```json
{
  "success": true,
  "message": "Incremental training completed successfully",
  "statistics": {
    "total_records_added": 3,
    "records_per_model": {
      "availability": 3,
      "fiability": 3,
      "process_safety": 3
    },
    "training_time_seconds": 45.67,
    "models_retrained": {
      "availability": true,
      "fiability": true, 
      "process_safety": true
    },
    "backup_location": "model_backups/backup_20241205_143022"
  },
  "training_session_id": 5,
  "is_training": false
}
```

### GET `/train/status`
**Check current training status and history**

**Response:**
```json
{
  "is_training": false,
  "training_sessions_completed": 5,
  "available_backups": 8,
  "last_training": {
    "timestamp": "2024-12-05T14:30:22.123456",
    "records_added": 3,
    "training_time": 45.67,
    "models_retrained": 3,
    "backup_path": "model_backups/backup_20241205_143022",
    "success": true
  }
}
```

### POST `/train/reload`
**Reload models after training to use updated versions**

**Response:**
```json
{
  "success": true,
  "message": "Models reloaded successfully",
  "models_loaded": true
}
```

### GET `/train/backups`
**List available model backups**

**Response:**
```json
{
  "backups": [
    {
      "name": "backup_20241205_143022",
      "created": "2024-12-05 14:30:22",
      "path": "model_backups/backup_20241205_143022"
    }
  ],
  "count": 1
}
```

### POST `/train/restore/<backup_name>`
**Restore models from a specific backup**

**Response:**
```json
{
  "success": true,
  "message": "Successfully restored from backup backup_20241205_143022",
  "models_reloaded": true
}
```

## 🔧 Training Pipeline Features

### Thread Safety
- **Concurrent protection**: Only one training session can run at a time
- **Lock mechanism**: Uses threading locks to prevent conflicts
- **Status tracking**: Real-time training status available via API

### Data Validation
- **Required fields**: All anomaly records must include required fields
- **Score validation**: Scores must be numeric values between 1-5
- **Data type checking**: String fields validated for type and length
- **Batch limits**: Maximum 1000 records per training session
- **Description limits**: Max 2000 characters per description

### Backup System
- **Automatic backups**: Models and data automatically backed up before training
- **Timestamped backups**: Each backup includes creation timestamp
- **Easy restoration**: Restore any previous model version via API
- **Backup listing**: View all available backups

### Model Management
- **Incremental training**: Add new data without losing existing training
- **Multi-model updates**: All three models retrained simultaneously
- **CSV integration**: New training data seamlessly added to existing CSV files
- **Model versioning**: Track training sessions and model versions

## 🎯 Integration Examples

### Python Integration
```python
import requests

def add_training_data(anomalies):
    """Add new anomaly data for training"""
    response = requests.post(
        "http://localhost:5000/train",
        json=anomalies,
        headers={"Content-Type": "application/json"}
    )
    return response.json()

# Example usage
new_anomalies = [{
    "anomaly_id": "MAINT_001",
    "description": "Pump bearing overheating during high-load operation",
    "equipment_name": "Primary Circulation Pump",
    "equipment_id": "pump-uuid-123",
    "availability_score": 2,
    "fiability_score": 1,
    "process_safety_score": 3
}]

result = add_training_data(new_anomalies)
print(f"Training completed: {result['success']}")
```

### cURL Examples
```bash
# Train with new data
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '[{
    "anomaly_id": "CURL_001",
    "description": "Equipment malfunction detected",
    "equipment_name": "Test Equipment",
    "equipment_id": "test-123",
    "availability_score": 3,
    "fiability_score": 2,
    "process_safety_score": 4
  }]'

# Check training status
curl http://localhost:5000/train/status

# Reload models
curl -X POST http://localhost:5000/train/reload

# List backups
curl http://localhost:5000/train/backups
```

## 📊 Training Workflow

### Recommended Training Process

1. **Collect New Anomaly Data**
   - Gather real anomaly observations
   - Expert scoring for availability, fiability, process safety
   - Ensure data quality and completeness

2. **Validate and Prepare Data**
   - Check required fields are present
   - Verify scores are in 1-5 range
   - Limit batch size to ≤1000 records

3. **Execute Training**
   - Send POST request to `/train` endpoint
   - Monitor training progress via status endpoint
   - Verify successful completion

4. **Reload Models** 
   - Use `/train/reload` to activate updated models
   - Test predictions to verify improvements
   - Monitor model performance

5. **Backup Management**
   - Review available backups regularly
   - Restore previous versions if needed
   - Clean up old backups periodically

### Best Practices

#### Data Quality
- **Consistent scoring**: Use standardized scoring criteria across teams
- **Rich descriptions**: Provide detailed anomaly descriptions (>50 words)
- **Representative data**: Include diverse equipment types and anomaly patterns
- **Expert validation**: Have domain experts validate scores before training

#### Training Frequency
- **Regular intervals**: Train weekly or bi-weekly with new data
- **Batch processing**: Accumulate 10-50 anomalies before training
- **Emergency updates**: Train immediately for critical safety anomalies
- **Performance monitoring**: Track model accuracy after each training session

#### Model Management
- **Test after training**: Always test predictions after retraining
- **Backup before training**: Automatic backups protect against issues
- **Version tracking**: Document training sessions and data sources
- **Rollback capability**: Keep ability to restore previous model versions

## 🚨 Error Handling

### Common Training Errors

**Validation Failures:**
```json
{
  "success": false,
  "error": "Data validation failed",
  "validation_errors": [
    "Record 1: availability_score must be between 1 and 5, got 6",
    "Record 2: Missing required fields: ['equipment_id']"
  ]
}
```

**Concurrent Training:**
```json
{
  "success": false,
  "error": "Training already in progress. Please wait and try again.",
  "is_training": true
}
```

**Model Loading Failures:**
```json
{
  "success": false,
  "message": "Failed to reload models",
  "models_loaded": false
}
```

### Troubleshooting

1. **Training Fails**
   - Check data validation errors
   - Ensure all required fields are present
   - Verify scores are in valid range (1-5)
   - Check API server logs for detailed errors

2. **Models Not Loading**
   - Verify model files exist in directory
   - Check file permissions
   - Review training completion status
   - Use backup restoration if needed

3. **Performance Issues**
   - Monitor training time for large batches
   - Consider splitting large datasets
   - Check system resources during training
   - Review backup disk space usage

## 🔄 Automation and Monitoring

### Integration with Monitoring Systems

The training pipeline can be integrated with existing monitoring systems:

```python
# Example automated training integration
import schedule
import time

def automated_training():
    """Scheduled training with accumulated anomalies"""
    # Fetch new anomalies from your monitoring system
    new_anomalies = fetch_new_anomalies_from_database()
    
    if len(new_anomalies) >= 10:  # Train when we have enough data
        result = add_training_data(new_anomalies)
        if result['success']:
            # Reload models to use updates
            requests.post("http://localhost:5000/train/reload")
            log_training_success(result)
        else:
            alert_training_failure(result)

# Schedule training every Sunday at 2 AM
schedule.every().sunday.at("02:00").do(automated_training)
```

### Performance Monitoring

Track these metrics to ensure training effectiveness:

- **Training frequency**: Number of training sessions per month
- **Data volume**: Records added per training session  
- **Training time**: Time required for model retraining
- **Model accuracy**: Prediction accuracy on test datasets
- **Backup usage**: Storage used for model versioning

## 📋 File Structure

The training system uses these key files:

```
ML/
├── app.py                          # Main Flask API with training endpoints
├── training_manager.py             # Core training logic and management
├── comprehensive_prediction_system.py  # ML model wrapper
├── disponibilite.csv              # Availability training data
├── fiabilite.csv                  # Fiability training data  
├── process_safty.csv              # Process safety training data
├── availability_model.pkl         # Trained availability model
├── fiability_model.pkl           # Trained fiability model
├── process_safety_model.pkl      # Trained process safety model
└── model_backups/                 # Automated backup directory
    ├── backup_20241205_143022/    # Timestamped backup folders
    └── backup_20241204_091234/
```

## 🎉 Success! Your Training Pipeline is Ready

Your equipment anomaly prediction system now includes:

✅ **Complete training pipeline** with all features implemented  
✅ **Thread-safe incremental learning** for continuous improvement  
✅ **Comprehensive validation** ensuring data quality  
✅ **Automatic backup system** protecting against issues  
✅ **RESTful API endpoints** for easy integration  
✅ **Model versioning** with restore capabilities  
✅ **Real-time status tracking** for monitoring  

Use the provided example scripts to start training your models with new anomaly data immediately! 