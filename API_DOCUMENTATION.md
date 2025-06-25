# 🚀 TAQA Anomaly Priority API Documentation

## Overview

The TAQA Anomaly Priority API provides intelligent priority calculation for maintenance anomalies at TAQA Morocco power plant. It uses a hybrid system combining historical equipment data lookup with smart text analysis to determine anomaly priorities.

## 🌐 Base URLs

- **Development**: `http://localhost:8000`
- **Production**: `https://your-app.ondigitalocean.app`

## 🔐 Authentication

Currently no authentication required. The API is ready for integration with your existing systems.

---

## 📋 API Endpoints

### 1. Single Anomaly Priority Calculation

**Endpoint**: `POST /api/v1/calculate_priority`

Calculate priority for a single anomaly.

#### Request

```json
{
    "description": "Description of the anomaly in French",
    "equipment": "Equipment type (optional)",
    "department": "Department/section code (optional)"
}
```

#### Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `description` | string | ✅ Yes | Anomaly description in French |
| `equipment` | string | ❌ No | Equipment type (e.g., "POMPE", "ALTERNATEUR") |
| `department` | string | ❌ No | Department code (e.g., "34MC", "34EL") |

#### Response

```json
{
    "status": "success",
    "priority_score": 2.5,
    "priority_label": "Medium Priority",
    "confidence": 0.85,
    "method": "Equipment Lookup",
    "explanation": "Based on historical data for pump equipment",
    "color": "#ffa500",
    "urgency": "Normal",
    "processing_time_ms": 12,
    "input_data": {
        "description": "Pompe fait du bruit",
        "equipment": "POMPE",
        "department": "34MC"
    },
    "timestamp": "2024-01-15 14:30:25 UTC"
}
```

#### Example Usage

```bash
curl -X POST "https://your-app.ondigitalocean.app/api/v1/calculate_priority" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Pompe alimentaire principale fait du bruit anormal et vibrations importantes",
    "equipment": "POMPE ALIMENTAIRE PRINCIPALE",
    "department": "34MC"
  }'
```

```python
import requests

url = "https://your-app.ondigitalocean.app/api/v1/calculate_priority"
data = {
    "description": "Panne électrique sur alternateur unité 2 - arrêt immédiat requis",
    "equipment": "ALTERNATEUR UNITE 2",
    "department": "34EL"
}

response = requests.post(url, json=data)
result = response.json()
print(f"Priority: {result['priority_score']} - {result['priority_label']}")
```

---

### 2. Batch Anomaly Processing

**Endpoint**: `POST /api/v1/batch_calculate`

Process multiple anomalies in a single request (max 100 anomalies).

#### Request

```json
{
    "anomalies": [
        {
            "id": "unique_identifier_1",
            "description": "First anomaly description",
            "equipment": "POMPE",
            "department": "34MC"
        },
        {
            "id": "unique_identifier_2",
            "description": "Second anomaly description",
            "equipment": "ALTERNATEUR",
            "department": "34EL"
        }
    ]
}
```

#### Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `anomalies` | array | ✅ Yes | Array of anomaly objects (max 100) |
| `anomalies[].id` | string | ❌ No | Unique identifier for tracking |
| `anomalies[].description` | string | ✅ Yes | Anomaly description |
| `anomalies[].equipment` | string | ❌ No | Equipment type |
| `anomalies[].department` | string | ❌ No | Department code |

#### Response

```json
{
    "status": "completed",
    "total_processed": 4,
    "successful": 4,
    "failed": 0,
    "results": [
        {
            "id": "ANOM_001",
            "status": "success",
            "priority_score": 2.9,
            "priority_label": "High Priority",
            "confidence": 0.85,
            "method": "Equipment Lookup",
            "explanation": "Historical data for transformer equipment",
            "color": "#ff6b6b",
            "urgency": "High"
        }
    ],
    "errors": [],
    "processing_time_ms": 45,
    "timestamp": "2024-01-15 14:30:25 UTC"
}
```

#### Example Usage

```python
import requests

url = "https://your-app.ondigitalocean.app/api/v1/batch_calculate"
data = {
    "anomalies": [
        {
            "id": "ANOM_001",
            "description": "Transformateur en surchauffe",
            "equipment": "TRANSFORMATEUR",
            "department": "34EL"
        },
        {
            "id": "ANOM_002",
            "description": "Maintenance préventive programmée",
            "equipment": "POMPE",
            "department": "34MM"
        }
    ]
}

response = requests.post(url, json=data)
result = response.json()

for anomaly in result['results']:
    print(f"{anomaly['id']}: Priority {anomaly['priority_score']}")
```

---

### 3. API Information

**Endpoint**: `GET /api/v1/info`

Get API documentation and system information.

#### Response

```json
{
    "api_name": "TAQA Anomaly Priority Calculator",
    "version": "1.0",
    "status": "active",
    "endpoints": {
        "/api/v1/calculate_priority": {
            "method": "POST",
            "description": "Calculate priority for a single anomaly"
        }
    },
    "supported_departments": [
        "34MC - Mechanical Coal",
        "34EL - Electrical"
    ],
    "priority_scale": {
        "1.0-2.0": "Low Priority (Green)",
        "2.0-3.0": "Medium Priority (Orange)",
        "3.0-4.0": "High Priority (Red)"
    },
    "accuracy": {
        "known_equipment": "82%",
        "text_analysis": "67.8%"
    }
}
```

---

### 4. Health Check

**Endpoint**: `GET /health`

Check API health and system status.

#### Response

```json
{
    "status": "healthy",
    "model": "Simple Reliable System",
    "lookup_accuracy": "82%",
    "text_analysis": "Smart keyword detection"
}
```

---

## 📊 Priority Scale

| Score Range | Label | Color | Description |
|-------------|-------|-------|-------------|
| 1.0 - 2.0 | Low Priority | 🟢 Green | Routine maintenance, non-critical |
| 2.0 - 3.0 | Medium Priority | 🟠 Orange | Standard priority, schedule soon |
| 3.0 - 4.0 | High Priority | 🔴 Red | Urgent, address quickly |
| 4.0+ | Critical Priority | 🔴 Dark Red | Emergency, immediate attention |

## 🏭 Supported Departments

| Code | Description |
|------|-------------|
| 34MC | Mechanical Coal |
| 34EL | Electrical |
| 34CT | Control |
| 34MD | Mechanical Diesel |
| 34MM | Mechanical Maintenance |
| 34MG | Mechanical General |

## 🔧 Supported Equipment Types

The system recognizes 18+ equipment types including:
- POMPE (Pumps)
- ALTERNATEUR (Generators)
- TRANSFORMATEUR (Transformers)
- CHAUDIERE (Boilers)
- VENTILATEUR (Fans)
- ECLAIRAGE (Lighting)
- And more...

## 🎯 Accuracy & Performance

- **Known Equipment**: 82% accuracy via historical lookup
- **Text Analysis**: 67.8% accuracy using French keywords
- **Response Time**: ~10-50ms per anomaly
- **Memory Usage**: <50MB RAM
- **Batch Limit**: 100 anomalies per request

## ❌ Error Handling

### Common Error Responses

#### 400 Bad Request
```json
{
    "error": "Description field is required",
    "status": "error",
    "required_fields": ["description"],
    "optional_fields": ["equipment", "department"]
}
```

#### 500 Internal Server Error
```json
{
    "error": "TAQA classifier not initialized",
    "status": "error",
    "processing_time_ms": 5,
    "timestamp": "2024-01-15 14:30:25 UTC"
}
```

## 🚀 Integration Examples

### JavaScript/Node.js

```javascript
const axios = require('axios');

async function calculatePriority(description, equipment, department) {
    try {
        const response = await axios.post(
            'https://your-app.ondigitalocean.app/api/v1/calculate_priority',
            {
                description,
                equipment,
                department
            }
        );
        
        return response.data;
    } catch (error) {
        console.error('API Error:', error.response.data);
        throw error;
    }
}

// Usage
calculatePriority(
    "Pompe fait du bruit anormal",
    "POMPE",
    "34MC"
).then(result => {
    console.log(`Priority: ${result.priority_score} (${result.priority_label})`);
});
```

### PHP

```php
<?php
function calculatePriority($description, $equipment = '', $department = '') {
    $url = 'https://your-app.ondigitalocean.app/api/v1/calculate_priority';
    
    $data = [
        'description' => $description,
        'equipment' => $equipment,
        'department' => $department
    ];
    
    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, $url);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
    curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    
    $response = curl_exec($ch);
    curl_close($ch);
    
    return json_decode($response, true);
}

// Usage
$result = calculatePriority(
    "Transformateur en surchauffe",
    "TRANSFORMATEUR", 
    "34EL"
);

echo "Priority: " . $result['priority_score'] . " (" . $result['priority_label'] . ")";
?>
```

### C#

```csharp
using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

public class TAQAClient
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;
    
    public TAQAClient(string baseUrl)
    {
        _httpClient = new HttpClient();
        _baseUrl = baseUrl;
    }
    
    public async Task<dynamic> CalculatePriorityAsync(string description, string equipment = "", string department = "")
    {
        var data = new
        {
            description = description,
            equipment = equipment,
            department = department
        };
        
        var json = JsonConvert.SerializeObject(data);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        
        var response = await _httpClient.PostAsync($"{_baseUrl}/api/v1/calculate_priority", content);
        var responseContent = await response.Content.ReadAsStringAsync();
        
        return JsonConvert.DeserializeObject(responseContent);
    }
}

// Usage
var client = new TAQAClient("https://your-app.ondigitalocean.app");
var result = await client.CalculatePriorityAsync(
    "Panne électrique sur alternateur",
    "ALTERNATEUR",
    "34EL"
);

Console.WriteLine($"Priority: {result.priority_score} ({result.priority_label})");
```

## 📞 Support

For integration support or questions:
1. Check `/api/v1/info` endpoint for latest documentation
2. Use `/health` endpoint to verify system status
3. Test with the provided test script: `python test_api_endpoints.py`

## 🔄 Rate Limiting

Currently no rate limiting implemented. For production use, consider:
- Maximum 100 anomalies per batch request
- Reasonable request frequency
- Monitor system resources

---

**Ready to integrate? Start with a simple test call to `/api/v1/info` to verify connectivity!** 🚀 