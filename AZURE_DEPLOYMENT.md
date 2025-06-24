# 🚀 Azure Deployment Guide - TAQA Anomaly Classifier

This guide provides step-by-step instructions to deploy your TAQA Anomaly Priority Classifier to Microsoft Azure.

## 📋 Prerequisites

1. **Azure Account**: Sign up at [azure.microsoft.com](https://azure.microsoft.com)
2. **Azure CLI**: Install from [docs.microsoft.com/cli/azure/install-azure-cli](https://docs.microsoft.com/cli/azure/install-azure-cli)
3. **Git**: For code deployment

## 🎯 Deployment Options

### Option 1: Azure App Service (Recommended)
**Cost**: ~$13-55/month | **Best for**: Production web applications

### Option 2: Azure Container Instances
**Cost**: ~$30-100/month | **Best for**: Flexible scaling

### Option 3: Azure Functions
**Cost**: ~$0-20/month | **Best for**: API-only, serverless

---

## 🚀 Option 1: Azure App Service Deployment

### Step 1: Login to Azure
```bash
az login
```

### Step 2: Create Resource Group
```bash
az group create --name taqa-classifier-rg --location "East US"
```

### Step 3: Deploy Using ARM Template
```bash
az deployment group create \
  --resource-group taqa-classifier-rg \
  --template-file azure-deploy.json \
  --parameters appName=taqa-anomaly-classifier-prod
```

### Step 4: Configure Git Deployment
```bash
# Get the deployment URL
az webapp deployment source config-local-git \
  --name taqa-anomaly-classifier-prod \
  --resource-group taqa-classifier-rg \
  --query url --output tsv
```

### Step 5: Deploy Your Code
```bash
# Initialize git if not already done
git init
git add .
git commit -m "Initial deployment to Azure"

# Add Azure remote (replace URL with output from Step 4)
git remote add azure <DEPLOYMENT_URL_FROM_STEP_4>

# Deploy to Azure
git push azure main
```

### Step 6: Verify Deployment
Visit: `https://taqa-anomaly-classifier-prod.azurewebsites.net`

---

## 🐳 Option 2: Azure Container Instances

### Step 1: Build and Push Docker Image
```bash
# Create Azure Container Registry
az acr create --resource-group taqa-classifier-rg \
  --name taqaclassifieracr --sku Basic

# Login to ACR
az acr login --name taqaclassifieracr

# Build and push image
docker build -t taqaclassifieracr.azurecr.io/taqa-classifier:latest .
docker push taqaclassifieracr.azurecr.io/taqa-classifier:latest
```

### Step 2: Deploy Container Instance
```bash
az container create \
  --resource-group taqa-classifier-rg \
  --name taqa-classifier-container \
  --image taqaclassifieracr.azurecr.io/taqa-classifier:latest \
  --cpu 1 \
  --memory 2 \
  --ip-address public \
  --ports 8000 \
  --registry-login-server taqaclassifieracr.azurecr.io \
  --registry-username taqaclassifieracr \
  --registry-password $(az acr credential show --name taqaclassifieracr --query passwords[0].value --output tsv)
```

### Step 3: Get Public IP
```bash
az container show --resource-group taqa-classifier-rg \
  --name taqa-classifier-container \
  --query ipAddress.ip --output tsv
```

---

## ⚡ Option 3: Azure Functions (Serverless)

### Step 1: Create Function App
```bash
az functionapp create \
  --resource-group taqa-classifier-rg \
  --consumption-plan-location eastus \
  --runtime python \
  --runtime-version 3.9 \
  --functions-version 4 \
  --name taqa-classifier-functions \
  --storage-account taqaclassifierstorage
```

### Step 2: Deploy Function Code
```bash
func azure functionapp publish taqa-classifier-functions
```

---

## 📁 Files Included in Deployment

### ✅ Essential Files
- `app.py` - Main Flask application
- `taqa_lookup_api.py` - Lookup-based prediction engine
- `taqa_priority_lookup.json` - Priority lookup tables
- `requirements.txt` - Python dependencies
- `templates/index.html` - Web interface
- `Dockerfile` - Container configuration
- `azure-deploy.json` - Azure ARM template

### ❌ Excluded Files (see .dockerignore)
- Development scripts (`debug_*.py`, `train_*.py`)
- Large data files (`*.xlsx`, `*.csv`)
- Model files (`*.joblib`) - not needed for lookup system
- Virtual environment (`taqa_venv/`)

---

## 🔧 Production Configuration

### Environment Variables
Set these in Azure App Service > Configuration:

| Variable | Value | Description |
|----------|-------|-------------|
| `FLASK_ENV` | `production` | Flask environment |
| `FLASK_DEBUG` | `False` | Disable debug mode |
| `PORT` | `8000` | Application port |

### SSL Certificate
- Azure provides free SSL certificates
- Custom domain setup available
- HTTPS automatically enabled

### Monitoring
Enable Application Insights:
```bash
az webapp config appsettings set \
  --name taqa-anomaly-classifier-prod \
  --resource-group taqa-classifier-rg \
  --settings APPINSIGHTS_INSTRUMENTATIONKEY=your-key
```

---

## 🎯 Performance Expectations

### Lookup System Benefits
- **Fast Startup**: No model loading (JSON lookup only)
- **Low Memory**: ~100MB RAM usage
- **High Accuracy**: 82% based on TAQA historical patterns
- **Instant Predictions**: Sub-second response times

### Scaling
- **App Service**: Auto-scale based on CPU/memory
- **Container**: Manual scaling or KEDA
- **Functions**: Automatic serverless scaling

---

## 📊 Cost Estimation

| Service Tier | Monthly Cost | CPU | RAM | Storage |
|--------------|--------------|-----|-----|---------|
| **App Service F1** | Free | Shared | 1GB | 1GB |
| **App Service B1** | ~$13 | 1 Core | 1.75GB | 10GB |
| **App Service B2** | ~$26 | 2 Cores | 3.5GB | 10GB |
| **Container Instances** | ~$30-100 | 1-4 Cores | 1-16GB | Custom |

---

## 🔍 Troubleshooting

### Common Issues

**Deployment Failed**
```bash
# Check deployment logs
az webapp log tail --name taqa-anomaly-classifier-prod --resource-group taqa-classifier-rg
```

**Module Not Found**
- Ensure `requirements.txt` includes all dependencies
- Check file paths in Docker/deployment

**Health Check Failed**
```bash
# Test health endpoint
curl https://your-app.azurewebsites.net/health
```

**Performance Issues**
- Upgrade to higher tier (B2, S1)
- Enable Application Insights for monitoring
- Check memory usage in Azure portal

### Support Commands

```bash
# Restart app
az webapp restart --name taqa-anomaly-classifier-prod --resource-group taqa-classifier-rg

# View logs
az webapp log tail --name taqa-anomaly-classifier-prod --resource-group taqa-classifier-rg

# SSH into container
az webapp ssh --name taqa-anomaly-classifier-prod --resource-group taqa-classifier-rg
```

---

## 🎉 Post-Deployment

### 1. Test the Application
- Visit your Azure URL
- Test predictions with TAQA equipment data
- Verify 82% accuracy performance

### 2. Set Up Monitoring
- Enable Application Insights
- Configure alerts for errors/performance
- Set up availability tests

### 3. Custom Domain (Optional)
```bash
az webapp config hostname add \
  --webapp-name taqa-anomaly-classifier-prod \
  --resource-group taqa-classifier-rg \
  --hostname your-domain.com
```

### 4. Backup Strategy
- Azure automatically backs up App Service
- Export lookup data regularly
- Version control your code

---

## 📞 Support

For deployment issues:
1. Check Azure portal logs
2. Verify all required files are uploaded
3. Test locally with `python app.py`
4. Contact Azure support if needed

**Your TAQA classifier is now production-ready on Azure! 🚀** 