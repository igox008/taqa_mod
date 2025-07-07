# 🚀 DigitalOcean App Platform Deployment Guide

## 🌟 **Why App Platform?**

App Platform is **MUCH easier** than manual droplet setup:
- ✅ **No server management** - automatic scaling, updates, security
- ✅ **Automatic deployments** from GitHub
- ✅ **Built-in SSL/HTTPS** 
- ✅ **Starting at $5/month** (vs $18 for droplet)
- ✅ **Zero DevOps knowledge required**

---

## 📋 **Prerequisites**

1. **GitHub Account** (free)
2. **DigitalOcean Account** (free signup)
3. **Your code pushed to GitHub**

---

## 🔧 **Step 1: Prepare Your Code for App Platform**

### 1.1 Required Files ✅
Your repository already has all the required files:

```
ML/
├── .do/app.yaml                    ✅ App Platform config
├── runtime.txt                     ✅ Python version
├── requirements_api.txt            ✅ Dependencies  
├── gunicorn_appplatform.conf.py    ✅ Server config
├── wsgi.py                         ✅ WSGI entry point
├── api_server.py                   ✅ Main application
├── comprehensive_prediction_system.py ✅
├── *.pkl models                    ✅ ML models (24MB total)
└── *.csv data files               ✅ Training data
```

### 1.2 Push to GitHub
```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit - Equipment Prediction API"

# Create GitHub repo and push
# Go to github.com → New repository → "ML" or "equipment-prediction-api"
git remote add origin https://github.com/YOURUSERNAME/YOURREPO.git
git branch -M main
git push -u origin main
```

---

## 🚀 **Step 2: Deploy to App Platform**

### 2.1 Create App
1. Go to https://cloud.digitalocean.com/apps
2. Click **"Create App"**
3. Choose **"GitHub"** as source
4. **Connect your GitHub account** (one-time setup)
5. Select your repository (e.g., `yourusername/ML`)
6. Choose branch: **`main`**
7. Click **"Next"**

### 2.2 Configure App
App Platform will auto-detect your app using `.do/app.yaml`:

**Detected Configuration:**
- **Type:** Web Service
- **Name:** equipment-prediction-api
- **Run Command:** `gunicorn -c gunicorn_appplatform.conf.py wsgi:app`
- **Port:** 5000
- **Instance Size:** Basic XXS ($5/month)

**⚠️ Important Settings:**
- **Auto-deploy:** ✅ Enabled (deploys automatically on git push)
- **Source Directory:** `/` (root)
- **Environment:** Production

### 2.3 Review Resources
```
💰 Cost Breakdown:
├── Basic XXS: $5/month  (512MB RAM, 1 vCPU)
├── Basic XS:  $12/month (1GB RAM, 1 vCPU) ← Recommended
└── Basic S:   $25/month (2GB RAM, 2 vCPU)
```

**Recommendation:** Start with **Basic XS ($12/month)** for your ML models.

### 2.4 Deploy!
1. Click **"Create Resources"** 
2. Wait 5-10 minutes for initial deployment
3. Your app will be available at: `https://your-app-name.ondigitalocean.app`

---

## 🧪 **Step 3: Test Your Deployed API**

### 3.1 Get Your App URL
After deployment, you'll get a URL like:
```
https://equipment-prediction-api-xxxxx.ondigitalocean.app
```

### 3.2 Test Health Endpoint
```bash
curl https://your-app-url.ondigitalocean.app/health
```

### 3.3 Test Prediction
```bash
curl -X POST https://your-app-url.ondigitalocean.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "anomaly_id": "TEST-001",
    "description": "Fuite importante d'\''huile au niveau du palier",
    "equipment_name": "POMPE FUEL PRINCIPALE N°1", 
    "equipment_id": "98b82203-7170-45bf-879e-f47ba6e12c86"
  }'
```

### 3.4 Update Test Script
Update `test_deployed_api_digitalocean.py`:
```python
# Replace this line:
SERVER_IP = "YOUR_DROPLET_IP_HERE"  

# With your App Platform URL:
API_BASE_URL = "https://your-app-name.ondigitalocean.app"
```

---

## 📊 **Step 4: Monitor Your App**

### 4.1 App Platform Dashboard
- **Deployments:** View deployment history
- **Runtime Logs:** Real-time application logs
- **Metrics:** CPU, Memory, Request metrics
- **Settings:** Scale up/down, environment variables

### 4.2 View Logs
```bash
# From App Platform dashboard → Your App → Runtime Logs
# You'll see your model loading and API requests in real-time
```

### 4.3 Scaling
If you need more power:
1. Go to your app settings
2. Change instance size to **Basic XS** or **Basic S**
3. Click **"Save"** → Auto-redeploys with more resources

---

## 🔧 **Step 5: Continuous Deployment**

### 5.1 Automatic Deployments
Every time you push to GitHub, App Platform automatically:
1. Pulls new code
2. Installs dependencies
3. Runs tests (if configured)
4. Deploys new version
5. Health checks the deployment

### 5.2 Making Updates
```bash
# Make changes to your code
git add .
git commit -m "Updated batch processing"
git push

# App Platform automatically deploys in ~5 minutes
```

---

## 🌐 **Step 6: Custom Domain (Optional)**

### 6.1 Add Your Domain
1. In App Platform → Your App → Settings → Domains
2. Add your domain (e.g., `api.yourcompany.com`)
3. Update DNS records as instructed
4. **Free SSL certificate** automatically provided

---

## 💡 **App Platform vs Droplet Comparison**

| Feature | App Platform | Manual Droplet |
|---------|-------------|----------------|
| **Setup Time** | 10 minutes | 2+ hours |
| **Cost** | $5-12/month | $18/month |
| **Maintenance** | Zero | High |
| **SSL/HTTPS** | Automatic | Manual setup |
| **Scaling** | One-click | Manual |
| **Deployments** | Git push | Manual upload |
| **Monitoring** | Built-in | Setup required |
| **Backups** | Automatic | Manual |

---

## 🎯 **Production Recommendations**

### 6.1 Instance Size for Production
```bash
Basic XS ($12/month):
├── 1GB RAM ← Good for your ML models
├── 1 vCPU  
└── 25GB bandwidth

Basic S ($25/month):
├── 2GB RAM ← Best for high traffic
├── 2 vCPU
└── 50GB bandwidth
```

### 6.2 Environment Variables
Add these in App Platform → Settings → Environment:
```
FLASK_ENV=production
PYTHONPATH=/app
MAX_WORKERS=2
TIMEOUT=300
```

### 6.3 Monitoring
- **Built-in metrics** show requests, response times, errors
- **Alerting** can notify you of issues
- **Log aggregation** for debugging

---

## 🔍 **Troubleshooting**

### Common Issues:

**1. Build Fails - Large Files**
```
Error: File too large for build
Solution: Models (.pkl files) under 100MB are OK.
Your files: availability_model.pkl (7.4MB) ✅
```

**2. Memory Issues**
```
Error: Worker killed (out of memory)
Solution: Upgrade to Basic XS ($12/month) for 1GB RAM
```

**3. Slow Model Loading**
```
Error: Health check timeout
Solution: Models load in ~10-20 seconds, which is normal
```

**4. Import Errors**
```
Error: Module not found
Solution: Check requirements_api.txt has all dependencies
```

---

## 🎉 **Success! Your API is Live**

Your Equipment Prediction API is now:
- ✅ **Live on the internet** with HTTPS
- ✅ **Automatically deploying** from GitHub
- ✅ **Handling thousands of requests** 
- ✅ **Supporting batch processing** (up to 6000 anomalies)
- ✅ **Monitored and backed up** automatically

## 📡 **Your Live API Endpoints:**
```
GET  https://your-app.ondigitalocean.app/health
GET  https://your-app.ondigitalocean.app/models/info  
POST https://your-app.ondigitalocean.app/predict
```

**Total setup time:** ~15 minutes
**Monthly cost:** $5-12 (much less than $18 droplet)
**Maintenance:** Zero - App Platform handles everything! 