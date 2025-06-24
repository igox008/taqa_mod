# 🌊 DigitalOcean Deployment Guide - TAQA Anomaly Classifier

This guide shows you how to deploy your TAQA Anomaly Priority Classifier to DigitalOcean using multiple methods.

## 📋 Prerequisites

1. **DigitalOcean Account**: Sign up at [digitalocean.com](https://digitalocean.com)
2. **GitHub Account**: For code repository
3. **Git**: For version control

## 💰 **Cost Comparison**

| Service | Monthly Cost | CPU | RAM | Best For |
|---------|-------------|-----|-----|----------|
| **App Platform Basic** | $5 | 0.5 vCPU | 512MB | Web apps |
| **Droplet Basic** | $4 | 1 vCPU | 512MB | Full control |
| **App Platform Pro** | $12 | 1 vCPU | 1GB | Production |

---

## 🚀 **Method 1: App Platform (Recommended)**

### Step 1: Push to GitHub

```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit - TAQA Classifier"

# Push to GitHub (create repo first on github.com)
git remote add origin https://github.com/yourusername/taqa-classifier.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy via DigitalOcean App Platform

1. **Go to**: [cloud.digitalocean.com/apps](https://cloud.digitalocean.com/apps)
2. **Click**: "Create App"
3. **Connect**: Your GitHub repository
4. **Select**: Repository and branch (main)
5. **Configure**:
   - **App name**: `taqa-anomaly-classifier`
   - **Environment**: Python
   - **Plan**: Basic ($5/month)
6. **Deploy**: Click "Create Resources"

### Step 3: Configure Environment Variables

In App Platform dashboard:
- **Settings** → **App-Level Environment Variables**
- Add:
  - `FLASK_ENV` = `production`
  - `FLASK_APP` = `app.py`

---

## 🐳 **Method 2: Droplet + Docker**

### Step 1: Create Droplet

1. **DigitalOcean Dashboard** → **Create** → **Droplet**
2. **Choose Image**: Ubuntu 22.04 LTS
3. **Plan**: Basic $4/month (512MB RAM)
4. **Add SSH Key** or use password
5. **Create Droplet**

### Step 2: SSH into Droplet

```bash
ssh root@your-droplet-ip
```

### Step 3: Install Docker

```bash
# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Start Docker
systemctl start docker
systemctl enable docker
```

### Step 4: Deploy with Docker

```bash
# Clone your repository
git clone https://github.com/yourusername/taqa-classifier.git
cd taqa-classifier

# Build Docker image
docker build -t taqa-classifier .

# Run container
docker run -d \
  --name taqa-app \
  -p 80:8000 \
  --restart unless-stopped \
  taqa-classifier

# Check if running
docker ps
```

### Step 5: Configure Firewall

```bash
# Allow HTTP traffic
ufw allow 80
ufw allow 22
ufw enable
```

---

## ⚡ **Method 3: Functions (Serverless)**

For API-only deployment:

### Step 1: Install doctl CLI

```bash
# Download and install doctl
wget https://github.com/digitalocean/doctl/releases/download/v1.94.0/doctl-1.94.0-linux-amd64.tar.gz
tar xf doctl-1.94.0-linux-amd64.tar.gz
sudo mv doctl /usr/local/bin
```

### Step 2: Create Function

```bash
# Initialize functions project
doctl serverless init taqa-functions
cd taqa-functions

# Create function
mkdir packages/taqa
cp app.py packages/taqa/
cp taqa_lookup_api.py packages/taqa/
cp taqa_priority_lookup.json packages/taqa/
```

---

## 📁 **Files Created for DigitalOcean**

- ✅ `runtime.txt` - Python version specification
- ✅ `Procfile` - Process commands
- ✅ `.do/app.yaml` - App Platform configuration
- ✅ `Dockerfile` - Container configuration (existing)
- ✅ `requirements.txt` - Dependencies (existing)

---

## 🔧 **Essential Deployment Files**

### For App Platform:
- `app.py`
- `taqa_lookup_api.py`
- `taqa_priority_lookup.json`
- `requirements.txt`
- `templates/index.html`
- `runtime.txt`
- `Procfile`

### For Docker Droplet:
- All above files
- `Dockerfile`

---

## 🎯 **Expected Performance**

### App Platform:
- **Startup**: ~60 seconds
- **Memory**: ~100MB usage
- **Response**: <1 second
- **Uptime**: 99.9%

### Droplet:
- **Startup**: ~30 seconds
- **Memory**: ~80MB usage
- **Response**: <500ms
- **Control**: Full server access

---

## 🔍 **Testing Your Deployment**

Once deployed, test these endpoints:

```bash
# Health check
curl https://your-app.ondigitalocean.app/health

# Model info
curl https://your-app.ondigitalocean.app/model_info

# Prediction test
curl -X POST https://your-app.ondigitalocean.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Maintenance évents ballon chaudière",
    "equipment_type": "évents ballon chaudière", 
    "section": "34MC"
  }'
```

---

## 🚀 **Advantages of DigitalOcean vs Azure**

| Feature | DigitalOcean | Azure |
|---------|-------------|-------|
| **Cost** | $5/month | $13+/month |
| **Setup** | 5 minutes | 30+ minutes |
| **Simplicity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Documentation** | Clear | Complex |
| **Git Integration** | Built-in | Requires setup |

---

## 🎉 **Quick Start Commands**

If you want to deploy RIGHT NOW:

```bash
# 1. Initialize git
git init
git add .
git commit -m "Deploy TAQA classifier"

# 2. Push to GitHub (create repo first)
git remote add origin https://github.com/yourusername/taqa-classifier.git
git push -u origin main

# 3. Go to cloud.digitalocean.com/apps
# 4. Create App → Connect GitHub → Deploy
```

**Your app will be live in ~5 minutes at:**
`https://taqa-anomaly-classifier-xxxxx.ondigitalocean.app`

---

## 🔧 **Troubleshooting**

### Build Failed
- Check `requirements.txt` format
- Ensure Python version in `runtime.txt`
- Verify `Procfile` syntax

### App Won't Start
- Check environment variables
- Verify startup command: `gunicorn app:app`
- Check logs in App Platform dashboard

### Dependencies Missing
- Ensure `requirements.txt` includes all packages
- Check Python version compatibility

---

**Ready to deploy? Which method would you prefer? 🌊** 