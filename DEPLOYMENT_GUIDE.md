# 🚀 DigitalOcean Deployment Guide - Equipment Prediction API

## 📋 Prerequisites

- DigitalOcean account
- Credit card or PayPal for billing
- SSH key pair (recommended)

## 🖥️ Step 1: Create DigitalOcean Droplet

### 1.1 Create New Droplet
1. Log into DigitalOcean console: https://cloud.digitalocean.com/
2. Click **"Create"** → **"Droplets"**
3. Choose configuration:
   - **OS**: Ubuntu 22.04 LTS (recommended)
   - **Plan**: Basic
   - **CPU**: Regular Intel (2 GB RAM, 1 vCPU, 50 GB SSD) - **$18/month**
   - **Datacenter**: Choose closest to your users
   - **Authentication**: SSH Key (recommended) or Password
   - **Hostname**: `equipment-api-server`

### 1.2 Configure Firewall (Optional but Recommended)
1. In droplet creation, click **"Advanced Options"**
2. Create firewall with these rules:
   - **SSH (22)**: Your IP only
   - **HTTP (80)**: All IPv4, All IPv6
   - **HTTPS (443)**: All IPv4, All IPv6
   - **Custom (5000)**: All IPv4, All IPv6 (for API)

## 🔧 Step 2: Initial Server Setup

### 2.1 Connect to Server
```bash
# Replace YOUR_SERVER_IP with your droplet IP
ssh root@YOUR_SERVER_IP
```

### 2.2 Update System
```bash
apt update && apt upgrade -y
```

### 2.3 Install Required Software
```bash
# Install Python, pip, and git
apt install -y python3 python3-pip python3-venv git nginx

# Install build tools for Python packages
apt install -y build-essential python3-dev
```

## 📦 Step 3: Deploy Your Code

### 3.1 Upload Your Project Files
**Option A: Using SCP (Recommended)**
```bash
# From your local machine (ml3 directory)
scp -r . root@YOUR_SERVER_IP:/root/ml3/
```

**Option B: Using Git**
```bash
# On server: create and clone repository
cd /root
git clone https://github.com/yourusername/ml3.git
cd ml3
```

### 3.2 Required Files Checklist
Ensure these files are on the server:
```
/root/ml3/
├── api_server.py                     ✓
├── wsgi.py                          ✓
├── gunicorn.conf.py                 ✓
├── requirements_api.txt             ✓
├── start_api.sh                     ✓
├── equipment-api.service            ✓
├── comprehensive_prediction_system.py ✓
├── ml_feature_engine.py             ✓
├── ml_fiability_engine.py           ✓
├── ml_process_safety_engine.py      ✓
├── availability_model.pkl           ✓ (7.4MB)
├── fiability_model.pkl              ✓ (5.5MB)
├── process_safety_model.pkl         ✓ (11MB)
├── equipment_simple.csv             ✓
├── severe_words_simple.csv          ✓
├── equipment_fiability_simple.csv   ✓
├── severe_words_fiability_simple.csv ✓
├── equipment_process_safety_simple.csv ✓
└── severe_words_process_safety_simple.csv ✓
```

## 🏃 Step 4: Setup and Test API

### 4.1 Make Scripts Executable
```bash
cd /root/ml3
chmod +x start_api.sh
```

### 4.2 Test Manual Startup
```bash
# Test the API startup script
./start_api.sh
```

You should see:
```
🚀 Starting Equipment Prediction API...
✅ All model files found
📦 Creating virtual environment...
🔧 Activating virtual environment...
📚 Installing dependencies...
🌐 Starting API server with gunicorn...
```

### 4.3 Test API Endpoints
Open a new terminal and test:
```bash
# Health check
curl http://YOUR_SERVER_IP:5000/health

# Expected response:
{
  "status": "healthy",
  "models_loaded": true,
  "message": "Equipment Prediction API is running"
}
```

### 4.4 Test Prediction
```bash
curl -X POST http://YOUR_SERVER_IP:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "anomaly_id": "TEST-001",
    "description": "Fuite importante d'\''huile au niveau du palier",
    "equipment_name": "POMPE FUEL PRINCIPALE N°1", 
    "equipment_id": "98b82203-7170-45bf-879e-f47ba6e12c86"
  }'
```

Stop the test server: **Ctrl+C**

## ⚙️ Step 5: Setup as System Service

### 5.1 Install Service File
```bash
# Copy service file to systemd
cp /root/ml3/equipment-api.service /etc/systemd/system/

# Reload systemd
systemctl daemon-reload

# Enable service to start on boot
systemctl enable equipment-api

# Start the service
systemctl start equipment-api
```

### 5.2 Check Service Status
```bash
# Check if service is running
systemctl status equipment-api

# View logs
journalctl -u equipment-api -f
```

## 🌐 Step 6: Setup Nginx Reverse Proxy (Optional)

### 6.1 Create Nginx Configuration
```bash
cat > /etc/nginx/sites-available/equipment-api << 'EOF'
server {
    listen 80;
    server_name YOUR_SERVER_IP;  # Replace with your domain or IP

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }
}
EOF
```

### 6.2 Enable Nginx Site
```bash
# Enable the site
ln -s /etc/nginx/sites-available/equipment-api /etc/nginx/sites-enabled/

# Remove default site
rm /etc/nginx/sites-enabled/default

# Test nginx configuration
nginx -t

# Restart nginx
systemctl restart nginx
systemctl enable nginx
```

## ✅ Step 7: Final Testing

### 7.1 Test API Through Nginx
```bash
# Test through nginx (port 80)
curl http://YOUR_SERVER_IP/health

# Test prediction through nginx
curl -X POST http://YOUR_SERVER_IP/predict \
  -H "Content-Type: application/json" \
  -d '{
    "anomaly_id": "PROD-001",
    "description": "Vibration anormale détectée sur moteur principal",
    "equipment_name": "MOTEUR POMPE PRINCIPALE",
    "equipment_id": "b16f90d0-7119-42fe-bea3-0b30e2a3b0a8"
  }'
```

## 📡 API Usage

### Endpoints
- **GET** `http://YOUR_SERVER_IP/health` - Health check
- **GET** `http://YOUR_SERVER_IP/models/info` - Model information
- **POST** `http://YOUR_SERVER_IP/predict` - Make predictions

### Example API Call
```bash
curl -X POST http://YOUR_SERVER_IP/predict \
  -H "Content-Type: application/json" \
  -d '{
    "anomaly_id": "ANO-2024-001",
    "description": "Fuite importante vapeur avec alarme sécurité",
    "equipment_name": "CHAUDIERE UNITE 3",
    "equipment_id": "f4653d21-1b81-4ea3-b483-42b746e06051"
  }'
```

### Response Format
```json
{
  "anomaly_id": "ANO-2024-001",
  "equipment_id": "f4653d21-1b81-4ea3-b483-42b746e06051",
  "equipment_name": "CHAUDIERE UNITE 3",
  "predictions": {
    "availability": {
      "score": 2.1,
      "description": "Equipment uptime and operational readiness"
    },
    "reliability": {
      "score": 1.8,
      "description": "Equipment integrity and dependability"
    },
    "process_safety": {
      "score": 1.5,
      "description": "Safety risk assessment and hazard identification"
    }
  },
  "overall_score": 1.8,
  "total_sum": 5.4,
  "risk_assessment": {
    "overall_risk_level": "HIGH",
    "priority_level": "URGENT",
    "requires_immediate_attention": true
  },
  "status": "success"
}
```

## 🔧 Management Commands

### Service Management
```bash
# Start service
systemctl start equipment-api

# Stop service
systemctl stop equipment-api

# Restart service
systemctl restart equipment-api

# Check status
systemctl status equipment-api

# View logs
journalctl -u equipment-api -f
```

### Manual Management
```bash
# Start manually
cd /root/ml3 && ./start_api.sh

# Kill all gunicorn processes
pkill -f gunicorn
```

## 💰 Cost Estimation

- **Droplet**: $18/month (2GB RAM, 1 vCPU, 50GB SSD)
- **Bandwidth**: 2TB included (additional $0.01/GB)
- **Total**: ~$18-25/month depending on usage

## 🔒 Security Best Practices

1. **Change SSH port** from 22 to custom port
2. **Disable root login** and create sudo user
3. **Enable UFW firewall**
4. **Install fail2ban** for intrusion protection
5. **Regular system updates**
6. **SSL certificate** for HTTPS (Let's Encrypt)

## 📊 Monitoring & Logs

### View API Logs
```bash
# Service logs
journalctl -u equipment-api -f

# Nginx access logs
tail -f /var/log/nginx/access.log

# Nginx error logs
tail -f /var/log/nginx/error.log
```

### Monitor Performance
```bash
# CPU and memory usage
htop

# API process details
ps aux | grep gunicorn

# Network connections
netstat -tulpn | grep :5000
```

## 🚨 Troubleshooting

### Common Issues

**1. Models not loading**
```bash
# Check file permissions
ls -la /root/ml3/*.pkl

# Check Python path
cd /root/ml3 && python3 -c "import comprehensive_prediction_system"
```

**2. Service won't start**
```bash
# Check service logs
journalctl -u equipment-api --no-pager

# Test gunicorn directly
cd /root/ml3
source api_env/bin/activate
gunicorn -c gunicorn.conf.py wsgi:app
```

**3. Memory issues**
```bash
# Check available memory
free -h

# Monitor process memory
ps aux --sort=-%mem | head
```

**4. Connection refused**
```bash
# Check if service is running
systemctl status equipment-api

# Check firewall
ufw status

# Check nginx
systemctl status nginx
```

## 🎯 Production Recommendations

1. **Setup SSL/HTTPS** with Let's Encrypt
2. **Domain name** instead of IP address
3. **Load balancer** for high availability
4. **Database logging** for API requests
5. **Monitoring** with tools like Grafana
6. **Backup strategy** for model files
7. **Auto-scaling** based on demand

Your API is now production-ready on DigitalOcean! 🎉 