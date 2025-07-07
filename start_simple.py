#!/usr/bin/env python3
"""
Simple startup script for App Platform deployment
Alternative to gunicorn if there are issues
"""

import os
from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False) 