#!/usr/bin/env python3
"""
CV Extractor Web Interface
--------------------------
This script runs the Flask web application for CV extraction.
It allows uploading and processing individual PDF files.

Usage:
    python run_web.py
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API key
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("\nERROR: Google API key is missing. Please add your API key to the .env file.")
    print("1. Create a .env file in the project root directory")
    print("2. Add your key: GOOGLE_API_KEY=your_api_key_here\n")
    sys.exit(1)

# Set Python path to include the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load the Flask app directly from the scripts directory
from scripts.app import app

# Verify if uploads directory exists, create if needed
uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'uploads')
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir, exist_ok=True)

# Verify if results directory exists, create if needed
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir, exist_ok=True)

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "development") == "development"
    
    print(f"\n--- CV Extractor Web Interface ---")
    print(f"Upload and process individual PDF files")
    print(f"Access the web interface at: http://localhost:{port}")
    
    app.run(debug=debug, host="0.0.0.0", port=port) 