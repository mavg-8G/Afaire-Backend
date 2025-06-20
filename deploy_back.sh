#!/bin/bash

# Exit immediately if a command fails
set -e

# Pull the latest changes from Git
echo "Pulling latest code..."
git pull

# Install required Python packages
echo "Installing dependencies..."
pip install -r requirements.txt

#Think to use: gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000 --workers 4