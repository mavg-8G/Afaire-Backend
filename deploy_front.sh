#!/bin/bash

# Exit immediately if a command fails
set -e

# Pull the latest changes from Git
echo "Pulling latest code..."
git pull

# Install latest Node.js dependencies
echo "Installing dependencies..."

#Think to use: gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000 --workers 4