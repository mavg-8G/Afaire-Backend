#!/bin/bash

# Exit immediately if a command fails
set -e
# Navigate to the project directory
cd ../studio/

# Pull the latest changes from Git
echo "Pulling latest code..."
git pull

# Install latest Node.js dependencies
echo "Installing dependencies..."
npm install

# Build the project
echo "Building the project..."
npm run build

# Restart Service
echo "Restarting services..."
sudo systemctl restart ToDo.service
sudo systemctl restart ToDoBackend.service
sudo systemctl restart caddy-api.service
sudo systemctl restart caddy.service