#!/bin/bash

# Create necessary directories
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating directories..."
mkdir -p models/ollama
mkdir -p logs/ollama

# Check if Ollama container is running
if ! docker ps | grep -q multisource-rag-local-ollama-1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Error: Ollama container is not running. Please start it first with 'docker-compose up -d'"
    exit 1
fi

# Copy entire models directory from container to local machine
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Copying models from container..."
docker cp multisource-rag-local-ollama-1:/root/.ollama/models/. models/ollama/

# Check if models were copied successfully
if [ -d "models/ollama/blobs" ] && [ -d "models/ollama/manifests" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Successfully copied models to models/ollama/"
    echo "Model files:"
    ls -lh models/ollama/
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed to copy models. Please check if the models exist in the container."
    exit 1
fi 