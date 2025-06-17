#!/bin/bash

# Ollama Startup Script
# This script starts Ollama and pre-pulls the tinyllama model

echo "Starting Ollama server..."

# Start Ollama server in the background
ollama serve &
OLLAMA_PID=$!

echo "Ollama server started with PID: $OLLAMA_PID"

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready!"
        break
    fi
    echo "Waiting... (attempt $i/30)"
    sleep 2
done

# Check if Ollama is ready
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "ERROR: Ollama failed to start properly"
    exit 1
fi

# Pull the tinyllama model
echo "Pulling tinyllama model..."
if ollama pull tinyllama; then
    echo "Successfully pulled tinyllama model"
else
    echo "WARNING: Failed to pull tinyllama model, but continuing..."
fi

# List available models
echo "Available models:"
ollama list

echo "Ollama startup complete. Server is running with PID: $OLLAMA_PID"

# Wait for the Ollama process to finish
wait $OLLAMA_PID