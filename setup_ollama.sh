#!/bin/bash

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 1
done

# Pull recommended models for M2 Mac
echo "Pulling Llama2 7B model..."
ollama pull llama2:7b

echo "Pulling Mistral 7B model..."
ollama pull mistral:7b

echo "Pulling CodeLlama 7B model..."
ollama pull codellama:7b

# Show memory usage
echo "Current Ollama models:"
ollama list

echo "Setup complete! Models are ready to use." 