#!/bin/bash

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Ollama server..."
ollama serve &
SERVER_PID=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ollama server started with PID: $SERVER_PID"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for server to initialize..."
sleep 10

# Use smaller model that fits in available memory
MODEL_NAME="${OLLAMA_MODEL:-phi3:mini}"
CONTEXT_SIZE="${OLLAMA_CONTEXT_SIZE:-8192}"

# ✅ FIXED: Proper model name conversion
# Replace : and / with - to create safe model names
SAFE_BASE_NAME=$(echo "$MODEL_NAME" | sed 's/[:\\/]/-/g')
CUSTOM_MODEL_NAME="${SAFE_BASE_NAME}-${CONTEXT_SIZE}k"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Target model: $MODEL_NAME with ${CONTEXT_SIZE} context"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Safe base name: $SAFE_BASE_NAME"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Custom model name: $CUSTOM_MODEL_NAME"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking available models..."
ollama list

# Function to check if specific model exists locally
check_local_model_exists() {
    local model_name="$1"
    local base_model="${model_name%:*}"  # Remove tag if present
    local model_tag="${model_name#*:}"   # Extract tag, default to 'latest'
    
    # If no tag specified, default to latest
    if [ "$model_tag" = "$model_name" ]; then
        model_tag="latest"
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Looking for local model: $base_model:$model_tag"
    
    # Check if model directories exist
    if [ ! -d "/models/ollama/blobs" ] || [ ! -d "/models/ollama/manifests" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Local model directories not found"
        return 1
    fi
    
    # Check for model manifest (Ollama stores models as manifests + blobs)
    local manifest_paths=(
        "/models/ollama/manifests/registry.ollama.ai/library/$base_model/$model_tag"
        "/models/ollama/manifests/$base_model/$model_tag"
        "/models/ollama/manifests/$model_name"
    )
    
    for manifest_path in "${manifest_paths[@]}"; do
        if [ -f "$manifest_path" ] || [ -d "$manifest_path" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found local model manifest at: $manifest_path"
            return 0
        fi
    done
    
    # Alternative check: look for any manifest containing the model name
    if find "/models/ollama/manifests" -name "*$base_model*" 2>/dev/null | grep -q .; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found model $base_model in local manifests"
        return 0
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model $model_name not found locally"
    return 1
}

# Function to create custom model with desired context size
create_custom_model() {
    local base_model="$1"
    local custom_name="$2"
    local ctx_size="$3"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating custom model $custom_name with context size $ctx_size..."
    
    # Check if custom model already exists (check multiple possible names)
    if ollama list | grep -E "^${custom_name}[[:space:]]|^${custom_name}:" | head -1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Custom model $custom_name already exists"
        return 0
    fi
    
    # Create temporary Modelfile with enhanced parameters
    cat > /tmp/Modelfile << EOF
FROM $base_model
PARAMETER num_ctx $ctx_size
PARAMETER num_predict 4096
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER repeat_last_n 64
EOF

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Modelfile created with context size $ctx_size"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Modelfile contents:"
    cat /tmp/Modelfile
    
    # Create the custom model
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Executing: ollama create \"$custom_name\" -f /tmp/Modelfile"
    if ollama create "$custom_name" -f /tmp/Modelfile; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Successfully created custom model $custom_name"
        
        # Wait a moment for the model to be ready
        sleep 3
        
        # Test the custom model
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Testing custom model..."
        if echo "Hello" | timeout 30 ollama run "$custom_name" >/dev/null 2>&1; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Custom model $custom_name is working correctly!"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Custom model $custom_name created but may need warm-up"
        fi
        
        # Clean up temporary file
        rm -f /tmp/Modelfile
        return 0
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed to create custom model $custom_name"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking if model was created anyway..."
        ollama list
        rm -f /tmp/Modelfile
        return 1
    fi
}

# Check if the specific model exists locally
if check_local_model_exists "$MODEL_NAME"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found local model $MODEL_NAME, copying to container..."
    # Create models directory if it doesn't exist
    mkdir -p /root/.ollama/models
    
    # Copy all model data (manifests and blobs are interconnected)
    cp -r /models/ollama/* /root/.ollama/models/
    
    if [ $? -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Local model $MODEL_NAME copied successfully"
        LOCAL_MODEL_COPIED=true
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed to copy local model $MODEL_NAME"
        LOCAL_MODEL_COPIED=false
    fi
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model $MODEL_NAME not found locally, will pull from registry"
    LOCAL_MODEL_COPIED=false
fi

# Function to check model readiness using only ollama CLI
check_model_readiness() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking if Ollama service is ready..."
    
    # Wait for ollama list command to work reliably
    local retries=0
    local max_retries=30
    
    while [ $retries -lt $max_retries ]; do
        if ollama list >/dev/null 2>&1; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ollama service is ready!"
            break
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ollama service is not ready yet, waiting... (attempt $((retries + 1))/$max_retries)"
            sleep 2
            retries=$((retries + 1))
        fi
    done
    
    if [ $retries -eq $max_retries ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: Ollama service readiness check timed out"
        return 1
    fi

    # List current models to see what's available
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Current available models:"
    ollama list

    # Check if base model exists and is ready
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking for base model: $MODEL_NAME"
    
    # Check for exact match or with :latest tag
    if ollama list | grep -E "^$MODEL_NAME[[:space:]]|^${MODEL_NAME}:|^${MODEL_NAME%:*}:latest" | head -1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Base model $MODEL_NAME is available"
        
        # Test model with a simple generation to ensure it's ready
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Testing base model readiness..."
        if echo "Hello" | timeout 30 ollama run "$MODEL_NAME" >/dev/null 2>&1; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Base model $MODEL_NAME is ready for inference!"
            BASE_MODEL_READY=true
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Base model $MODEL_NAME loaded but may need warm-up"
            BASE_MODEL_READY=true
        fi
    else
        if [ "$LOCAL_MODEL_COPIED" = "true" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Local model was copied but not showing in list. Waiting for indexing..."
            sleep 5
            ollama list
        fi
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Base model $MODEL_NAME not found, attempting to pull..."
        ollama pull "$MODEL_NAME"
        
        if [ $? -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Successfully pulled $MODEL_NAME"
            BASE_MODEL_READY=true
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed to pull $MODEL_NAME"
            # Try alternative smaller models if primary fails
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Trying alternative small model: gemma:2b"
            if ollama pull gemma:2b; then
                MODEL_NAME="gemma:2b"
                SAFE_BASE_NAME=$(echo "$MODEL_NAME" | sed 's/[:\\/]/-/g')
                CUSTOM_MODEL_NAME="${SAFE_BASE_NAME}-${CONTEXT_SIZE}k"
                BASE_MODEL_READY=true
            else
                BASE_MODEL_READY=false
            fi
        fi
    fi
    
    # Create custom model with desired context size if base model is ready
    if [ "$BASE_MODEL_READY" = "true" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating custom model with $CONTEXT_SIZE context size..."
        create_custom_model "$MODEL_NAME" "$CUSTOM_MODEL_NAME" "$CONTEXT_SIZE"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skipping custom model creation - base model not ready"
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model setup complete!"
}

# Run model readiness check synchronously (not in background)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running model setup..."
check_model_readiness

# Show final model status after setup is complete
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Final available models:"
ollama list

# ✅ FIXED: Better model detection for final message
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking which model to recommend..."
if ollama list | grep -E "^${CUSTOM_MODEL_NAME}[[:space:]]|^${CUSTOM_MODEL_NAME}:" | head -1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Use model: $CUSTOM_MODEL_NAME (with ${CONTEXT_SIZE} context)"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔧 Update your LLM orchestrator DEFAULT_MODEL to: $CUSTOM_MODEL_NAME"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  Using base model: $MODEL_NAME (default context)"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔧 Update your LLM orchestrator DEFAULT_MODEL to: $MODEL_NAME"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Setup complete. Keeping container running..."
wait $SERVER_PID