#!/bin/bash

# Enhanced Ollama startup script with improved model readiness checks
set -e  # Exit on any error

# Debug mode (set DEBUG=1 environment variable to enable verbose logging)
DEBUG="${DEBUG:-0}"

# Ensure unbuffered output for Docker
export PYTHONUNBUFFERED=1
stdbuf -oL -eL bash

# Logging function with timestamp (Docker-friendly)
log() {
    # Use both stdout and stderr to ensure visibility
    printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1" | tee /dev/stderr
    # Force immediate flush
    sync
    
    # Debug mode: also write to a log file
    if [ "$DEBUG" = "1" ]; then
        printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >> /tmp/ollama-debug.log
    fi
}

log "Starting Ollama server..."
ollama serve &
SERVER_PID=$!
log "Ollama server started with PID: $SERVER_PID"

log "Waiting for server to initialize..."
sleep 10

# Configuration from environment variables
MODEL_NAME="${OLLAMA_MODEL:-phi3-mini-8192k:latest}"
CONTEXT_SIZE="${OLLAMA_CONTEXT_SIZE:-8192}"
MAX_RETRIES=30
RETRY_DELAY=2

log "Target model: $MODEL_NAME with ${CONTEXT_SIZE} context"

# Function to check if Ollama service is responsive
check_ollama_service() {
    local retries=0
    log "Checking if Ollama service is responsive..."
    
    while [ $retries -lt $MAX_RETRIES ]; do
        if ollama list >/dev/null 2>&1; then
            log "✅ Ollama service is responsive!"
            return 0
        else
            log "⏳ Ollama service not ready yet (attempt $((retries + 1))/$MAX_RETRIES)"
            sleep $RETRY_DELAY
            retries=$((retries + 1))
        fi
    done
    
    log "❌ ERROR: Ollama service failed to become responsive after $MAX_RETRIES attempts"
    return 1
}

# Function to check if a specific model exists in the list
model_exists() {
    local model_name="$1"
    log "Checking if model '$model_name' exists..."
    
    # Try exact match first
    if ollama list | grep -E "^${model_name}[[:space:]]" >/dev/null 2>&1; then
        log "✅ Found exact match: $model_name"
        return 0
    fi
    
    # Try with :latest tag if not already present
    if [[ "$model_name" != *":"* ]]; then
        if ollama list | grep -E "^${model_name}:latest[[:space:]]" >/dev/null 2>&1; then
            log "✅ Found model with :latest tag: ${model_name}:latest"
            return 0
        fi
    fi
    
    # Try partial match (base name without tag)
    local base_name="${model_name%:*}"
    if ollama list | grep -E "^${base_name}:" >/dev/null 2>&1; then
        log "✅ Found model with different tag: $base_name"
        return 0
    fi
    
    log "❌ Model '$model_name' not found"
    return 1
}

# Function to test model functionality using Ollama CLI
test_model_functionality() {
    local model_name="$1"
    local test_prompt="Hello"
    
    # Explicit debug output
    echo "🔍 DEBUG: Starting test_model_functionality for $model_name" >&2
    echo "🔍 DEBUG: Current working directory: $(pwd)" >&2
    echo "🔍 DEBUG: Ollama version: $(ollama --version 2>/dev/null || echo 'version unknown')" >&2
    
    log "🔍 TESTING model functionality for: $model_name" 
    sleep 1  # Small delay to ensure log appears
    log "📝 Using test prompt: '$test_prompt'"
    sleep 1
    log "⏳ Running: ollama run $model_name (timeout: 30s)"
    sleep 1
    
    # Test using ollama with non-interactive mode and timeout
    local test_output
    local start_time=$(date +%s)
    
    log "🚀 Executing ollama command now..."
    
    # Run test and capture both exit code and output
    set +e  # Temporarily disable exit on error
    test_output=$(timeout 30 ollama run "$model_name" <<< "$test_prompt" 2>&1)
    local exit_code=$?
    set -e  # Re-enable exit on error
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "⏱️  Test completed in ${duration}s with exit code: $exit_code"
    
    if [ $exit_code -eq 0 ] && [ -n "$test_output" ] && [ "$test_output" != "$test_prompt" ]; then
        log "✅ Model '$model_name' is functioning correctly!"
        # Show a brief preview of the output (first 50 chars)
        local preview=$(echo "$test_output" | tr -d '\n' | head -c 50)
        log "   📄 Response preview: $preview..."
        return 0
    elif [ $exit_code -eq 124 ]; then
        log "⏰ Model '$model_name' test timed out after 30 seconds"
        return 1
    elif [ -z "$test_output" ]; then
        log "❌ Model '$model_name' produced no output"
        return 1
    elif [ "$test_output" = "$test_prompt" ]; then
        log "⚠️  Model '$model_name' only echoed the input"
        log "   📄 Output: $test_output"
        return 1
    else
        log "❌ Model '$model_name' test failed (exit code: $exit_code)"
        if [ -n "$test_output" ]; then
            log "   📄 Error output: $(echo "$test_output" | head -c 100)..."
        fi
        return 1
    fi
}

# Function to pull model if it doesn't exist
pull_model() {
    local model_name="$1"
    log "Attempting to pull model: $model_name"
    
    if ollama pull "$model_name"; then
        log "✅ Successfully pulled model: $model_name"
        return 0
    else
        log "❌ Failed to pull model: $model_name"
        return 1
    fi
}

# Function to try alternative models if primary fails
try_alternative_models() {
    local alternatives=("tinyllama:latest" "phi3:mini" "gemma:2b" "llama2:latest")
    
    log "Primary model failed, trying alternatives..."
    
    for alt_model in "${alternatives[@]}"; do
        log "Trying alternative model: $alt_model"
        
        if model_exists "$alt_model"; then
            log "✅ Alternative model found: $alt_model"
            MODEL_NAME="$alt_model"
            return 0
        elif pull_model "$alt_model"; then
            log "✅ Alternative model pulled successfully: $alt_model"
            MODEL_NAME="$alt_model"
            return 0
        fi
    done
    
    log "❌ No alternative models could be loaded"
    return 1
}

# Function to create custom model with context size (if needed)
create_custom_model() {
    local base_model="$1"
    local custom_name="$2"
    local ctx_size="$3"
    
    # Skip custom model creation if the model already has the desired context size in name
    if [[ "$base_model" == *"$ctx_size"* ]]; then
        log "ℹ️  Model '$base_model' appears to have context size in name, skipping custom model creation"
        return 0
    fi
    
    log "Creating custom model '$custom_name' with context size $ctx_size..."
    
    # Check if custom model already exists
    if model_exists "$custom_name"; then
        log "✅ Custom model '$custom_name' already exists"
        return 0
    fi
    
    # Create Modelfile
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

    log "Creating custom model with Modelfile..."
    if ollama create "$custom_name" -f /tmp/Modelfile; then
        log "✅ Successfully created custom model: $custom_name"
        rm -f /tmp/Modelfile
        return 0
    else
        log "❌ Failed to create custom model: $custom_name"
        rm -f /tmp/Modelfile
        return 1
    fi
}

# Function to get memory usage
get_memory_usage() {
    if command -v free >/dev/null 2>&1; then
        local mem_info=$(free -h | grep "Mem:")
        log "Memory usage: $mem_info"
    elif command -v vm_stat >/dev/null 2>&1; then
        # macOS
        local pages_free=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
        local pages_total=$(vm_stat | grep "Pages wired\|Pages active\|Pages inactive\|Pages free" | awk '{sum += $3} END {print sum}')
        log "Memory: ~$((pages_free * 4096 / 1024 / 1024))MB free of ~$((pages_total * 4096 / 1024 / 1024))MB total"
    fi
}

# Main model readiness check function
check_model_readiness() {
    log "=== Starting Enhanced Model Readiness Check ==="
    
    # Step 1: Check Ollama service
    if ! check_ollama_service; then
        log "❌ FATAL: Ollama service check failed"
        return 1
    fi
    
    # Step 2: Show current available models
    log "Current available models:"
    ollama list || log "⚠️  Failed to list models"
    
    # Step 3: Show memory usage
    get_memory_usage
    
    # Step 4: Check if target model exists
    if model_exists "$MODEL_NAME"; then
        log "✅ Target model '$MODEL_NAME' found!"
        FINAL_MODEL="$MODEL_NAME"
        
    else
        log "❌ Target model '$MODEL_NAME' not found"
        
        # Step 6: Try to pull the model
        if pull_model "$MODEL_NAME"; then
            log "✅ Model '$MODEL_NAME' pulled successfully!"
            FINAL_MODEL="$MODEL_NAME"
        else
            log "❌ Failed to pull target model"
            
            # Step 7: Try alternative models
            if try_alternative_models; then
                FINAL_MODEL="$MODEL_NAME"  # MODEL_NAME updated by try_alternative_models
            else
                log "❌ FATAL: No models available"
                return 1
            fi
        fi
    fi
    
    # Step 8: Create custom model if needed (only for models without context in name)
    local safe_base_name=$(echo "$FINAL_MODEL" | sed 's/[:\\/]/-/g')
    local custom_model_name="${safe_base_name}-${CONTEXT_SIZE}k"
    
    if create_custom_model "$FINAL_MODEL" "$custom_model_name" "$CONTEXT_SIZE"; then
        if model_exists "$custom_model_name"; then
            log "✅ Using custom model: $custom_model_name"
            FINAL_MODEL="$custom_model_name"
        fi
    fi
    
    # Step 9: Test the FINAL model that will actually be used
    log "🧪 Starting model functionality test for FINAL model..."
    echo "🧪 DIRECT ECHO: About to test FINAL model functionality" >&2
    
    if test_model_functionality "$FINAL_MODEL"; then
        log "✅ Final model '$FINAL_MODEL' is ready for use!"
    else
        log "⚠️  Final model '$FINAL_MODEL' needs warm-up but will be used anyway"
    fi
    
    echo "🧪 DIRECT ECHO: Final model functionality test completed" >&2
    log "🧪 Final model functionality test completed"
    
    log "=== Model Readiness Check Complete ==="
    return 0
}

# Copy local models if available (keep existing functionality)
copy_local_models() {
    if [ -d "/models/ollama" ]; then
        log "Checking for local models to copy..."
        if [ -d "/models/ollama/blobs" ] && [ -d "/models/ollama/manifests" ]; then
            log "Found local models, copying to container..."
            mkdir -p /root/.ollama/models
            cp -r /models/ollama/* /root/.ollama/models/ 2>/dev/null || true
            log "✅ Local models copied"
        fi
    fi
}

# Main execution
main() {
    log "🚀 Starting Enhanced Ollama Model Setup"
    
    # Copy any local models first
    copy_local_models
    
    # Run model readiness check
    if check_model_readiness; then
        log "🎉 SUCCESS: Model setup completed successfully!"
        log "📋 Final model status:"
        ollama list
        log "🔧 Recommended model for your applications: $FINAL_MODEL"
        log "💡 Set DEFAULT_MODEL environment variable to: $FINAL_MODEL"
    else
        log "💥 FAILURE: Model setup failed!"
        log "📋 Available models (if any):"
        ollama list || log "No models available"
        log "🔍 Check logs above for troubleshooting information"
        return 1
    fi
    
    log "✅ Setup complete. Keeping container running..."
    wait $SERVER_PID
}

# Trap to ensure cleanup
trap 'log "Received signal, shutting down..."; kill $SERVER_PID 2>/dev/null || true; exit 0' SIGTERM SIGINT

# Run main function
main