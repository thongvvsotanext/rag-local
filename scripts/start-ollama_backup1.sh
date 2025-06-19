#!/bin/bash

LOG_FILE="/var/log/ollama/startup.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Ollama server..." | tee -a "$LOG_FILE"
ollama serve &
SERVER_PID=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ollama server started with PID: $SERVER_PID" | tee -a "$LOG_FILE"

# Check server process
if ! ps -p $SERVER_PID > /dev/null; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Error: Ollama server failed to start" | tee -a "$LOG_FILE"
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for server to initialize..." | tee -a "$LOG_FILE"
sleep 10

# Model configuration
MODEL_NAME="${OLLAMA_MODEL:-phi3:mini}"
CONTEXT_SIZE="${OLLAMA_CONTEXT_SIZE:-8192}"
SAFE_BASE_NAME=$(echo "$MODEL_NAME" | sed 's/[:\\/]/-/g')
CUSTOM_MODEL_NAME="${SAFE_BASE_NAME}-${CONTEXT_SIZE}k:latest"  # Explicit :latest tag

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Target model: $MODEL_NAME with ${CONTEXT_SIZE} context" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Custom model name: $CUSTOM_MODEL_NAME" | tee -a "$LOG_FILE"

# Function to pull model with retries
pull_model() {
  local model="$1"
  local retries=0
  local max_retries=3
  local delay=5

  while [ $retries -lt $max_retries ]; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pulling model $model (attempt $((retries + 1))/$max_retries)..." | tee -a "$LOG_FILE"
    if ollama pull "$model"; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Successfully pulled $model" | tee -a "$LOG_FILE"
      return 0
    fi
    retries=$((retries + 1))
    sleep $delay
    delay=$((delay * 2))  # Exponential backoff
  done
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed to pull $model after $max_retries attempts" | tee -a "$LOG_FILE"
  return 1
}

# Function to clean duplicate model names
clean_duplicate_models() {
  local model_pattern="$1"
  local canonical_name="$2"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking for duplicate model names matching $model_pattern..." | tee -a "$LOG_FILE"
  ollama list | grep "$model_pattern" | grep -v "^${canonical_name}[[:space:]]" | awk '{print $1}' | while read -r listed_name; do
    if [ -n "$listed_name" ] && [ "$listed_name" != "$canonical_name" ]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Removing duplicate model name: $listed_name" | tee -a "$LOG_FILE"
      ollama rm "$listed_name" || echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed to remove $listed_name" | tee -a "$LOG_FILE"
    fi
  done
}

# Function to clean unused models (optional)
clean_unused_models() {
  local canonical_name="$1"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaning unused models..." | tee -a "$LOG_FILE"
  ollama list | awk '{print $1}' | grep -vE "^${canonical_name}[[:space:]]|^${MODEL_NAME}[[:space:]]|^${MODEL_NAME}:|^${MODEL_NAME%:*}:latest" | while read -r model; do
    if [ -n "$model" ]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Removing unused model: $model" | tee -a "$LOG_FILE"
      ollama rm "$model" || echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed to remove $model" | tee -a "$LOG_FILE"
    fi
  done
}

# Function to create custom model
create_custom_model() {
  local base_model="$1"
  local custom_name="$2"
  local ctx_size="$3"
  
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating custom model $custom_name..." | tee -a "$LOG_FILE"
  
  # Check if custom model exists
  if ollama list | grep -E "^${custom_name}[[:space:]]|^${custom_name%:*}:"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Custom model $custom_name already exists" | tee -a "$LOG_FILE"
    # Clean duplicates
    clean_duplicate_models "phi3.*8192k" "$custom_name"
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

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Modelfile created:" | tee -a "$LOG_FILE"
  cat /tmp/Modelfile | tee -a "$LOG_FILE"
  
  # Create the custom model
  if ollama create "${custom_name%:*}" -f /tmp/Modelfile; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Successfully created custom model ${custom_name%:*}" | tee -a "$LOG_FILE"
    
    # Explicitly tag the model
    ollama tag "${custom_name%:*}" "$custom_name"
    
    # Clean up potential duplicates
    clean_duplicate_models "phi3.*8192k" "$custom_name"
    
    # Log registry state
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model registry state:" | tee -a "$LOG_FILE"
    ls -R /root/.ollama/models/manifests | tee -a "$LOG_FILE"
    
    # Test model readiness with native command
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Testing custom model $custom_name..." | tee -a "$LOG_FILE"
    if echo "test" | timeout 15 ollama run "$custom_name" >/dev/null 2>&1; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Custom model $custom_name is ready!" | tee -a "$LOG_FILE"
    else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Custom model $custom_name created but may need warm-up" | tee -a "$LOG_FILE"
    fi
    
    rm -f /tmp/Modelfile
    return 0
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed to create custom model ${custom_name%:*}" | tee -a "$LOG_FILE"
    rm -f /tmp/Modelfile
    return 1
  fi
}

# Function to check local model (simplified, assuming volume mount)
check_local_model_exists() {
  local model_name="$1"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking for local model $model_name..." | tee -a "$LOG_FILE"
  if ollama list | grep -E "^${model_name}[[:space:]]|^${model_name}:|^${model_name%:*}:latest"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model $model_name found locally" | tee -a "$LOG_FILE"
    return 0
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Model $model_name not found locally" | tee -a "$LOG_FILE"
  return 1
}

# Check model readiness
check_model_readiness() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking Ollama service readiness..." | tee -a "$LOG_FILE"
  local retries=0
  local max_retries=30
  local server_ready=false

  while [ $retries -lt $max_retries ]; do
    if ollama list >/dev/null 2>&1; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ollama server is responsive!" | tee -a "$LOG_FILE"
      server_ready=true
      break
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for server (attempt $((retries + 1))/$max_retries)..." | tee -a "$LOG_FILE"
    sleep 2
    retries=$((retries + 1))
  done

  if [ "$server_ready" != "true" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Error: Ollama server not ready after $max_retries attempts" | tee -a "$LOG_FILE"
    exit 1
  fi

  # Check base model
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking base model: $MODEL_NAME" | tee -a "$LOG_FILE"
  if ! check_local_model_exists "$MODEL_NAME"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Base model $MODEL_NAME not found, pulling..." | tee -a "$LOG_FILE"
    if ! pull_model "$MODEL_NAME"; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Trying fallback model: gemma:2b" | tee -a "$LOG_FILE"
      if pull_model "gemma:2b"; then
        MODEL_NAME="gemma:2b"
        SAFE_BASE_NAME=$(echo "$MODEL_NAME" | sed 's/[:\\/]/-/g')
        CUSTOM_MODEL_NAME="${SAFE_BASE_NAME}-${CONTEXT_SIZE}k:latest"
      else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Error: Failed to pull fallback model" | tee -a "$LOG_FILE"
        exit 1
      fi
    fi
  fi

  # Create custom model
  create_custom_model "$MODEL_NAME" "$CUSTOM_MODEL_NAME" "$CONTEXT_SIZE"
  
  # Write model name for dynamic use (e.g., healthcheck)
  echo "$CUSTOM_MODEL_NAME" > /tmp/model_name
}

# Run readiness check
check_model_readiness

# Clean unused models (optional, uncomment to enable)
clean_unused_models "$CUSTOM_MODEL_NAME"

# Final status
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Final available models:" | tee -a "$LOG_FILE"
ollama list | tee -a "$LOG_FILE"
if ollama list | grep -q "^${CUSTOM_MODEL_NAME}[[:space:]]"; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Model $CUSTOM_MODEL_NAME is ready for use" | tee -a "$LOG_FILE"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ Model $CUSTOM_MODEL_NAME not found, check logs" | tee -a "$LOG_FILE"
  exit 1
fi
