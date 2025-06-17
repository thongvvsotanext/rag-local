#!/bin/bash

# Troubleshooting script for Ollama and LLM Orchestrator
echo "=== Ollama and LLM Orchestrator Troubleshooting ==="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

OLLAMA_URL="http://localhost:11434"
LLM_ORCHESTRATOR_URL="http://localhost:8008"

echo "1. Checking Ollama connectivity..."
if curl -s -f "$OLLAMA_URL/api/tags" > /dev/null; then
    echo -e "${GREEN}✓ Ollama is accessible${NC}"
    
    echo "2. Getting available models..."
    models=$(curl -s "$OLLAMA_URL/api/tags" | python3 -c "import sys, json; data=json.load(sys.stdin); print('\n'.join([m['name'] for m in data.get('models', [])]))")
    if [ -n "$models" ]; then
        echo -e "${GREEN}✓ Available models:${NC}"
        echo "$models"
    else
        echo -e "${YELLOW}⚠ No models found${NC}"
        echo "3. Pulling tinyllama model..."
        curl -X POST "$OLLAMA_URL/api/pull" \
             -H "Content-Type: application/json" \
             -d '{"name": "tinyllama"}' \
             --no-buffer
    fi
else
    echo -e "${RED}✗ Ollama is not accessible${NC}"
    echo "Check if Ollama container is running:"
    echo "  docker ps | grep ollama"
    echo "Check Ollama logs:"
    echo "  docker logs <container_name>"
    exit 1
fi

echo
echo "3. Testing Ollama generation..."
test_response=$(curl -s -X POST "$OLLAMA_URL/api/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "tinyllama",
        "prompt": "Hello, this is a test.",
        "stream": false
    }')

if echo "$test_response" | grep -q '"response"'; then
    echo -e "${GREEN}✓ Ollama generation test successful${NC}"
    response=$(echo "$test_response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('response', 'No response')[:100])")
    echo "Response preview: $response"
else
    echo -e "${RED}✗ Ollama generation test failed${NC}"
    echo "Response: $test_response"
fi

echo
echo "4. Checking LLM Orchestrator..."
if curl -s -f "$LLM_ORCHESTRATOR_URL/health" > /dev/null; then
    echo -e "${GREEN}✓ LLM Orchestrator is accessible${NC}"
    
    health_status=$(curl -s "$LLM_ORCHESTRATOR_URL/health" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('status', 'unknown'))")
    echo "Health status: $health_status"
    
    echo "5. Testing LLM Orchestrator generation..."
    llm_response=$(curl -s -X POST "$LLM_ORCHESTRATOR_URL/generate" \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "Hello, this is a test.",
            "parameters": {
                "temperature": 0.1,
                "max_tokens": 50
            }
        }')
    
    if echo "$llm_response" | grep -q '"response"'; then
        echo -e "${GREEN}✓ LLM Orchestrator generation test successful${NC}"
        response=$(echo "$llm_response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('response', 'No response')[:100])")
        echo "Response preview: $response"
    else
        echo -e "${RED}✗ LLM Orchestrator generation test failed${NC}"
        echo "Response: $llm_response"
    fi
else
    echo -e "${RED}✗ LLM Orchestrator is not accessible${NC}"
    echo "Check if LLM Orchestrator container is running:"
    echo "  docker ps | grep llm-orchestrator"
    echo "Check LLM Orchestrator logs:"
    echo "  docker logs <container_name>"
fi

echo
echo "6. Container status check..."
echo "Running containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(ollama|llm-orchestrator)"

echo
echo "=== Troubleshooting Complete ==="
echo
echo "If issues persist:"
echo "1. Check container logs: docker logs <container_name>"
echo "2. Restart containers: docker-compose restart ollama llm-orchestrator"
echo "3. Check resource usage: docker stats"
echo "4. Verify network connectivity: docker network ls"