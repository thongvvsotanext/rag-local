import logging
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import aiohttp
import json
import os
from dotenv import load_dotenv
import asyncio
import time
from contextlib import asynccontextmanager
import tiktoken
import traceback
import uuid

# Load environment variables
load_dotenv()

# Configure enhanced logging with structured format
class StructuredFormatter(logging.Formatter):
    def format(self, record):
        # Add request_id if available
        if hasattr(record, 'request_id'):
            record.request_id = getattr(record, 'request_id', 'N/A')
        else:
            record.request_id = 'N/A'
        
        # Add endpoint if available
        if hasattr(record, 'endpoint'):
            record.endpoint = getattr(record, 'endpoint', 'N/A')
        else:
            record.endpoint = 'N/A'
            
        return super().format(record)

# Configure logging
formatter = StructuredFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - [%(endpoint)s] - %(message)s'
)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

# File handler
file_handler = logging.FileHandler('llm_orchestrator.log')
file_handler.setFormatter(formatter)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)

# Helper function to create logger with context
def get_contextual_logger(request_id: str, endpoint: str):
    """Create a logger with request context."""
    contextual_logger = logging.getLogger(__name__)
    
    # Create a custom LoggerAdapter to inject context
    class ContextAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            # Properly merge extra parameters
            if 'extra' in kwargs:
                # Merge the context with any additional extra parameters
                merged_extra = {
                    'request_id': self.extra.get('request_id', 'N/A'),
                    'endpoint': self.extra.get('endpoint', 'N/A'),
                    **kwargs['extra']
                }
                kwargs['extra'] = merged_extra
            else:
                kwargs['extra'] = {
                    'request_id': self.extra.get('request_id', 'N/A'),
                    'endpoint': self.extra.get('endpoint', 'N/A')
                }
            return msg, kwargs
    
    return ContextAdapter(contextual_logger, {'request_id': request_id, 'endpoint': endpoint})

# Alternative simple logging helper for error cases
def log_with_context(logger_instance, level, message, request_id, endpoint, **extra_data):
    """Simple logging helper that ensures context is properly logged."""
    log_data = {
        'request_id': request_id,
        'endpoint': endpoint,
        **extra_data
    }
    
    # Format the message with context
    formatted_message = f"{message}"
    if extra_data:
        formatted_message += f" | Context: {json.dumps(log_data, default=str, indent=2)}"
    
    # Use the appropriate log level
    if level == 'error':
        logger_instance.error(formatted_message, extra=log_data)
    elif level == 'info':
        logger_instance.info(formatted_message, extra=log_data)
    elif level == 'warning':
        logger_instance.warning(formatted_message, extra=log_data)
    elif level == 'debug':
        logger_instance.debug(formatted_message, extra=log_data)
    else:
        logger_instance.info(formatted_message, extra=log_data)

# Logging configurations
LOG_PROMPT_MAX_LENGTH = int(os.getenv("LOG_PROMPT_MAX_LENGTH", "200"))  # Max prompt length to log
LOG_RESPONSE_MAX_LENGTH = int(os.getenv("LOG_RESPONSE_MAX_LENGTH", "200"))  # Max response length to log
LOG_FULL_CONTENT = os.getenv("LOG_FULL_CONTENT", "false").lower() == "true"  # Log full content flag

# Enhanced Configuration with new model management
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

# Model configuration with three-tier approach
BASE_MODEL = os.getenv("BASE_MODEL", "phi3:mini")  # Base model to pull/use
CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", "8192"))  # Desired context size
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", None)  # Final model name to use

# Generate default model name if not specified
if not DEFAULT_MODEL:
    # Convert base model name to safe custom model name
    safe_base_name = BASE_MODEL.replace(":", "-").replace("/", "-").replace("\\", "-")
    DEFAULT_MODEL = f"{safe_base_name}-{CONTEXT_SIZE}k"

# Other configurations
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "2.0"))

# Ollama generation parameters from environment
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))           # Default max tokens
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "4000"))         # Ollama-specific max prediction tokens
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))        # Sampling temperature
TOP_P = float(os.getenv("TOP_P", "0.9"))                    # Nucleus sampling
TOP_K = int(os.getenv("TOP_K", "40"))                       # Top-k sampling
REPEAT_PENALTY = float(os.getenv("REPEAT_PENALTY", "1.1"))  # Repetition penalty
REPEAT_LAST_N = int(os.getenv("REPEAT_LAST_N", "64"))       # Consider last N tokens for repetition
SEED = int(os.getenv("SEED", "-1"))                         # Random seed (-1 for random)
STOP_SEQUENCES = os.getenv("STOP_SEQUENCES", "").split(",") if os.getenv("STOP_SEQUENCES") else []

# Timeout configurations
OLLAMA_CONNECT_TIMEOUT = int(os.getenv("OLLAMA_CONNECT_TIMEOUT", "30"))
OLLAMA_READ_TIMEOUT = int(os.getenv("OLLAMA_READ_TIMEOUT", "300"))
OLLAMA_TOTAL_TIMEOUT = int(os.getenv("OLLAMA_TOTAL_TIMEOUT", "600"))

# Startup configurations
MAX_STARTUP_RETRIES = int(os.getenv("MAX_STARTUP_RETRIES", "30"))
STARTUP_RETRY_DELAY = float(os.getenv("STARTUP_RETRY_DELAY", "10.0"))

# Log enhanced configuration on startup
logger.info("LLM_ORCHESTRATOR_STARTUP - LLM Orchestrator Service starting", extra={
    'config': {
        'ollama_url': OLLAMA_URL,
        'base_model': BASE_MODEL,
        'context_size': CONTEXT_SIZE,
        'default_model': DEFAULT_MODEL,
        'generation_params': {
            'max_tokens': MAX_TOKENS,
            'num_predict': NUM_PREDICT,
            'temperature': TEMPERATURE,
            'top_p': TOP_P,
            'top_k': TOP_K,
            'repeat_penalty': REPEAT_PENALTY,
            'repeat_last_n': REPEAT_LAST_N,
            'seed': SEED,
            'stop_sequences': STOP_SEQUENCES
        },
        'timeouts': {
            'connect_timeout': OLLAMA_CONNECT_TIMEOUT,
            'read_timeout': OLLAMA_READ_TIMEOUT,
            'total_timeout': OLLAMA_TOTAL_TIMEOUT
        }
    }
})

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Global variables to store model state
ACTUAL_MODEL_NAME = None
BASE_MODEL_AVAILABLE = False
CUSTOM_MODEL_AVAILABLE = False
OLLAMA_READY = False

# Input/Output Models
class LLMRequest(BaseModel):
    prompt: str
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "num_predict": NUM_PREDICT,
            "context_size": CONTEXT_SIZE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "repeat_penalty": REPEAT_PENALTY,
            "repeat_last_n": REPEAT_LAST_N,
            "seed": SEED,
            "stop_sequences": STOP_SEQUENCES
        }
    )

class LLMResponse(BaseModel):
    response: str
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    model_status: Dict[str, str]
    metrics: Dict[str, Any]

# Metrics tracking
class Metrics:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        self.total_processing_time = 0
        self.retry_count = 0
        self.last_reset = datetime.utcnow()
    
    def reset(self):
        self.__init__()
    
    def to_dict(self):
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_tokens": self.total_tokens,
            "average_processing_time": self.total_processing_time / max(1, self.successful_requests),
            "retry_rate": self.retry_count / max(1, self.total_requests),
            "uptime": (datetime.utcnow() - self.last_reset).total_seconds()
        }

metrics = Metrics()

# Helper function to truncate text for logging
def truncate_for_log(text: str, max_length: int = 200) -> str:
    """Truncate text for logging purposes."""
    if not text:
        return ""
    
    if LOG_FULL_CONTENT or len(text) <= max_length:
        return text
    
    return text[:max_length] + "..." + f" (truncated from {len(text)} chars)"


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(tokenizer.encode(text))

def find_model_name(available_models: List[str], target_model: str) -> Optional[str]:
    """
    Find the actual model name from available models.
    Prioritizes exact matches, then semantic matches for custom models.
    """
    if not available_models:
        return None
    
    # Log what we're looking for and what's available
    logger.info(f"Looking for model: '{target_model}' in available models: {available_models}")
    
    # ✅ PRIORITY 1: Exact match first (highest priority)
    if target_model in available_models:
        logger.info(f"Found exact match: {target_model}")
        return target_model
    
    # ✅ PRIORITY 2: Check if target_model:latest exists
    if f"{target_model}:latest" in available_models:
        logger.info(f"Found with :latest tag: {target_model}:latest")
        return f"{target_model}:latest"
    
    # ✅ PRIORITY 3: For custom models, look for semantic equivalents
    # If target is phi3-mini-8192k, also look for phi3/mini-8192k and phi3:mini-8192k
    if "-" in target_model and "8192k" in target_model:
        # Extract base and context from target
        parts = target_model.split("-")
        if len(parts) >= 3:  # e.g., ["phi3", "mini", "8192k"]
            base_part = parts[0]  # "phi3"
            middle_part = parts[1]  # "mini" 
            context_part = parts[2]  # "8192k"
            
            # Generate semantic equivalents
            semantic_variants = [
                f"{base_part}:{middle_part}-{context_part}",           # phi3:mini-8192k
                f"{base_part}/{middle_part}-{context_part}",           # phi3/mini-8192k
                f"{base_part}:{middle_part}-{context_part}:latest",    # phi3:mini-8192k:latest
                f"{base_part}/{middle_part}-{context_part}:latest",    # phi3/mini-8192k:latest
                f"{base_part}:{middle_part}:{context_part}",           # phi3:mini:8192k
                f"{base_part}/{middle_part}/{context_part}",           # phi3/mini/8192k
            ]
            
            logger.info(f"Checking semantic variants for custom model: {semantic_variants}")
            
            # Check each semantic variant
            for variant in semantic_variants:
                if variant in available_models:
                    logger.info(f"Found semantic match for custom model: {variant}")
                    return variant
    
    # ✅ PRIORITY 4: Original fuzzy matching logic
    # Convert target model to different separator formats for comparison
    target_variants = [
        target_model,
        target_model.replace("-", ":"),  # phi3-mini-8192k -> phi3:mini:8192k
        target_model.replace("-", "/"),  # phi3-mini-8192k -> phi3/mini/8192k
        target_model.replace(":", "-"),  # phi3:mini -> phi3-mini
        target_model.replace("/", "-"),  # phi3/mini -> phi3-mini
        target_model.replace(":", "/"),  # phi3:mini -> phi3/mini
        target_model.replace("/", ":"),  # phi3/mini -> phi3:mini
    ]
    
    # Remove duplicates while preserving order
    target_variants = list(dict.fromkeys(target_variants))
    
    logger.info(f"Trying variant matches: {target_variants}")
    
    for variant in target_variants:
        # Check exact match with variants
        if variant in available_models:
            logger.info(f"Found variant exact match: {variant}")
            return variant
            
        # Check with :latest suffix
        if f"{variant}:latest" in available_models:
            logger.info(f"Found variant with :latest: {variant}:latest")
            return f"{variant}:latest"
    
    # ✅ PRIORITY 5: Fuzzy matching for similar models
    # Extract the base model name (before first separator) and context size
    target_base = target_model.split(":")[0].split("/")[0].split("-")[0]
    target_context = None
    
    # Try to extract context size from target model name
    for part in target_model.replace(":", "-").replace("/", "-").split("-"):
        if part.endswith("k") and part[:-1].isdigit():
            target_context = part
            break
    
    logger.info(f"Fuzzy matching with base: '{target_base}', context: '{target_context}'")
    
    for model in available_models:
        model_base = model.split(":")[0].split("/")[0].split("-")[0]
        model_full = model.replace(":", "-").replace("/", "-")
        
        # Check if base model matches and context size matches
        if (model_base == target_base and 
            target_context and 
            target_context in model_full):
            logger.info(f"Found fuzzy match: {model} (base: {model_base}, context: {target_context})")
            return model
    
    # ✅ PRIORITY 6: Last resort - find any model with the same base name
    for model in available_models:
        model_base = model.split(":")[0].split("/")[0].split("-")[0]
        if model_base == target_base:
            logger.info(f"Found base model fallback: {model}")
            return model
    
    logger.warning(f"No match found for target model: {target_model}")
    return None

# Add this helper function after the find_model_name function:

def is_custom_model(model_name: str, base_model: str, context_size: int) -> bool:
    """
    Check if a model name represents a custom model with enhanced context.
    """
    if not model_name:
        return False
    
    # Extract base name and check for context size indicator
    model_normalized = model_name.replace(":", "-").replace("/", "-").replace(":latest", "")
    base_normalized = base_model.replace(":", "-").replace("/", "-")
    context_indicator = f"{context_size}k"
    
    # Check if this model contains both the base model name and context size
    return (base_normalized in model_normalized and 
            context_indicator in model_normalized and
            model_normalized != base_normalized)

async def check_ollama_connectivity(request_id: str = "startup") -> bool:
    """Check if Ollama is accessible."""
    try:
        log_with_context(
            logger, 'info',
            f"OLLAMA_CONNECTIVITY_CHECK - Checking Ollama connectivity",
            request_id, "check_connectivity",
            ollama_url=OLLAMA_URL
        )
        
        timeout = aiohttp.ClientTimeout(
            total=30,
            connect=10,
            sock_read=10
        )
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{OLLAMA_URL}/api/tags") as response:
                if response.status == 200:
                    log_with_context(
                        logger, 'info',
                        f"OLLAMA_CONNECTIVITY_SUCCESS - Ollama is accessible",
                        request_id, "check_connectivity",
                        status_code=response.status
                    )
                    return True
                else:
                    log_with_context(
                        logger, 'warning',
                        f"OLLAMA_CONNECTIVITY_ERROR - Ollama returned non-200 status",
                        request_id, "check_connectivity",
                        status_code=response.status
                    )
                    return False
    except Exception as e:
        log_with_context(
            logger, 'error',
            f"OLLAMA_CONNECTIVITY_ERROR - Failed to connect to Ollama",
            request_id, "check_connectivity",
            error=str(e),
            traceback=traceback.format_exc()
        )
        return False

async def wait_for_ollama(max_retries: int = 30, delay: float = 10.0, request_id: str = "startup") -> bool:
    """Wait for Ollama to become available."""
    log_with_context(
        logger, 'info',
        f"OLLAMA_WAIT_START - Waiting for Ollama to become available",
        request_id, "wait_for_ollama",
        max_retries=max_retries,
        delay=delay
    )
    
    for attempt in range(max_retries):
        try:
            if await check_ollama_connectivity(request_id):
                log_with_context(
                    logger, 'info',
                    f"OLLAMA_WAIT_SUCCESS - Ollama is now available",
                    request_id, "wait_for_ollama",
                    attempt=attempt + 1
                )
                return True
            
            log_with_context(
                logger, 'info',
                f"OLLAMA_WAIT_RETRY - Ollama not yet available, retrying",
                request_id, "wait_for_ollama",
                attempt=attempt + 1,
                max_retries=max_retries,
                next_retry_in_seconds=delay
            )
            
            await asyncio.sleep(delay)
        except Exception as e:
            log_with_context(
                logger, 'error',
                f"OLLAMA_WAIT_ERROR - Error while waiting for Ollama",
                request_id, "wait_for_ollama",
                attempt=attempt + 1,
                error=str(e)
            )
            await asyncio.sleep(delay)
    
    log_with_context(
        logger, 'error',
        f"OLLAMA_WAIT_TIMEOUT - Ollama did not become available after maximum retries",
        request_id, "wait_for_ollama",
        max_retries=max_retries,
        total_wait_time_seconds=max_retries * delay
    )
    return False

async def get_available_models(request_id: str = "startup") -> List[str]:
    """Get list of available models from Ollama."""
    try:
        log_with_context(
            logger, 'info',
            f"GET_MODELS_START - Getting available models from Ollama",
            request_id, "get_models"
        )
        
        timeout = aiohttp.ClientTimeout(
            total=30,
            connect=10,
            sock_read=20
        )
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{OLLAMA_URL}/api/tags") as response:
                if response.status != 200:
                    error_text = await response.text()
                    log_with_context(
                        logger, 'error',
                        f"GET_MODELS_ERROR - Failed to get models from Ollama",
                        request_id, "get_models",
                        status_code=response.status,
                        error_response=error_text
                    )
                    return []
                
                models_data = await response.json()
                available_models = [m['name'] for m in models_data.get('models', [])]
                
                log_with_context(
                    logger, 'info',
                    f"GET_MODELS_SUCCESS - Retrieved available models",
                    request_id, "get_models",
                    model_count=len(available_models),
                    available_models=available_models
                )
                
                return available_models
    except Exception as e:
        log_with_context(
            logger, 'error',
            f"GET_MODELS_ERROR - Error getting models from Ollama",
            request_id, "get_models",
            error=str(e),
            traceback=traceback.format_exc()
        )
        return []

async def create_custom_model(base_model: str, custom_name: str, context_size: int, request_id: str = "startup") -> bool:
    """Create custom model with specific context size."""
    log_with_context(
        logger, 'info',
        f"CUSTOM_MODEL_CREATE_START - Creating custom model",
        request_id, "create_custom_model",
        base_model=base_model,
        custom_name=custom_name,
        context_size=context_size
    )
    
    try:
        # Create Modelfile content with comprehensive parameters
        modelfile_content = f"""FROM {base_model}
PARAMETER num_ctx {context_size}
PARAMETER num_predict {NUM_PREDICT}
PARAMETER temperature {TEMPERATURE}
PARAMETER top_p {TOP_P}
PARAMETER top_k {TOP_K}
PARAMETER repeat_penalty {REPEAT_PENALTY}
PARAMETER repeat_last_n {REPEAT_LAST_N}
"""
        
        # Add seed if specified (not -1)
        if SEED != -1:
            modelfile_content += f"PARAMETER seed {SEED}\n"
            
        # Add stop sequences if specified
        if STOP_SEQUENCES:
            for stop_seq in STOP_SEQUENCES:
                if stop_seq.strip():
                    modelfile_content += f'PARAMETER stop "{stop_seq.strip()}"\n'
        
        timeout = aiohttp.ClientTimeout(
            total=300,  # 5 minutes for model creation
            connect=30,
            sock_read=180
        )
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{OLLAMA_URL}/api/create",
                json={
                    "name": custom_name,
                    "modelfile": modelfile_content
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    log_with_context(
                        logger, 'error',
                        f"CUSTOM_MODEL_CREATE_ERROR - Failed to create custom model",
                        request_id, "create_custom_model",
                        base_model=base_model,
                        custom_name=custom_name,
                        status_code=response.status,
                        error_response=error_text
                    )
                    return False
                
                # Handle streaming response for model creation
                async for line in response.content:
                    try:
                        if line.strip():
                            data = json.loads(line)
                            if "status" in data:
                                status = data["status"]
                                log_with_context(
                                    logger, 'info',
                                    f"CUSTOM_MODEL_CREATE_STATUS - Creation status update",
                                    request_id, "create_custom_model",
                                    custom_name=custom_name,
                                    status=status
                                )
                                
                                if "success" in status.lower() or "complete" in status.lower():
                                    log_with_context(
                                        logger, 'info',
                                        f"CUSTOM_MODEL_CREATE_SUCCESS - Custom model created successfully",
                                        request_id, "create_custom_model",
                                        custom_name=custom_name
                                    )
                                    return True
                                    
                            if "error" in data:
                                error_msg = data["error"]
                                log_with_context(
                                    logger, 'error',
                                    f"CUSTOM_MODEL_CREATE_ERROR - Error during custom model creation",
                                    request_id, "create_custom_model",
                                    custom_name=custom_name,
                                    error=error_msg
                                )
                                return False
                    except json.JSONDecodeError:
                        continue
                
                log_with_context(
                    logger, 'info',
                    f"CUSTOM_MODEL_CREATE_COMPLETED - Custom model creation stream ended",
                    request_id, "create_custom_model",
                    custom_name=custom_name
                )
                return True
                
    except Exception as e:
        log_with_context(
            logger, 'error',
            f"CUSTOM_MODEL_CREATE_ERROR - Error creating custom model",
            request_id, "create_custom_model",
            base_model=base_model,
            custom_name=custom_name,
            error=str(e),
            traceback=traceback.format_exc()
        )
        return False

async def pull_model(model_name: str, request_id: str = "startup") -> bool:
    """Pull model with detailed logging and timeout handling."""
    log_with_context(
        logger, 'info',
        f"MODEL_PULL_START - Starting to pull model",
        request_id, "pull_model",
        model_name=model_name
    )
    
    try:
        # Use longer timeout for model pulling
        timeout = aiohttp.ClientTimeout(
            total=1800,  # 30 minutes total
            connect=30,
            sock_read=300
        )
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{OLLAMA_URL}/api/pull",
                json={"name": model_name}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    log_with_context(
                        logger, 'error',
                        f"MODEL_PULL_ERROR - Failed to start model pull",
                        request_id, "pull_model",
                        model_name=model_name,
                        status_code=response.status,
                        error_response=error_text
                    )
                    return False
                
                # Handle streaming response
                async for line in response.content:
                    try:
                        if line.strip():
                            data = json.loads(line)
                            if "status" in data:
                                status = data["status"]
                                log_with_context(
                                    logger, 'info',
                                    f"MODEL_PULL_STATUS - Pull status update",
                                    request_id, "pull_model",
                                    model_name=model_name,
                                    status=status,
                                    completed=data.get("completed", 0),
                                    total=data.get("total", 0)
                                )
                                
                                if "success" in status.lower() or "complete" in status.lower():
                                    log_with_context(
                                        logger, 'info',
                                        f"MODEL_PULL_SUCCESS - Model pull completed successfully",
                                        request_id, "pull_model",
                                        model_name=model_name
                                    )
                                    return True
                                    
                            if "error" in data:
                                error_msg = data["error"]
                                log_with_context(
                                    logger, 'error',
                                    f"MODEL_PULL_ERROR - Error during model pull",
                                    request_id, "pull_model",
                                    model_name=model_name,
                                    error=error_msg
                                )
                                return False
                    except json.JSONDecodeError:
                        continue
                
                log_with_context(
                    logger, 'info',
                    f"MODEL_PULL_COMPLETED - Model pull stream ended",
                    request_id, "pull_model",
                    model_name=model_name
                )
                return True
                
    except asyncio.TimeoutError:
        log_with_context(
            logger, 'error',
            f"MODEL_PULL_TIMEOUT - Model pull timed out",
            request_id, "pull_model",
            model_name=model_name,
            timeout_seconds=1800
        )
        return False
    except Exception as e:
        log_with_context(
            logger, 'error',
            f"MODEL_PULL_ERROR - Error during model pull",
            request_id, "pull_model",
            model_name=model_name,
            error=str(e),
            traceback=traceback.format_exc()
        )
        return False

async def ensure_models_available(request_id: str = "startup") -> tuple[Optional[str], bool, bool]:
    """
    Ensure both base and custom models are available.
    Returns: (actual_model_name, base_model_ready, custom_model_ready)
    """
    global BASE_MODEL_AVAILABLE, CUSTOM_MODEL_AVAILABLE
    
    log_with_context(
        logger, 'info',
        f"ENSURE_MODELS_START - Ensuring models are available",
        request_id, "ensure_models",
        base_model=BASE_MODEL,
        default_model=DEFAULT_MODEL,
        context_size=CONTEXT_SIZE
    )
    
    # Get available models
    available_models = await get_available_models(request_id)
    
    # Check if base model is available
    base_model_found = find_model_name(available_models, BASE_MODEL)
    if base_model_found:
        BASE_MODEL_AVAILABLE = True
        log_with_context(
            logger, 'info',
            f"ENSURE_MODELS_BASE_FOUND - Base model is available",
            request_id, "ensure_models",
            base_model=BASE_MODEL,
            found_as=base_model_found
        )
    else:
        # Try to pull base model
        log_with_context(
            logger, 'info',
            f"ENSURE_MODELS_BASE_PULLING - Base model not found, attempting to pull",
            request_id, "ensure_models",
            base_model=BASE_MODEL
        )
        
        if await pull_model(BASE_MODEL, request_id):
            await asyncio.sleep(5)  # Wait for model to be ready
            available_models = await get_available_models(request_id)
            base_model_found = find_model_name(available_models, BASE_MODEL)
            BASE_MODEL_AVAILABLE = base_model_found is not None
            
            if BASE_MODEL_AVAILABLE:
                log_with_context(
                    logger, 'info',
                    f"ENSURE_MODELS_BASE_PULLED - Base model pulled successfully",
                    request_id, "ensure_models",
                    base_model=BASE_MODEL,
                    found_as=base_model_found
                )
            else:
                log_with_context(
                    logger, 'error',
                    f"ENSURE_MODELS_BASE_FAILED - Failed to pull base model",
                    request_id, "ensure_models",
                    base_model=BASE_MODEL
                )
                return None, False, False
    
    # Check if custom model is available
    custom_model_found = find_model_name(available_models, DEFAULT_MODEL)
    if custom_model_found:
        CUSTOM_MODEL_AVAILABLE = True
        log_with_context(
            logger, 'info',
            f"ENSURE_MODELS_CUSTOM_FOUND - Custom model is available",
            request_id, "ensure_models",
            default_model=DEFAULT_MODEL,
            found_as=custom_model_found
        )
        return custom_model_found, BASE_MODEL_AVAILABLE, True
    
    # Create custom model if base model is available
    if BASE_MODEL_AVAILABLE:
        log_with_context(
            logger, 'info',
            f"ENSURE_MODELS_CUSTOM_CREATING - Creating custom model",
            request_id, "ensure_models",
            base_model=base_model_found,
            custom_model=DEFAULT_MODEL,
            context_size=CONTEXT_SIZE
        )
        
        if await create_custom_model(base_model_found, DEFAULT_MODEL, CONTEXT_SIZE, request_id):
            await asyncio.sleep(5)  # Wait for model to be ready
            available_models = await get_available_models(request_id)
            custom_model_found = find_model_name(available_models, DEFAULT_MODEL)
            CUSTOM_MODEL_AVAILABLE = custom_model_found is not None
            
            if CUSTOM_MODEL_AVAILABLE:
                log_with_context(
                    logger, 'info',
                    f"ENSURE_MODELS_CUSTOM_CREATED - Custom model created successfully",
                    request_id, "ensure_models",
                    custom_model=DEFAULT_MODEL,
                    found_as=custom_model_found
                )
                return custom_model_found, True, True
            else:
                log_with_context(
                    logger, 'warning',
                    f"ENSURE_MODELS_CUSTOM_FAILED - Custom model creation failed, using base model",
                    request_id, "ensure_models",
                    base_model=base_model_found
                )
                return base_model_found, True, False
    
    log_with_context(
        logger, 'error',
        f"ENSURE_MODELS_FAILED - Failed to ensure any model availability",
        request_id, "ensure_models",
        base_model=BASE_MODEL,
        default_model=DEFAULT_MODEL
    )
    return None, False, False

async def generate_with_retry(request: LLMRequest, max_retries: int = 3, request_id: str = "N/A") -> Dict[str, Any]:
    """Generate response with retry logic and improved error handling."""
    global ACTUAL_MODEL_NAME, OLLAMA_READY
    
    if not OLLAMA_READY:
        log_with_context(
            logger, 'error',
            f"GENERATE_ERROR - Ollama not ready",
            request_id, "generate",
            ollama_ready=OLLAMA_READY
        )
        raise HTTPException(status_code=503, detail="Ollama service is not ready")
    
    model_to_use = ACTUAL_MODEL_NAME or DEFAULT_MODEL
    
    # Override context size from request if provided
    context_size = request.parameters.get("context_size", CONTEXT_SIZE)

    # Configure timeouts
    timeout = aiohttp.ClientTimeout(
        total=OLLAMA_TOTAL_TIMEOUT,
        connect=OLLAMA_CONNECT_TIMEOUT,
        sock_read=OLLAMA_READ_TIMEOUT
    )
    
    log_with_context(
        logger, 'info',
        f"GENERATE_START - Starting generation",
        request_id, "generate",
        model=model_to_use,
        prompt_length=len(request.prompt),
        parameters=request.parameters,
        context_size=context_size,
        custom_model_available=CUSTOM_MODEL_AVAILABLE,
        base_model_available=BASE_MODEL_AVAILABLE,
        timeout_config={
            'total': OLLAMA_TOTAL_TIMEOUT,
            'connect': OLLAMA_CONNECT_TIMEOUT,
            'sock_read': OLLAMA_READ_TIMEOUT
        }
    )
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            log_with_context(
                logger, 'info',
                f"GENERATE_ATTEMPT - Starting generation attempt",
                request_id, "generate",
                attempt=attempt + 1,
                max_retries=max_retries,
                model=model_to_use
            )
            
            # Call Ollama API with comprehensive parameters
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Extract parameters with defaults
                temperature = request.parameters.get("temperature", TEMPERATURE)
                num_predict = request.parameters.get("num_predict", request.parameters.get("max_tokens", NUM_PREDICT))
                top_p = request.parameters.get("top_p", TOP_P)
                top_k = request.parameters.get("top_k", TOP_K)
                repeat_penalty = request.parameters.get("repeat_penalty", REPEAT_PENALTY)
                repeat_last_n = request.parameters.get("repeat_last_n", REPEAT_LAST_N)
                seed = request.parameters.get("seed", SEED)
                stop_sequences = request.parameters.get("stop_sequences", STOP_SEQUENCES)
                
                # Build options dict
                options = {
                    "num_ctx": context_size,
                    "num_predict": num_predict,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repeat_penalty": repeat_penalty,
                    "repeat_last_n": repeat_last_n
                }
                
                # Add seed if not -1
                if seed != -1:
                    options["seed"] = seed
                
                # Add stop sequences if provided
                if stop_sequences and any(s.strip() for s in stop_sequences):
                    options["stop"] = [s.strip() for s in stop_sequences if s.strip()]
                
                request_data = {
                    "model": model_to_use,
                    "prompt": request.prompt,
                    "options": options,
                    "stream": False
                }
                
                log_with_context(
                    logger, 'debug',
                    f"GENERATE_REQUEST - Sending request to Ollama",
                    request_id, "generate",
                    request_data_preview={
                        'model': request_data['model'],
                        'prompt_length': len(request_data['prompt']),
                        'options': request_data['options']
                    }
                )
                
                async with session.post(
                    f"{OLLAMA_URL}/api/generate",
                    json=request_data
                ) as response:
                    if response.status != 200:
                        error_detail = f"Ollama API error: Status {response.status}"
                        try:
                            error_body = await response.text()
                            error_detail += f", Response: {error_body}"
                        except:
                            pass
                        
                        log_with_context(
                            logger, 'error',
                            f"GENERATE_API_ERROR - Ollama API returned error",
                            request_id, "generate",
                            status_code=response.status,
                            error_detail=error_detail,
                            attempt=attempt + 1
                        )
                        
                        if attempt == max_retries - 1:
                            raise HTTPException(status_code=response.status, detail=error_detail)
                        continue
                    
                    result = await response.json()
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    # Extract token count from response
                    token_count = result.get("eval_count", 0) or len(tokenizer.encode(result.get("response", "")))
                    
                    response_data = {
                        "response": result.get("response", ""),
                        "metadata": {
                            "model": model_to_use,
                            "base_model": BASE_MODEL,
                            "context_size": context_size,
                            "custom_model_used": is_custom_model(model_to_use, BASE_MODEL, context_size),
                            "processing_time": processing_time,
                            "token_count": token_count,
                            "attempt": attempt + 1,
                            "generation_params": {
                                "num_predict": num_predict,
                                "temperature": temperature,
                                "top_p": top_p,
                                "top_k": top_k,
                                "repeat_penalty": repeat_penalty,
                                "repeat_last_n": repeat_last_n,
                                "seed": seed if seed != -1 else "random",
                                "stop_sequences": stop_sequences
                            },
                            "ollama_stats": {
                                "prompt_eval_count": result.get("prompt_eval_count", 0),
                                "eval_count": result.get("eval_count", 0),
                                "total_duration": result.get("total_duration", 0),
                                "load_duration": result.get("load_duration", 0),
                                "prompt_eval_duration": result.get("prompt_eval_duration", 0),
                                "eval_duration": result.get("eval_duration", 0)
                            }
                        }
                    }
                    
                    log_with_context(
                        logger, 'info',
                        f"GENERATE_SUCCESS - Generation completed successfully",
                        request_id, "generate",
                        model=model_to_use,
                        processing_time_seconds=round(processing_time, 3),
                        token_count=token_count,
                        attempt=attempt + 1,
                        response_length=len(response_data["response"]),
                        context_size=context_size,
                        custom_model_used=response_data["metadata"]["custom_model_used"]
                    )
                    
                    return response_data
                    
        except asyncio.TimeoutError:
            log_with_context(
                logger, 'warning',
                f"GENERATE_TIMEOUT - Generation attempt timed out",
                request_id, "generate",
                attempt=attempt + 1,
                timeout_seconds=OLLAMA_TOTAL_TIMEOUT
            )
            
            if attempt == max_retries - 1:
                raise HTTPException(status_code=408, detail=f"Generation timeout after {OLLAMA_TOTAL_TIMEOUT} seconds")
            
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            continue
            
        except aiohttp.ClientError as e:
            log_with_context(
                logger, 'error',
                f"GENERATE_CLIENT_ERROR - HTTP client error during generation",
                request_id, "generate",
                error=str(e),
                error_type=type(e).__name__,
                attempt=attempt + 1
            )
            
            if attempt == max_retries - 1:
                raise HTTPException(status_code=503, detail=f"Failed to connect to Ollama: {str(e)}")
            
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            continue
            
        except Exception as e:
            log_with_context(
                logger, 'error',
                f"GENERATE_ERROR - Unexpected error during generation",
                request_id, "generate",
                error=str(e),
                traceback=traceback.format_exc(),
                attempt=attempt + 1
            )
            
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=f"Generation failed after {max_retries} attempts: {str(e)}")
            
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            continue

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI app."""
    global ACTUAL_MODEL_NAME, OLLAMA_READY, BASE_MODEL_AVAILABLE, CUSTOM_MODEL_AVAILABLE
    
    # Startup
    startup_request_id = str(uuid.uuid4())
    logger.info("LLM_ORCHESTRATOR_STARTUP - Starting up LLM Orchestrator service...")
    
    try:
        # Wait for Ollama to become available
        if not await wait_for_ollama(MAX_STARTUP_RETRIES, STARTUP_RETRY_DELAY, startup_request_id):
            raise Exception("Ollama did not become available within the timeout period")
        
        # Ensure models are available
        ACTUAL_MODEL_NAME, BASE_MODEL_AVAILABLE, CUSTOM_MODEL_AVAILABLE = await ensure_models_available(startup_request_id)
        
        if not ACTUAL_MODEL_NAME:
            raise Exception(f"Required models could not be made available")
        
        OLLAMA_READY = True
        
        log_with_context(
            logger, 'info',
            f"LLM_ORCHESTRATOR_STARTUP_SUCCESS - Startup completed successfully",
            startup_request_id, "startup",
            base_model=BASE_MODEL,
            default_model=DEFAULT_MODEL,
            actual_model=ACTUAL_MODEL_NAME,
            context_size=CONTEXT_SIZE,
            base_model_available=BASE_MODEL_AVAILABLE,
            custom_model_available=CUSTOM_MODEL_AVAILABLE,
            ollama_ready=OLLAMA_READY
        )
        
    except Exception as e:
        log_with_context(
            logger, 'error',
            f"LLM_ORCHESTRATOR_STARTUP_FAILED - Startup failed",
            startup_request_id, "startup",
            error=str(e),
            traceback=traceback.format_exc()
        )
        raise
    
    yield
    
    # Shutdown
    logger.info("LLM_ORCHESTRATOR_SHUTDOWN - Shutting down LLM Orchestrator service...")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="LLM Orchestrator Service",
    description="Service for orchestrating LLM interactions with enhanced model management",
    version="2.0.0",
    lifespan=lifespan
)

@app.post("/generate", response_model=LLMResponse)
async def generate_response(request: LLMRequest):
    """
    Generate response from LLM with retry logic and monitoring.
    """
    request_id = str(uuid.uuid4())
    endpoint_logger = get_contextual_logger(request_id, "/generate")
    start_time = time.time()
    
    try:
        token_count = count_tokens(request.prompt)
        
        log_with_context(
            logger, 'info',
            f"GENERATE_REQUEST_START - Received generation request",
            request_id, "/generate",
            prompt_tokens=token_count,
            parameters=request.parameters
        )
        
        
        # Update metrics
        metrics.total_requests += 1
        
        # Generate response with retry logic
        result = await generate_with_retry(request, request_id=request_id)
        
        # Update success metrics
        metrics.successful_requests += 1
        metrics.total_tokens += result['metadata']['token_count']
        metrics.total_processing_time += result['metadata']['processing_time']
        
        total_time = time.time() - start_time
        
        log_with_context(
            logger, 'info',
            f"GENERATE_REQUEST_SUCCESS - Generation request completed successfully",
            request_id, "/generate",
            input_tokens=token_count,
            output_tokens=result['metadata']['token_count'],
            processing_time_seconds=round(result['metadata']['processing_time'], 3),
            total_time_seconds=round(total_time, 3),
            model=result['metadata']['model'],
            context_size=result['metadata']['context_size'],
            custom_model_used=result['metadata']['custom_model_used']
        )
        
        return LLMResponse(**result)
    
    except HTTPException as he:
        metrics.failed_requests += 1
        total_time = time.time() - start_time
        
        log_with_context(
            logger, 'error',
            f"GENERATE_REQUEST_HTTP_ERROR - HTTP exception occurred",
            request_id, "/generate",
            error_detail=str(he.detail) if hasattr(he, 'detail') else str(he),
            status_code=getattr(he, 'status_code', 'unknown'),
            total_time_seconds=round(total_time, 3)
        )
        raise he
    
    except Exception as e:
        metrics.failed_requests += 1
        total_time = time.time() - start_time
        
        log_with_context(
            logger, 'error',
            f"GENERATE_REQUEST_ERROR - Unexpected error occurred",
            request_id, "/generate",
            error=str(e),
            traceback=traceback.format_exc(),
            total_time_seconds=round(total_time, 3)
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint."""
    global ACTUAL_MODEL_NAME, OLLAMA_READY, BASE_MODEL_AVAILABLE, CUSTOM_MODEL_AVAILABLE
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        log_with_context(
            logger, 'info',
            f"HEALTH_CHECK_START - Starting health check",
            request_id, "/health"
        )
        
        # Check Ollama connectivity
        ollama_accessible = await check_ollama_connectivity(request_id)
        
        model_status = {}
        if ollama_accessible:
            # Check model availability
            available_models = await get_available_models(request_id)
            
            # Check base model
            base_model_found = find_model_name(available_models, BASE_MODEL)
            model_status[f"base_model_{BASE_MODEL}"] = "available" if base_model_found else "unavailable"
            
            # Check custom model
            custom_model_found = find_model_name(available_models, DEFAULT_MODEL)
            model_status[f"custom_model_{DEFAULT_MODEL}"] = "available" if custom_model_found else "unavailable"
            
            # Update global state
            BASE_MODEL_AVAILABLE = base_model_found is not None
            CUSTOM_MODEL_AVAILABLE = custom_model_found is not None
            
            if custom_model_found:
                ACTUAL_MODEL_NAME = custom_model_found
                OLLAMA_READY = True
            elif base_model_found:
                ACTUAL_MODEL_NAME = base_model_found
                OLLAMA_READY = True
            else:
                ACTUAL_MODEL_NAME = None
                OLLAMA_READY = False
        else:
            model_status = {
                f"base_model_{BASE_MODEL}": "unavailable",
                f"custom_model_{DEFAULT_MODEL}": "unavailable"
            }
            OLLAMA_READY = False
        
        overall_status = "healthy" if (ollama_accessible and ACTUAL_MODEL_NAME and OLLAMA_READY) else "unhealthy"
        
        total_time = time.time() - start_time
        
        # Enhanced metrics
        enhanced_metrics = metrics.to_dict()
        enhanced_metrics.update({
            "base_model": BASE_MODEL,
            "default_model": DEFAULT_MODEL,
            "context_size": CONTEXT_SIZE,
            "actual_model": ACTUAL_MODEL_NAME,
            "base_model_available": BASE_MODEL_AVAILABLE,
            "custom_model_available": CUSTOM_MODEL_AVAILABLE,
            "generation_defaults": {
                "num_predict": NUM_PREDICT,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "repeat_penalty": REPEAT_PENALTY,
                "repeat_last_n": REPEAT_LAST_N,
                "seed": SEED if SEED != -1 else "random"
            }
        })
        
        health_data = {
            "status": overall_status,
            "model_status": model_status,
            "metrics": enhanced_metrics
        }
        
        log_with_context(
            logger, 'info',
            f"HEALTH_CHECK_SUCCESS - Health check completed",
            request_id, "/health",
            status=overall_status,
            ollama_accessible=ollama_accessible,
            actual_model=ACTUAL_MODEL_NAME,
            base_model_available=BASE_MODEL_AVAILABLE,
            custom_model_available=CUSTOM_MODEL_AVAILABLE,
            ollama_ready=OLLAMA_READY,
            total_time_seconds=round(total_time, 3)
        )
        
        return health_data
        
    except Exception as e:
        total_time = time.time() - start_time
        
        log_with_context(
            logger, 'error',
            f"HEALTH_CHECK_ERROR - Health check failed",
            request_id, "/health",
            error=str(e),
            traceback=traceback.format_exc(),
            total_time_seconds=round(total_time, 3)
        )
        
        return {
            "status": "unhealthy",
            "model_status": {
                f"base_model_{BASE_MODEL}": "unavailable",
                f"custom_model_{DEFAULT_MODEL}": "unavailable"
            },
            "metrics": metrics.to_dict()
        }

@app.post("/reset-metrics")
async def reset_metrics():
    """Reset metrics counters."""
    request_id = str(uuid.uuid4())
    
    log_with_context(
        logger, 'info',
        f"METRICS_RESET - Resetting metrics",
        request_id, "/reset-metrics"
    )
    
    metrics.reset()
    return {"status": "success", "message": "Metrics reset successfully"}

@app.get("/models")
async def get_models():
    """Get information about available models."""
    request_id = str(uuid.uuid4())
    
    try:
        available_models = await get_available_models(request_id)
        
        return {
            "base_model": BASE_MODEL,
            "default_model": DEFAULT_MODEL,
            "context_size": CONTEXT_SIZE,
            "actual_model": ACTUAL_MODEL_NAME,
            "base_model_available": BASE_MODEL_AVAILABLE,
            "custom_model_available": CUSTOM_MODEL_AVAILABLE,
            "available_models": available_models,
            "generation_defaults": {
                "max_tokens": MAX_TOKENS,
                "num_predict": NUM_PREDICT,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "repeat_penalty": REPEAT_PENALTY,
                "repeat_last_n": REPEAT_LAST_N,
                "seed": SEED if SEED != -1 else "random",
                "stop_sequences": STOP_SEQUENCES
            }
        }
    except Exception as e:
        log_with_context(
            logger, 'error',
            f"MODELS_INFO_ERROR - Error getting model information",
            request_id, "/models",
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))