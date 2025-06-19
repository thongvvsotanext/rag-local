# services/prompt_builder/main.py - Simplified version for better LLM performance

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import logging
import sys
from datetime import datetime
import json
import tiktoken
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('prompt_builder.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful enterprise assistant.")

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Input/Output Models
class EvidenceChunk(BaseModel):
    text: str
    source: str
    score: float
    type: str = "document"  # document, web, sql
    metadata: Optional[Dict[str, Any]] = None

class PromptRequest(BaseModel):
    query: str
    context: str
    retrieved_data: List[Dict[str, Any]]
    filters: Optional[Dict[str, Any]] = None

class PromptResponse(BaseModel):
    prompt: str
    metadata: Dict[str, Any]

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(tokenizer.encode(text))

def truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token limit."""
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens])

def format_evidence_simple(evidence: List[Dict[str, Any]]) -> str:
    """Format evidence chunks with simple citations - SIMPLIFIED VERSION."""
    # Sort evidence by score in descending order
    sorted_evidence = sorted(evidence, key=lambda x: x.get('score', 0), reverse=True)
    # Take top 3 chunks for better focus
    top_evidence = sorted_evidence[:3]
    
    formatted = []
    for i, chunk in enumerate(top_evidence, 1):
        text = chunk.get("text", "")
        source = chunk.get("source", "unknown")
        chunk_type = chunk.get("type", "document")
        
        # Simplify source information
        if chunk_type == "web":
            source_info = f"web, {source}"
        elif chunk_type == "sql":
            source_info = f"database, {source}"
        else:
            source_info = f"{chunk_type}, {source}"
        
        # Keep citation simple and clean
        formatted.append(f"[{i}] {text} (Source: {source_info})")
    
    return "\n".join(formatted)

def build_simplified_prompt(request: PromptRequest) -> str:
    """Build SIMPLIFIED prompt structure for better LLM performance."""
    
    # SIMPLIFIED STRUCTURE - No complex formatting instructions
    prompt_parts = []
    
    # 1. Simple system instruction
    prompt_parts.append("You are a helpful enterprise assistant.")
    
    # 2. Add relevant information (evidence) FIRST - this helps the model focus
    if request.retrieved_data:
        evidence_text = format_evidence_simple(request.retrieved_data)
        prompt_parts.append(f"\nRelevant information:\n{evidence_text}")
    
    # 3. Add context if available (but keep it brief)
    if request.context and len(request.context.strip()) > 0:
        # Truncate context if too long
        context = request.context[:300] + "..." if len(request.context) > 300 else request.context
        prompt_parts.append(f"\nPrevious context: {context}")
    
    # 4. User query - direct and clear
    prompt_parts.append(f"\nUser query: {request.query}")
    
    # 5. SIMPLIFIED instructions - no markdown formatting
    prompt_parts.append("\nInstructions:")
    prompt_parts.append("1. Provide a clear and helpful response")
    prompt_parts.append("2. Include citations like [1] when referencing information")
    prompt_parts.append("3. Be concise and focused")
    
    # Combine with simple newlines
    final_prompt = "\n".join(prompt_parts)
    
    return final_prompt

def build_ultra_simple_prompt(request: PromptRequest) -> str:
    """Build ULTRA SIMPLE prompt - minimal structure for maximum reliability."""
    
    # Start with just the essential context
    parts = ["You are a helpful assistant. Answer using the provided information."]
    
    # Add evidence in simple format
    if request.retrieved_data:
        parts.append("\nInformation:")
        for i, chunk in enumerate(request.retrieved_data[:3], 1):
            text = chunk.get("text", "")
            source = chunk.get("source", "unknown")
            parts.append(f"[{i}] {text} (Source: {source})")
    
    # Add the question directly
    parts.append(f"\nQuestion: {request.query}")
    
    # Simple instruction
    parts.append("\nProvide a clear answer and cite sources as [1] when relevant:")
    
    return "\n".join(parts)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI app."""
    # Startup
    logger.info("Starting up Prompt Builder service...")
    yield
    # Shutdown
    logger.info("Shutting down Prompt Builder service...")

app = FastAPI(title="Prompt Builder Service", lifespan=lifespan)

@app.post("/build", response_model=PromptResponse)
async def build_prompt_endpoint(request: PromptRequest):
    """
    Build an optimal LLM prompt with SIMPLIFIED structure for better performance.
    
    This version removes complex formatting instructions that can confuse the LLM
    and cause premature termination or poor responses.
    """
    try:
        # Choose prompt style based on complexity
        if len(request.retrieved_data) <= 2 and len(request.query) < 100:
            # Use ultra-simple for basic queries
            final_prompt = build_ultra_simple_prompt(request)
            style_used = "ultra_simple"
        else:
            # Use simplified for complex queries
            final_prompt = build_simplified_prompt(request)
            style_used = "simplified"
        
        # Check token count and truncate if necessary
        token_count = count_tokens(final_prompt)
        if token_count > MAX_TOKENS:
            final_prompt = truncate_text(final_prompt, MAX_TOKENS)
            token_count = count_tokens(final_prompt)
            logger.warning(f"Prompt truncated to {token_count} tokens")
        
        # Calculate metadata
        metadata = {
            "num_sources": len(request.retrieved_data),
            "sources_used": list(set(r.get("source", "unknown") for r in request.retrieved_data)),
            "has_context": bool(request.context and request.context.strip()),
            "token_count": token_count,
            "style_used": style_used,
            "timestamp": datetime.utcnow().isoformat(),
            "prompt_length": len(final_prompt),
            "truncated": token_count >= MAX_TOKENS
        }
        
        logger.info(f"Built prompt: {style_used} style, {token_count} tokens, {len(request.retrieved_data)} sources")
        
        return PromptResponse(
            prompt=final_prompt,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error building prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/build/simple")
async def build_simple_prompt_endpoint(request: PromptRequest):
    """
    Build ULTRA SIMPLE prompt for debugging LLM issues.
    Use this endpoint when the regular /build is causing problems.
    """
    try:
        final_prompt = build_ultra_simple_prompt(request)
        token_count = count_tokens(final_prompt)
        
        metadata = {
            "style": "ultra_simple",
            "token_count": token_count,
            "sources_count": len(request.retrieved_data),
            "has_context": bool(request.context),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Built ultra-simple prompt: {token_count} tokens")
        
        return PromptResponse(
            prompt=final_prompt,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error building simple prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/build/debug")
async def build_debug_prompt_endpoint(request: PromptRequest):
    """
    Debug endpoint that returns both prompt styles for comparison.
    """
    try:
        simple_prompt = build_ultra_simple_prompt(request)
        regular_prompt = build_simplified_prompt(request)
        
        return {
            "ultra_simple": {
                "prompt": simple_prompt,
                "token_count": count_tokens(simple_prompt),
                "length": len(simple_prompt)
            },
            "simplified": {
                "prompt": regular_prompt,
                "token_count": count_tokens(regular_prompt),
                "length": len(regular_prompt)
            },
            "comparison": {
                "simple_tokens": count_tokens(simple_prompt),
                "regular_tokens": count_tokens(regular_prompt),
                "token_difference": count_tokens(regular_prompt) - count_tokens(simple_prompt)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/template")
async def apply_template_endpoint(request: PromptRequest):
    """Apply a custom template to the prompt data."""
    # Keep existing template functionality for backward compatibility
    if not hasattr(request, 'template') or not request.template:
        raise HTTPException(status_code=400, detail="Template is required for this endpoint")
    
    try:
        data = {
            "query": request.query,
            "context": request.context,
            "evidence": format_evidence_simple(request.retrieved_data)
        }
        
        prompt = request.template.format(**data)
        token_count = count_tokens(prompt)
        
        return PromptResponse(
            prompt=prompt,
            metadata={
                "style": "template",
                "token_count": token_count,
                "template_used": True
            }
        )
    
    except Exception as e:
        logger.error(f"Error applying template: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "prompt-builder",
        "version": "2.0.0-simplified"
    }

@app.get("/test")
async def test_prompt_styles():
    """Test endpoint to see different prompt styles."""
    sample_request = PromptRequest(
        query="What is blockchain?",
        context="User is asking about technology concepts",
        retrieved_data=[
            {
                "text": "Blockchain is a distributed ledger technology that provides immutable features",
                "source": "tech_docs.pdf",
                "score": 0.95,
                "type": "document",
                "metadata": {"page": 1}
            }
        ]
    )
    
    return {
        "ultra_simple": build_ultra_simple_prompt(sample_request),
        "simplified": build_simplified_prompt(sample_request),
        "token_comparison": {
            "ultra_simple": count_tokens(build_ultra_simple_prompt(sample_request)),
            "simplified": count_tokens(build_simplified_prompt(sample_request))
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)