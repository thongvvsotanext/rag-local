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

def format_evidence(evidence: List[EvidenceChunk]) -> str:
    """Format evidence chunks with citations."""
    # Sort evidence by score in descending order
    sorted_evidence = sorted(evidence, key=lambda x: x.score, reverse=True)
    # Take top 3-5 chunks as per spec
    top_evidence = sorted_evidence[:min(5, len(sorted_evidence))]
    
    formatted = []
    for i, chunk in enumerate(top_evidence, 1):
        source = chunk.source
        if chunk.type == "web":
            source = f"web: {source}"
        elif chunk.type == "sql":
            source = f"database: {source}"
        
        formatted.append(f"[{i}] {chunk.text} (Source: {source})")
    return "\n".join(formatted)

def build_prompt(request: PromptRequest) -> str:
    """Build the complete prompt with query, context, and evidence."""
    # Start with system prompt
    prompt_parts = [f"System: {SYSTEM_PROMPT}\n\n"]
    
    # Add context if available
    if request.context:
        prompt_parts.append(f"Context Summary: {request.context}\n\n")
    
    # Add evidence
    if request.retrieved_data:
        evidence_text = format_evidence([EvidenceChunk(text=d['text'], source=d['source'], score=d['score'], type=d['type'], metadata=d['metadata']) for d in request.retrieved_data])
        prompt_parts.append(f"Evidence:\n{evidence_text}\n\n")
    
    # Add query
    prompt_parts.append(f"Query: {request.query}\n\n")
    
    # Add format instructions
    prompt_parts.append("Please provide your response in markdown format with citations like [1], [2].\n\n")
    
    # Combine all parts
    prompt = "\n".join(prompt_parts)
    
    # Truncate if necessary
    if count_tokens(prompt) > request.max_tokens:
        prompt = truncate_text(prompt, request.max_tokens)
    
    return prompt

def apply_template(template: str, data: Dict[str, Any]) -> str:
    """Apply a custom template with the provided data."""
    try:
        return template.format(**data)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing template variable: {str(e)}")

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
    Build an optimal LLM prompt that blends query + top evidence + instructions.
    
    Input:
    {
        "query": "What's our refund policy?",
        "context": "Previous conversation context...",
        "retrieved_data": [
            {
                "text": "Returns must be made within 30 days",
                "metadata": {"source": "policy.pdf", "page": 1},
                "score": 0.95,
                "source": "vector",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        ],
        "filters": {
            "source_types": ["document", "web"],
            "date_range": {"start": "2024-01-01", "end": "2024-12-31"}
        }
    }
    
    Output:
    {
        "prompt": "You are a helpful enterprise assistant...",
        "metadata": {
            "num_sources": 3,
            "total_tokens": 150,
            "sources_used": ["vector", "web"]
        }
    }
    """
    try:
        # 1. Start with system role
        prompt_parts = ["You are a helpful enterprise assistant."]
        
        # 2. Add context if exists
        if request.context:
            prompt_parts.append(f"\nContext from previous conversation:\n{request.context}")
        
        # 3. Add top evidence with citations
        if request.retrieved_data:
            prompt_parts.append("\nRelevant information:")
            for i, result in enumerate(request.retrieved_data, 1):
                source = result.get("source", "unknown")
                text = result.get("text", "")
                metadata = result.get("metadata", {})
                prompt_parts.append(f"[{i}] {text} (Source: {source}, {metadata})")
        
        # 4. Add user query
        prompt_parts.append(f"\nUser query: {request.query}")
        
        # 5. Add formatting instructions
        prompt_parts.append("\nInstructions:")
        prompt_parts.append("1. Reply in markdown format")
        prompt_parts.append("2. Include citations like [1] when referencing information")
        prompt_parts.append("3. If you're unsure about something, say so")
        prompt_parts.append("4. Keep the response concise and focused")
        
        # Combine all parts
        final_prompt = "\n".join(prompt_parts)
        
        # Calculate metadata
        metadata = {
            "num_sources": len(request.retrieved_data),
            "sources_used": list(set(r.get("source", "unknown") for r in request.retrieved_data)),
            "has_context": bool(request.context),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return PromptResponse(
            prompt=final_prompt,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error building prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/template")
async def apply_template_endpoint(request: PromptRequest):
    """Apply a custom template to the prompt data."""
    if not request.template:
        raise HTTPException(status_code=400, detail="Template is required")
    
    try:
        data = {
            "query": request.query,
            "context": request.context,
            "evidence": format_evidence(request.evidence),
            "format": request.output_format
        }
        
        prompt = apply_template(request.template, data)
        token_count = count_tokens(prompt)
        
        return PromptResponse(
            prompt=prompt,
            token_count=token_count,
            evidence_used=[
                {
                    "text": chunk.text,
                    "source": chunk.source,
                    "type": chunk.type,
                    "score": chunk.score,
                    "metadata": chunk.metadata
                }
                for chunk in request.evidence
            ],
            format=request.output_format
        )
    
    except Exception as e:
        logger.error(f"Error applying template: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 