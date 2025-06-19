import logging
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import re
import json
import html
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator
import markdown
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('response_formatter.log')
    ]
)
logger = logging.getLogger(__name__)

# Input/Output Models
class Citation(BaseModel):
    ref: str
    doc: str
    value: str

class FormatRequest(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    format: str = "markdown"

class FormatResponse(BaseModel):
    response: str
    citations: List[Citation]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime

# Citation parsing
def extract_citations(text: str) -> List[str]:
    """Extract citation references like [1], [2] from text."""
    return re.findall(r'\[(\d+)\]', text)

def map_citations_to_sources(citations: List[str], sources: List[Dict[str, Any]]) -> List[Citation]:
    """Map citation numbers to source documents."""
    result = []
    for i, source in enumerate(sources, 1):
        if str(i) in citations:
            result.append(Citation(
                ref=f"[{i}]",
                doc=source.get("source_type", "unknown"),
                value=source.get("content", "")[:100] + "..."  # Truncate long content
            ))
    return result

def sanitize_html(text: str) -> str:
    """Sanitize HTML content."""
    # First escape HTML special characters
    escaped = html.escape(text)
    # Then convert markdown to HTML
    html_content = markdown.markdown(escaped)
    # Use BeautifulSoup to clean any remaining unsafe tags
    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove any potentially dangerous tags
    for tag in soup.find_all(['script', 'style', 'iframe']):
        tag.decompose()
    return str(soup)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI app."""
    # Startup
    logger.info("Starting up Response Formatter service...")
    yield
    # Shutdown
    logger.info("Shutting down Response Formatter service...")

app = FastAPI(title="Response Formatter Service", lifespan=lifespan)

@app.post("/format", response_model=FormatResponse)
async def format_endpoint(request: FormatRequest):
    """
    Format LLM response with citations and metadata.
    
    Input:
    {
        "response": "In Jan, 130 items were returned [1].",
        "sources": [
            {
                "source_type": "sql",
                "content": "return stats Jan 2024",
                "metadata": {...}
            }
        ],
        "format": "markdown"
    }
    
    Output:
    {
        "answer": "In Jan, 130 items were returned [1].",
        "citations": [
            {
                "ref": "[1]",
                "doc": "sql",
                "value": "return stats Jan 2024"
            }
        ],
        "metadata": {
            "format": "markdown",
            "citation_count": 1
        }
    }
    """
    try:
        # Extract citations from response
        citations = extract_citations(request.response)
        
        # Map citations to sources
        citation_list = map_citations_to_sources(citations, request.sources)
        
        # Format response based on requested format
        if request.format == "markdown":
            formatted_response = request.response
        elif request.format == "json":
            # Validate JSON format
            try:
                json.loads(request.response)
                formatted_response = request.response
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON response")
        else:
            formatted_response = request.response
        
        return FormatResponse(
            response=formatted_response,
            citations=citation_list,
            metadata={
                "format": request.format,
                "citation_count": len(citations)
            }
        )
        
    except Exception as e:
        logger.error(f"Error formatting response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow()
    ) 