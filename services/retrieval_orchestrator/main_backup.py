from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import aiohttp
import asyncio
from datetime import datetime
import numpy as np
from collections import defaultdict
import logging
import sys
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('retrieval_orchestrator.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Retrieval Orchestrator Service")

# Configuration
VECTOR_RETRIEVER_URL = os.getenv("VECTOR_RETRIEVER_URL", "http://vector-service:8003")
SQL_RETRIEVER_URL = os.getenv("SQL_RETRIEVER_URL", "http://sql-retriever:8005")
WEB_RETRIEVER_URL = os.getenv("WEB_RETRIEVER_URL", "http://web-retriever:8006")
CONTEXT_MANAGER_URL = os.getenv("CONTEXT_MANAGER_URL", "http://context-manager:8001")
PROMPT_BUILDER_URL = os.getenv("PROMPT_BUILDER_URL", "http://prompt-builder:8007")

# Input/Output Models
class SearchRequest(BaseModel):
    query: str
    context: str
    filters: Optional[Dict[str, Any]] = None
    max_results: Optional[int] = 5

class SearchResult(BaseModel):
    text: str
    metadata: Dict[str, Any]
    score: float
    source: str
    timestamp: str

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_results: int
    processing_time: float
    prompt: Optional[str] = None

class PromptRequest(BaseModel):
    query: str
    context: str
    results: List[SearchResult]

class PromptResponse(BaseModel):
    prompt: str
    metadata: Dict[str, Any]

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using simple character overlap."""
    # Convert to sets of characters
    set1 = set(text1.lower())
    set2 = set(text2.lower())
    
    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0

def deduplicate_results(results: List[SearchResult], similarity_threshold: float = 0.8) -> List[SearchResult]:
    """Remove duplicate results based on text similarity."""
    if not results:
        return results
    
    # Sort by score in descending order
    sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
    unique_results = []
    seen_texts = set()
    
    for result in sorted_results:
        # Check if this result is too similar to any existing result
        is_duplicate = False
        for unique_result in unique_results:
            similarity = calculate_similarity(result.text, unique_result.text)
            if similarity > similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_results.append(result)
            seen_texts.add(result.text)
    
    return unique_results

def rank_results(results: List[SearchResult]) -> List[SearchResult]:
    """Rank results based on multiple factors."""
    if not results:
        return results
    
    # Calculate base scores
    for result in results:
        # Normalize score to 0-1 range
        result.score = min(max(result.score, 0), 1)
        
        # Apply source-specific boosts
        source_boost = {
            "vector": 1.0,  # Base boost
            "sql": 0.9,     # Slightly lower boost for SQL results
            "web": 0.8      # Lower boost for web results
        }.get(result.source, 1.0)
        
        # Apply recency boost
        try:
            result_time = datetime.fromisoformat(result.timestamp)
            age_hours = (datetime.utcnow() - result_time).total_seconds() / 3600
            recency_boost = 1.0 / (1.0 + age_hours / 24)  # Decay over 24 hours
        except:
            recency_boost = 1.0
        
        # Apply length penalty
        length_penalty = 1.0 / (1.0 + len(result.text) / 1000)  # Penalize very long texts
        
        # Combine all factors
        result.score = result.score * source_boost * recency_boost * length_penalty
    
    # Sort by final score
    return sorted(results, key=lambda x: x.score, reverse=True)

async def get_vector_results(query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
    """Get results from vector retriever."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{VECTOR_RETRIEVER_URL}/search",
            json={"query": query, "top_k": top_k, "filters": filters}
        ) as response:
            if response.status != 200:
                return []
            data = await response.json()
            return [
                SearchResult(
                    text=result["text"],
                    metadata=result["metadata"],
                    score=result["score"],
                    source="vector",
                    timestamp=datetime.utcnow().isoformat()
                )
                for result in data["results"]
            ]

async def get_sql_results(query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
    """Get results from SQL retriever."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{SQL_RETRIEVER_URL}/search",
            json={"query": query, "top_k": top_k, "filters": filters}
        ) as response:
            if response.status != 200:
                return []
            data = await response.json()
            return [
                SearchResult(
                    text=result["text"],
                    metadata=result["metadata"],
                    score=result["score"],
                    source="sql",
                    timestamp=datetime.utcnow().isoformat()
                )
                for result in data["results"]
            ]

async def get_web_results(query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
    """Get results from web retriever."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{WEB_RETRIEVER_URL}/search",
            json={"query": query, "top_k": top_k, "filters": filters}
        ) as response:
            if response.status != 200:
                return []
            data = await response.json()
            return [
                SearchResult(
                    text=result["text"],
                    metadata=result["metadata"],
                    score=result["score"],
                    source="web",
                    timestamp=datetime.utcnow().isoformat()
                )
                for result in data["results"]
            ]

async def build_prompt(query: str, context: str, results: List[SearchResult]) -> Optional[str]:
    """Build a prompt using the Prompt Builder service."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{PROMPT_BUILDER_URL}/build",
                json=PromptRequest(
                    query=query,
                    context=context,
                    results=results
                ).dict()
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to build prompt: {response.status}")
                    return None
                
                data = await response.json()
                return data.get("prompt")
    except Exception as e:
        logger.error(f"Error building prompt: {str(e)}")
        return None

@app.post("/search")
async def search(request: SearchRequest):
    """
    Coordinate parallel multi-source retrieval and result unification.
    
    Input:
    {
        "query": "What's our refund policy?",
        "context": "Previous conversation context...",
        "filters": {
            "source_types": ["document", "web"],
            "date_range": {"start": "2024-01-01", "end": "2024-12-31"}
        },
        "max_results": 5
    }
    
    Output:
    {
        "results": [
            {
                "type": "vector",
                "content": "Refund policy document...",
                "score": 0.95,
                "metadata": {
                    "source": "policy.pdf",
                    "page": 1
                }
            }
        ],
        "sources_used": ["vector", "web"],
        "confidence_score": 0.92
    }
    """
    try:
        logger.info(f"Received search request: {request.query}")
        start_time = time.time()
        
        # Prepare search tasks for each source
        search_tasks = []
        
        # Vector search task
        if not request.filters or "vector" in request.filters.get("source_types", ["vector"]):
            search_tasks.append(
                vector_search(
                    query=request.query,
                    context=request.context,
                    max_results=request.max_results
                )
            )
        
        # SQL search task
        if not request.filters or "sql" in request.filters.get("source_types", []):
            search_tasks.append(
                sql_search(
                    query=request.query,
                    context=request.context,
                    max_results=request.max_results
                )
            )
        
        # Web search task
        if not request.filters or "web" in request.filters.get("source_types", []):
            search_tasks.append(
                web_search(
                    query=request.query,
                    context=request.context,
                    max_results=request.max_results
                )
            )
        
        # Execute all searches in parallel
        logger.info(f"Executing {len(search_tasks)} parallel searches")
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process results and handle any errors
        all_results = []
        sources_used = set()
        
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                logger.error(f"Search task {i} failed: {str(result)}")
                continue
            
            if result and isinstance(result, list):
                all_results.extend(result)
                if result:
                    sources_used.add(result[0].get("type"))
        
        # Deduplicate results
        logger.info("Deduplicating results")
        unique_results = deduplicate_results(all_results)
        
        # Rank results
        logger.info("Ranking results")
        ranked_results = rank_results(unique_results)
        
        # Apply filters
        if request.filters:
            logger.info("Applying filters")
            ranked_results = apply_filters(ranked_results, request.filters)
        
        # Take top-k results
        final_results = ranked_results[:request.max_results]
        
        # Calculate confidence score
        confidence_score = calculate_confidence_score(final_results)
        
        processing_time = time.time() - start_time
        logger.info(f"Search completed in {processing_time:.2f}s")
        
        return {
            "results": final_results,
            "sources_used": list(sources_used),
            "confidence_score": confidence_score,
            "metadata": {
                "processing_time": processing_time,
                "total_results": len(ranked_results),
                "filtered_results": len(final_results)
            }
        }
    
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def vector_search(query: str, context: str, max_results: int) -> List[Dict]:
    """Perform vector similarity search."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{VECTOR_RETRIEVER_URL}/search",
                json={
                    "query": query,
                    "context": context,
                    "max_results": max_results
                }
            ) as response:
                if response.status != 200:
                    logger.error(f"Vector search failed: {response.status}")
                    return []
                data = await response.json()
                return data.get("results", [])
    except Exception as e:
        logger.error(f"Vector search error: {str(e)}")
        return []

async def sql_search(query: str, context: str, max_results: int) -> List[Dict]:
    """Perform SQL-based search."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{SQL_RETRIEVER_URL}/search",
                json={
                    "query": query,
                    "context": context,
                    "max_results": max_results
                }
            ) as response:
                if response.status != 200:
                    logger.error(f"SQL search failed: {response.status}")
                    return []
                data = await response.json()
                return data.get("results", [])
    except Exception as e:
        logger.error(f"SQL search error: {str(e)}")
        return []

async def web_search(query: str, context: str, max_results: int) -> List[Dict]:
    """Perform web search."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{WEB_RETRIEVER_URL}/search",
                json={
                    "query": query,
                    "context": context,
                    "max_results": max_results
                }
            ) as response:
                if response.status != 200:
                    logger.error(f"Web search failed: {response.status}")
                    return []
                data = await response.json()
                return data.get("results", [])
    except Exception as e:
        logger.error(f"Web search error: {str(e)}")
        return []

def apply_filters(results: List[Dict], filters: Dict) -> List[Dict]:
    """Apply filters to results."""
    if not filters:
        return results
    
    filtered_results = results
    
    # Filter by date range
    if "date_range" in filters:
        start_date = datetime.fromisoformat(filters["date_range"]["start"])
        end_date = datetime.fromisoformat(filters["date_range"]["end"])
        
        filtered_results = [
            r for r in filtered_results
            if "metadata" in r and "timestamp" in r["metadata"]
            and start_date <= datetime.fromisoformat(r["metadata"]["timestamp"]) <= end_date
        ]
    
    return filtered_results

def calculate_confidence_score(results: List[Dict]) -> float:
    """Calculate overall confidence score for results."""
    if not results:
        return 0.0
    
    # Weight factors
    score_weight = 0.6
    source_diversity_weight = 0.2
    result_count_weight = 0.2
    
    # Average of result scores
    avg_score = sum(r.get("score", 0) for r in results) / len(results)
    
    # Source diversity (number of unique sources)
    unique_sources = len(set(r.get("type", "") for r in results))
    source_diversity = min(unique_sources / 3, 1.0)  # Normalize to 0-1
    
    # Result count factor
    result_count = min(len(results) / 5, 1.0)  # Normalize to 0-1
    
    # Calculate final confidence score
    confidence_score = (
        avg_score * score_weight +
        source_diversity * source_diversity_weight +
        result_count * result_count_weight
    )
    
    return min(confidence_score, 1.0)  # Ensure score is between 0 and 1

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        logger.info("Performing health check")
        services = {
            "vector_retriever": VECTOR_RETRIEVER_URL,
            "sql_retriever": SQL_RETRIEVER_URL,
            "web_retriever": WEB_RETRIEVER_URL
        }
        
        health_status = {}
        async with aiohttp.ClientSession() as session:
            for service, url in services.items():
                try:
                    logger.info(f"Checking health of {service}")
                    async with session.get(f"{url}/health") as response:
                        if response.status == 200:
                            health_status[service] = "healthy"
                            logger.info(f"{service} is healthy")
                        else:
                            health_status[service] = "unhealthy"
                            logger.error(f"{service} returned status {response.status}")
                except Exception as e:
                    health_status[service] = "unreachable"
                    logger.error(f"Error checking {service}: {str(e)}")
        
        return {
            "status": "healthy" if all(v == "healthy" for v in health_status.values()) else "unhealthy",
            "services": health_status
        }
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 