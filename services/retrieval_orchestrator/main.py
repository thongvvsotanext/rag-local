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
import traceback
import json
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
file_handler = logging.FileHandler('retrieval_orchestrator.log')
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

app = FastAPI(title="Retrieval Orchestrator Service")

# Configuration
VECTOR_RETRIEVER_URL = os.getenv("VECTOR_RETRIEVER_URL", "http://vector-service:8003")
SQL_RETRIEVER_URL = os.getenv("SQL_RETRIEVER_URL", "http://sql-retriever:8005")
WEB_RETRIEVER_URL = os.getenv("WEB_RETRIEVER_URL", "http://web-retriever:8006")
CONTEXT_MANAGER_URL = os.getenv("CONTEXT_MANAGER_URL", "http://context-manager:8001")
PROMPT_BUILDER_URL = os.getenv("PROMPT_BUILDER_URL", "http://prompt-builder:8007")

# Log configuration on startup
logger.info("ORCHESTRATOR_STARTUP - Retrieval Orchestrator Service starting", extra={
    'config': {
        'vector_retriever_url': VECTOR_RETRIEVER_URL,
        'sql_retriever_url': SQL_RETRIEVER_URL,
        'web_retriever_url': WEB_RETRIEVER_URL,
        'context_manager_url': CONTEXT_MANAGER_URL,
        'prompt_builder_url': PROMPT_BUILDER_URL
    }
})

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
    try:
        # Convert to sets of characters
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        
        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity
    except Exception as e:
        logger.warning(f"SIMILARITY_CALCULATION_ERROR - Error calculating similarity: {str(e)}")
        return 0

def deduplicate_results(results: List[SearchResult], similarity_threshold: float = 0.8, request_id: str = "N/A") -> List[SearchResult]:
    """Remove duplicate results based on text similarity."""
    start_time = time.time()
    
    try:
        log_with_context(
            logger, 'info',
            f"DEDUPLICATION_START - Starting result deduplication",
            request_id, "deduplicate",
            input_count=len(results),
            similarity_threshold=similarity_threshold
        )
        
        if not results:
            return results
        
        # Sort by score in descending order
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        unique_results = []
        seen_texts = set()
        duplicates_removed = 0
        
        for i, result in enumerate(sorted_results):
            # Check if this result is too similar to any existing result
            is_duplicate = False
            for j, unique_result in enumerate(unique_results):
                similarity = calculate_similarity(result.text, unique_result.text)
                if similarity > similarity_threshold:
                    is_duplicate = True
                    duplicates_removed += 1
                    log_with_context(
                        logger, 'debug',
                        f"DEDUPLICATION_DUPLICATE - Found duplicate result",
                        request_id, "deduplicate",
                        duplicate_index=i,
                        similar_to_index=j,
                        similarity_score=round(similarity, 3),
                        threshold=similarity_threshold
                    )
                    break
            
            if not is_duplicate:
                unique_results.append(result)
                seen_texts.add(result.text)
        
        processing_time = time.time() - start_time
        
        log_with_context(
            logger, 'info',
            f"DEDUPLICATION_SUCCESS - Deduplication completed",
            request_id, "deduplicate",
            input_count=len(results),
            output_count=len(unique_results),
            duplicates_removed=duplicates_removed,
            processing_time_seconds=round(processing_time, 3),
            similarity_threshold=similarity_threshold
        )
        
        return unique_results
        
    except Exception as e:
        processing_time = time.time() - start_time
        log_with_context(
            logger, 'error',
            f"DEDUPLICATION_ERROR - Deduplication failed",
            request_id, "deduplicate",
            error=str(e),
            traceback=traceback.format_exc(),
            input_count=len(results),
            processing_time_seconds=round(processing_time, 3)
        )
        # Return original results on error
        return results

def rank_results(results: List[SearchResult], request_id: str = "N/A") -> List[SearchResult]:
    """Rank results based on multiple factors."""
    start_time = time.time()
    
    try:
        log_with_context(
            logger, 'info',
            f"RANKING_START - Starting result ranking",
            request_id, "rank",
            input_count=len(results)
        )
        
        if not results:
            return results
        
        original_scores = [r.score for r in results]
        
        # Calculate base scores
        for i, result in enumerate(results):
            original_score = result.score
            
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
            final_score = result.score * source_boost * recency_boost * length_penalty
            
            log_with_context(
                logger, 'debug',
                f"RANKING_CALCULATION - Calculated ranking score",
                request_id, "rank",
                result_index=i,
                original_score=round(original_score, 3),
                normalized_score=round(result.score, 3),
                source_boost=source_boost,
                recency_boost=round(recency_boost, 3),
                length_penalty=round(length_penalty, 3),
                final_score=round(final_score, 3),
                text_length=len(result.text),
                source=result.source
            )
            
            result.score = final_score
        
        # Sort by final score
        ranked_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        processing_time = time.time() - start_time
        
        log_with_context(
            logger, 'info',
            f"RANKING_SUCCESS - Result ranking completed",
            request_id, "rank",
            input_count=len(results),
            output_count=len(ranked_results),
            processing_time_seconds=round(processing_time, 3),
            score_stats={
                'original_avg': round(sum(original_scores) / len(original_scores), 3),
                'final_avg': round(sum(r.score for r in ranked_results) / len(ranked_results), 3),
                'top_score': round(ranked_results[0].score, 3) if ranked_results else 0,
                'bottom_score': round(ranked_results[-1].score, 3) if ranked_results else 0
            }
        )
        
        return ranked_results
        
    except Exception as e:
        processing_time = time.time() - start_time
        log_with_context(
            logger, 'error',
            f"RANKING_ERROR - Result ranking failed",
            request_id, "rank",
            error=str(e),
            traceback=traceback.format_exc(),
            input_count=len(results),
            processing_time_seconds=round(processing_time, 3)
        )
        # Return original results on error
        return results

async def get_vector_results(query: str, top_k: int, filters: Optional[Dict[str, Any]] = None, request_id: str = "N/A") -> List[SearchResult]:
    """Get results from vector retriever."""
    start_time = time.time()
    
    try:
        log_with_context(
            logger, 'info',
            f"VECTOR_SEARCH_START - Starting vector retriever call",
            request_id, "vector_search",
            query=query,
            top_k=top_k,
            filters=filters,
            url=VECTOR_RETRIEVER_URL
        )
        
        async with aiohttp.ClientSession() as session:
            # Update request body to match vector service's DocumentSearchRequest
            request_body = {
                "query": query,
                "top_k": top_k
            }
            
            async with session.post(
                f"{VECTOR_RETRIEVER_URL}/search",
                json=request_body
            ) as response:
                response_time = time.time() - start_time
                
                if response.status != 200:
                    log_with_context(
                        logger, 'error',
                        f"VECTOR_SEARCH_ERROR - Vector retriever returned error status",
                        request_id, "vector_search",
                        status_code=response.status,
                        response_time_seconds=round(response_time, 3),
                        url=VECTOR_RETRIEVER_URL
                    )
                    return []
                
                # The vector service returns List[DocumentSearchResult] directly, not wrapped in {"results": [...]}
                data = await response.json()
                
                # Map DocumentSearchResult fields to SearchResult fields
                results = [
                    SearchResult(
                        text=result["text"],
                        metadata={
                            "chunk_id": result["chunk_id"],
                            "page": result["page"],
                            "source_type": "document"
                        },
                        score=result["score"],
                        source="vector",
                        timestamp=datetime.utcnow().isoformat()
                    )
                    for result in data  # data is already a list, not {"results": [...]}
                ]
                
                log_with_context(
                    logger, 'info',
                    f"VECTOR_SEARCH_SUCCESS - Vector retriever call completed",
                    request_id, "vector_search",
                    results_count=len(results),
                    response_time_seconds=round(response_time, 3),
                    scores=[round(r.score, 3) for r in results[:5]]  # Top 5 scores
                )
                
                return results
                
    except Exception as e:
        response_time = time.time() - start_time
        log_with_context(
            logger, 'error',
            f"VECTOR_SEARCH_ERROR - Vector retriever call failed",
            request_id, "vector_search",
            error=str(e),
            traceback=traceback.format_exc(),
            response_time_seconds=round(response_time, 3),
            url=VECTOR_RETRIEVER_URL
        )
        return []

async def get_sql_results(query: str, top_k: int, filters: Optional[Dict[str, Any]] = None, request_id: str = "N/A") -> List[SearchResult]:
    """Get results from SQL retriever."""
    start_time = time.time()
    
    try:
        log_with_context(
            logger, 'info',
            f"SQL_SEARCH_START - Starting SQL retriever call",
            request_id, "sql_search",
            query=query,
            top_k=top_k,
            filters=filters,
            url=SQL_RETRIEVER_URL
        )
        
        # Prepare request body to match SQL service's SQLQueryRequest model
        request_body = {
            "query": query,
            "max_rows": top_k,  # SQL service uses max_rows instead of top_k
            "timeout": 30  # Default timeout
        }
        
        # Add timeout from filters if available
        if filters and "timeout" in filters:
            request_body["timeout"] = filters["timeout"]
        
        async with aiohttp.ClientSession() as session:
            # Note: SQL service uses /query endpoint, not /search
            async with session.post(
                f"{SQL_RETRIEVER_URL}/query",
                json=request_body
            ) as response:
                response_time = time.time() - start_time
                
                if response.status != 200:
                    log_with_context(
                        logger, 'error',
                        f"SQL_SEARCH_ERROR - SQL retriever returned error status",
                        request_id, "sql_search",
                        status_code=response.status,
                        response_time_seconds=round(response_time, 3),
                        url=SQL_RETRIEVER_URL
                    )
                    return []
                
                data = await response.json()
                
                # Map SQLQueryResponse fields to SearchResult fields
                # SQL service returns: {
                #   "results": List[Dict[str, Any]], 
                #   "sql_generated": str,
                #   "execution_time": float,
                #   "row_count": int,
                #   "schema_info": Dict[str, Any]
                # }
                
                results = []
                sql_results = data.get("results", [])
                
                # Convert each database row to a SearchResult
                for i, row in enumerate(sql_results):
                    # Convert row data to a readable text format
                    text_parts = []
                    for key, value in row.items():
                        if value is not None:
                            text_parts.append(f"{key}: {value}")
                    
                    result_text = " | ".join(text_parts) if text_parts else "No data"
                    
                    # Calculate a simple relevance score based on position
                    # (SQL results are typically ordered by relevance via the generated query)
                    score = max(0.1, 1.0 - (i * 0.1))  # Decreasing score: 1.0, 0.9, 0.8, etc.
                    
                    results.append(SearchResult(
                        text=result_text,
                        metadata={
                            "sql_generated": data.get("sql_generated", ""),
                            "execution_time": data.get("execution_time", 0),
                            "row_count": data.get("row_count", 0),
                            "row_index": i,
                            "row_data": row,
                            "source_type": "database"
                        },
                        score=score,
                        source="sql",
                        timestamp=datetime.utcnow().isoformat()
                    ))
                
                log_with_context(
                    logger, 'info',
                    f"SQL_SEARCH_SUCCESS - SQL retriever call completed",
                    request_id, "sql_search",
                    results_count=len(results),
                    response_time_seconds=round(response_time, 3),
                    sql_generated=data.get("sql_generated", ""),
                    execution_time_seconds=data.get("execution_time", 0),
                    row_count=data.get("row_count", 0),
                    scores=[round(r.score, 3) for r in results[:5]]  # Top 5 scores
                )
                
                return results
                
    except Exception as e:
        response_time = time.time() - start_time
        log_with_context(
            logger, 'error',
            f"SQL_SEARCH_ERROR - SQL retriever call failed",
            request_id, "sql_search",
            error=str(e),
            traceback=traceback.format_exc(),
            response_time_seconds=round(response_time, 3),
            url=SQL_RETRIEVER_URL
        )
        return []

async def get_web_results(query: str, top_k: int, filters: Optional[Dict[str, Any]] = None, request_id: str = "N/A") -> List[SearchResult]:
    """Get results from web retriever."""
    start_time = time.time()
    
    try:
        log_with_context(
            logger, 'info',
            f"WEB_SEARCH_START - Starting web retriever call",
            request_id, "web_search",
            query=query,
            top_k=top_k,
            filters=filters,
            url=WEB_RETRIEVER_URL
        )
        
        # Prepare request body to match web service's SearchRequest model
        request_body = {
            "query": query,
            "max_results": top_k
        }
        
        # Add source_types from filters if available
        if filters and "source_types" in filters:
            request_body["source_types"] = filters["source_types"]
        else:
            # Default source types for web search
            request_body["source_types"] = ["news", "official"]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{WEB_RETRIEVER_URL}/search",
                json=request_body
            ) as response:
                response_time = time.time() - start_time
                
                if response.status != 200:
                    log_with_context(
                        logger, 'error',
                        f"WEB_SEARCH_ERROR - Web retriever returned error status",
                        request_id, "web_search",
                        status_code=response.status,
                        response_time_seconds=round(response_time, 3),
                        url=WEB_RETRIEVER_URL
                    )
                    return []
                
                data = await response.json()
                
                # Map SearchResponse fields to SearchResult fields
                # Web service returns: {"results": [SearchResult...]}
                # Where each SearchResult has: url, title, summary, domain, content_type, crawled_at, relevance_score
                results = [
                    SearchResult(
                        text=result["summary"],  # Use summary as the main text content
                        metadata={
                            "url": result["url"],
                            "title": result["title"],
                            "domain": result["domain"],
                            "content_type": result["content_type"],
                            "crawled_at": result["crawled_at"],
                            "source_type": "web"
                        },
                        score=result["relevance_score"],
                        source="web",
                        timestamp=datetime.utcnow().isoformat()
                    )
                    for result in data.get("results", [])
                ]
                
                log_with_context(
                    logger, 'info',
                    f"WEB_SEARCH_SUCCESS - Web retriever call completed",
                    request_id, "web_search",
                    results_count=len(results),
                    response_time_seconds=round(response_time, 3),
                    scores=[round(r.score, 3) for r in results[:5]],  # Top 5 scores
                    domains=list(set([r.metadata.get("domain", "") for r in results]))
                )
                
                return results
                
    except Exception as e:
        response_time = time.time() - start_time
        log_with_context(
            logger, 'error',
            f"WEB_SEARCH_ERROR - Web retriever call failed",
            request_id, "web_search",
            error=str(e),
            traceback=traceback.format_exc(),
            response_time_seconds=round(response_time, 3),
            url=WEB_RETRIEVER_URL
        )
        return []

async def build_prompt(query: str, context: str, results: List[SearchResult], request_id: str = "N/A") -> Optional[str]:
    """Build a prompt using the Prompt Builder service."""
    start_time = time.time()
    
    try:
        log_with_context(
            logger, 'info',
            f"PROMPT_BUILD_START - Starting prompt builder call",
            request_id, "build_prompt",
            query=query,
            context_length=len(context),
            results_count=len(results),
            url=PROMPT_BUILDER_URL
        )
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{PROMPT_BUILDER_URL}/build",
                json=PromptRequest(
                    query=query,
                    context=context,
                    results=results
                ).dict()
            ) as response:
                response_time = time.time() - start_time
                
                if response.status != 200:
                    log_with_context(
                        logger, 'error',
                        f"PROMPT_BUILD_ERROR - Prompt builder returned error status",
                        request_id, "build_prompt",
                        status_code=response.status,
                        response_time_seconds=round(response_time, 3),
                        url=PROMPT_BUILDER_URL
                    )
                    return None
                
                data = await response.json()
                prompt = data.get("prompt")
                
                log_with_context(
                    logger, 'info',
                    f"PROMPT_BUILD_SUCCESS - Prompt builder call completed",
                    request_id, "build_prompt",
                    prompt_length=len(prompt) if prompt else 0,
                    response_time_seconds=round(response_time, 3)
                )
                
                return prompt
                
    except Exception as e:
        response_time = time.time() - start_time
        log_with_context(
            logger, 'error',
            f"PROMPT_BUILD_ERROR - Prompt builder call failed",
            request_id, "build_prompt",
            error=str(e),
            traceback=traceback.format_exc(),
            response_time_seconds=round(response_time, 3),
            url=PROMPT_BUILDER_URL
        )
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
    request_id = str(uuid.uuid4())
    endpoint_logger = get_contextual_logger(request_id, "/search")
    start_time = time.time()
    
    # Initialize processing steps tracking
    processing_steps = []
    
    try:
        # Log input parameters
        log_with_context(
            logger, 'info',
            f"ORCHESTRATOR_SEARCH_START - Starting coordinated search",
            request_id, "/search",
            input_params={
                'query': request.query,
                'context_length': len(request.context),
                'filters': request.filters,
                'max_results': request.max_results
            }
        )
        
        processing_steps.append(f"Started search at {datetime.utcnow().isoformat()}")
        
        # Determine which sources to search
        source_types = request.filters.get("source_types", ["vector", "sql", "web"]) if request.filters else ["vector", "sql", "web"]
        
        log_with_context(
            logger, 'info',
            f"ORCHESTRATOR_SEARCH_STEP - Preparing parallel searches",
            request_id, "/search",
            enabled_sources=source_types,
            total_sources=len(source_types)
        )
        
        # Prepare search tasks for each source
        search_tasks = []
        task_names = []
        
        # Vector search task
        if "vector" in source_types:
            search_tasks.append(
                get_vector_results(
                    query=request.query,
                    top_k=request.max_results,
                    filters=request.filters,
                    request_id=request_id
                )
            )
            task_names.append("vector")
        
        # SQL search task
        if "sql" in source_types:
            search_tasks.append(
                get_sql_results(
                    query=request.query,
                    top_k=request.max_results,
                    filters=request.filters,
                    request_id=request_id
                )
            )
            task_names.append("sql")
        
        # Web search task
        if "web" in source_types:
            search_tasks.append(
                get_web_results(
                    query=request.query,
                    top_k=request.max_results,
                    filters=request.filters,
                    request_id=request_id
                )
            )
            task_names.append("web")
        
        processing_steps.append(f"Prepared {len(search_tasks)} parallel search tasks")
        
        # Execute all searches in parallel
        log_with_context(
            logger, 'info',
            f"ORCHESTRATOR_SEARCH_STEP - Executing parallel searches",
            request_id, "/search",
            task_count=len(search_tasks),
            task_names=task_names
        )
        
        parallel_start_time = time.time()
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        parallel_time = time.time() - parallel_start_time
        
        log_with_context(
            logger, 'info',
            f"ORCHESTRATOR_SEARCH_STEP - Parallel searches completed",
            request_id, "/search",
            parallel_execution_time_seconds=round(parallel_time, 3),
            task_count=len(search_tasks)
        )
        
        # Process results and handle any errors
        processing_start_time = time.time()
        all_results = []
        sources_used = set()
        source_stats = {}
        
        for i, result in enumerate(search_results):
            task_name = task_names[i] if i < len(task_names) else f"task_{i}"
            
            if isinstance(result, Exception):
                log_with_context(
                    logger, 'error',
                    f"ORCHESTRATOR_SEARCH_ERROR - Search task failed",
                    request_id, "/search",
                    task_name=task_name,
                    task_index=i,
                    error=str(result),
                    traceback=traceback.format_exc()
                )
                source_stats[task_name] = {'status': 'failed', 'count': 0, 'error': str(result)}
                continue
            
            if result and isinstance(result, list):
                all_results.extend(result)
                sources_used.add(task_name)
                source_stats[task_name] = {'status': 'success', 'count': len(result)}
                
                log_with_context(
                    logger, 'info',
                    f"ORCHESTRATOR_SEARCH_STEP - Task results processed",
                    request_id, "/search",
                    task_name=task_name,
                    results_count=len(result),
                    scores=[round(r.score, 3) for r in result[:3]] if result else []
                )
            else:
                source_stats[task_name] = {'status': 'no_results', 'count': 0}
        
        processing_steps.append(f"Processed {len(all_results)} total results from {len(sources_used)} sources")
        
        # Deduplicate results
        log_with_context(
            logger, 'info',
            f"ORCHESTRATOR_SEARCH_STEP - Starting deduplication",
            request_id, "/search",
            input_count=len(all_results)
        )
        
        dedup_start_time = time.time()
        unique_results = deduplicate_results(all_results, request_id=request_id)
        dedup_time = time.time() - dedup_start_time
        processing_steps.append(f"Deduplicated {len(all_results)} -> {len(unique_results)} results")
        
        # Rank results
        log_with_context(
            logger, 'info',
            f"ORCHESTRATOR_SEARCH_STEP - Starting ranking",
            request_id, "/search",
            input_count=len(unique_results)
        )
        
        rank_start_time = time.time()
        ranked_results = rank_results(unique_results, request_id=request_id)
        rank_time = time.time() - rank_start_time
        processing_steps.append(f"Ranked {len(unique_results)} results")
        
        # Apply filters
        filter_start_time = time.time()
        if request.filters:
            log_with_context(
                logger, 'info',
                f"ORCHESTRATOR_SEARCH_STEP - Applying filters",
                request_id, "/search",
                filters=request.filters,
                input_count=len(ranked_results)
            )
            ranked_results = apply_filters(ranked_results, request.filters, request_id)
            processing_steps.append(f"Applied filters, {len(ranked_results)} results remain")
        filter_time = time.time() - filter_start_time
        
        # Take top-k results
        final_results = ranked_results[:request.max_results]
        processing_steps.append(f"Selected top {len(final_results)} results")
        
        # Calculate confidence score
        confidence_start_time = time.time()
        confidence_score = calculate_confidence_score(final_results, request_id)
        confidence_time = time.time() - confidence_start_time
        
        processing_time = time.time() - start_time
        process_total_time = time.time() - processing_start_time
        
        # Prepare response
        response_data = {
            "results": [result.dict() for result in final_results],
            "sources_used": list(sources_used),
            "confidence_score": confidence_score,
            "metadata": {
                "processing_time": processing_time,
                "total_results_found": len(all_results),
                "unique_results": len(unique_results),
                "filtered_results": len(final_results),
                "source_stats": source_stats,
                "timing_breakdown": {
                    "parallel_search_seconds": round(parallel_time, 3),
                    "deduplication_seconds": round(dedup_time, 3),
                    "ranking_seconds": round(rank_time, 3),
                    "filtering_seconds": round(filter_time, 3),
                    "confidence_calculation_seconds": round(confidence_time, 3),
                    "total_processing_seconds": round(process_total_time, 3)
                }
            }
        }
        
        # Log successful completion
        log_with_context(
            logger, 'info',
            f"ORCHESTRATOR_SEARCH_SUCCESS - Coordinated search completed successfully",
            request_id, "/search",
            query=request.query,
            sources_used=list(sources_used),
            total_results_found=len(all_results),
            unique_results=len(unique_results),
            final_results=len(final_results),
            confidence_score=round(confidence_score, 3),
            processing_time_seconds=round(processing_time, 3),
            processing_steps=processing_steps,
            timing_breakdown=response_data["metadata"]["timing_breakdown"],
            source_stats=source_stats
        )
        
        return response_data
    
    except Exception as e:
        processing_time = time.time() - start_time
        processing_steps.append(f"FATAL ERROR: {str(e)}")
        
        log_with_context(
            logger, 'error',
            f"ORCHESTRATOR_SEARCH_ERROR - Coordinated search failed",
            request_id, "/search",
            error=str(e),
            traceback=traceback.format_exc(),
            query=request.query,
            processing_time_seconds=round(processing_time, 3),
            processing_steps=processing_steps
        )
        
        raise HTTPException(status_code=500, detail=str(e))

async def vector_search(query: str, context: str, max_results: int, request_id: str = "N/A") -> List[Dict]:
    """Perform vector similarity search."""
    start_time = time.time()
    
    try:
        log_with_context(
            logger, 'info',
            f"LEGACY_VECTOR_SEARCH_START - Starting legacy vector search",
            request_id, "legacy_vector_search",
            query=query,
            context_length=len(context),
            max_results=max_results
        )
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{VECTOR_RETRIEVER_URL}/search",
                json={
                    "query": query,
                    "context": context,
                    "max_results": max_results
                }
            ) as response:
                response_time = time.time() - start_time
                
                if response.status != 200:
                    log_with_context(
                        logger, 'error',
                        f"LEGACY_VECTOR_SEARCH_ERROR - Vector search failed",
                        request_id, "legacy_vector_search",
                        status_code=response.status,
                        response_time_seconds=round(response_time, 3)
                    )
                    return []
                
                data = await response.json()
                results = data.get("results", [])
                
                log_with_context(
                    logger, 'info',
                    f"LEGACY_VECTOR_SEARCH_SUCCESS - Vector search completed",
                    request_id, "legacy_vector_search",
                    results_count=len(results),
                    response_time_seconds=round(response_time, 3)
                )
                
                return results
                
    except Exception as e:
        response_time = time.time() - start_time
        log_with_context(
            logger, 'error',
            f"LEGACY_VECTOR_SEARCH_ERROR - Vector search failed",
            request_id, "legacy_vector_search",
            error=str(e),
            traceback=traceback.format_exc(),
            response_time_seconds=round(response_time, 3)
        )
        return []

async def sql_search(query: str, context: str, max_results: int, request_id: str = "N/A") -> List[Dict]:
    """Perform SQL-based search."""
    start_time = time.time()
    
    try:
        log_with_context(
            logger, 'info',
            f"LEGACY_SQL_SEARCH_START - Starting legacy SQL search",
            request_id, "legacy_sql_search",
            query=query,
            context_length=len(context),
            max_results=max_results
        )
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{SQL_RETRIEVER_URL}/search",
                json={
                    "query": query,
                    "context": context,
                    "max_results": max_results
                }
            ) as response:
                response_time = time.time() - start_time
                
                if response.status != 200:
                    log_with_context(
                        logger, 'error',
                        f"LEGACY_SQL_SEARCH_ERROR - SQL search failed",
                        request_id, "legacy_sql_search",
                        status_code=response.status,
                        response_time_seconds=round(response_time, 3)
                    )
                    return []
                
                data = await response.json()
                results = data.get("results", [])
                
                log_with_context(
                    logger, 'info',
                    f"LEGACY_SQL_SEARCH_SUCCESS - SQL search completed",
                    request_id, "legacy_sql_search",
                    results_count=len(results),
                    response_time_seconds=round(response_time, 3)
                )
                
                return results
                
    except Exception as e:
        response_time = time.time() - start_time
        log_with_context(
            logger, 'error',
            f"LEGACY_SQL_SEARCH_ERROR - SQL search failed",
            request_id, "legacy_sql_search",
            error=str(e),
            traceback=traceback.format_exc(),
            response_time_seconds=round(response_time, 3)
        )
        return []

async def web_search(query: str, context: str, max_results: int, request_id: str = "N/A") -> List[Dict]:
    """Perform web search."""
    start_time = time.time()
    
    try:
        log_with_context(
            logger, 'info',
            f"LEGACY_WEB_SEARCH_START - Starting legacy web search",
            request_id, "legacy_web_search",
            query=query,
            context_length=len(context),
            max_results=max_results
        )
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{WEB_RETRIEVER_URL}/search",
                json={
                    "query": query,
                    "context": context,
                    "max_results": max_results
                }
            ) as response:
                response_time = time.time() - start_time
                
                if response.status != 200:
                    log_with_context(
                        logger, 'error',
                        f"LEGACY_WEB_SEARCH_ERROR - Web search failed",
                        request_id, "legacy_web_search",
                        status_code=response.status,
                        response_time_seconds=round(response_time, 3)
                    )
                    return []
                
                data = await response.json()
                results = data.get("results", [])
                
                log_with_context(
                    logger, 'info',
                    f"LEGACY_WEB_SEARCH_SUCCESS - Web search completed",
                    request_id, "legacy_web_search",
                    results_count=len(results),
                    response_time_seconds=round(response_time, 3)
                )
                
                return results
                
    except Exception as e:
        response_time = time.time() - start_time
        log_with_context(
            logger, 'error',
            f"LEGACY_WEB_SEARCH_ERROR - Web search failed",
            request_id, "legacy_web_search",
            error=str(e),
            traceback=traceback.format_exc(),
            response_time_seconds=round(response_time, 3)
        )
        return []

def apply_filters(results: List[SearchResult], filters: Dict, request_id: str = "N/A") -> List[SearchResult]:
    """Apply filters to results."""
    start_time = time.time()
    
    try:
        log_with_context(
            logger, 'info',
            f"FILTER_START - Starting result filtering",
            request_id, "apply_filters",
            input_count=len(results),
            filters=filters
        )
        
        if not filters:
            return results
        
        filtered_results = results
        filters_applied = []
        
        # Filter by date range
        if "date_range" in filters:
            initial_count = len(filtered_results)
            start_date = datetime.fromisoformat(filters["date_range"]["start"])
            end_date = datetime.fromisoformat(filters["date_range"]["end"])
            
            # Filter results based on timestamp or metadata timestamp
            new_filtered_results = []
            for r in filtered_results:
                # Try to get timestamp from metadata first, then fall back to result timestamp
                timestamp_str = None
                if hasattr(r, 'metadata') and r.metadata and "timestamp" in r.metadata:
                    timestamp_str = r.metadata["timestamp"]
                elif hasattr(r, 'timestamp') and r.timestamp:
                    timestamp_str = r.timestamp
                
                if timestamp_str:
                    try:
                        result_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if start_date <= result_time <= end_date:
                            new_filtered_results.append(r)
                    except (ValueError, AttributeError):
                        # If timestamp parsing fails, include the result
                        new_filtered_results.append(r)
                else:
                    # If no timestamp found, include the result
                    new_filtered_results.append(r)
            
            filtered_results = new_filtered_results
            
            filters_applied.append({
                'type': 'date_range',
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'removed_count': initial_count - len(filtered_results)
            })
        
        # Filter by source type
        if "source_types" in filters and filters["source_types"]:
            initial_count = len(filtered_results)
            allowed_sources = filters["source_types"]
            
            filtered_results = [
                r for r in filtered_results
                if hasattr(r, 'source') and r.source in allowed_sources
            ]
            
            filters_applied.append({
                'type': 'source_types',
                'allowed_sources': allowed_sources,
                'removed_count': initial_count - len(filtered_results)
            })
        
        # Filter by minimum score
        if "min_score" in filters:
            initial_count = len(filtered_results)
            min_score = float(filters["min_score"])
            
            filtered_results = [
                r for r in filtered_results
                if hasattr(r, 'score') and r.score >= min_score
            ]
            
            filters_applied.append({
                'type': 'min_score',
                'min_score': min_score,
                'removed_count': initial_count - len(filtered_results)
            })
        
        # Filter by content length
        if "min_content_length" in filters:
            initial_count = len(filtered_results)
            min_length = int(filters["min_content_length"])
            
            filtered_results = [
                r for r in filtered_results
                if hasattr(r, 'text') and len(r.text) >= min_length
            ]
            
            filters_applied.append({
                'type': 'min_content_length',
                'min_length': min_length,
                'removed_count': initial_count - len(filtered_results)
            })
        
        processing_time = time.time() - start_time
        
        log_with_context(
            logger, 'info',
            f"FILTER_SUCCESS - Result filtering completed",
            request_id, "apply_filters",
            input_count=len(results),
            output_count=len(filtered_results),
            removed_count=len(results) - len(filtered_results),
            filters_applied=filters_applied,
            processing_time_seconds=round(processing_time, 3)
        )
        
        return filtered_results
        
    except Exception as e:
        processing_time = time.time() - start_time
        log_with_context(
            logger, 'error',
            f"FILTER_ERROR - Result filtering failed",
            request_id, "apply_filters",
            error=str(e),
            traceback=traceback.format_exc(),
            input_count=len(results),
            processing_time_seconds=round(processing_time, 3)
        )
        # Return original results on error
        return results

def calculate_confidence_score(results: List[SearchResult], request_id: str = "N/A") -> float:
    """Calculate overall confidence score for results."""
    start_time = time.time()
    
    try:
        log_with_context(
            logger, 'debug',
            f"CONFIDENCE_START - Starting confidence calculation",
            request_id, "confidence",
            results_count=len(results)
        )
        
        if not results:
            log_with_context(
                logger, 'debug',
                f"CONFIDENCE_EMPTY - No results provided, returning 0.0",
                request_id, "confidence",
                results_count=0
            )
            return 0.0
        
        # Weight factors
        score_weight = 0.6
        source_diversity_weight = 0.2
        result_count_weight = 0.2
        
        # Average of result scores - access .score attribute directly
        scores = [r.score for r in results]
        avg_score = sum(scores) / len(scores)
        
        # Source diversity (number of unique sources) - access .source attribute directly
        unique_sources = len(set(r.source for r in results))
        source_diversity = min(unique_sources / 3, 1.0)  # Normalize to 0-1
        
        # Result count factor
        result_count = min(len(results) / 5, 1.0)  # Normalize to 0-1
        
        # Calculate final confidence score
        confidence_score = (
            avg_score * score_weight +
            source_diversity * source_diversity_weight +
            result_count * result_count_weight
        )
        
        final_confidence = min(confidence_score, 1.0)  # Ensure score is between 0 and 1
        
        processing_time = time.time() - start_time
        
        log_with_context(
            logger, 'debug',
            f"CONFIDENCE_SUCCESS - Confidence calculation completed",
            request_id, "confidence",
            results_count=len(results),
            avg_score=round(avg_score, 3),
            unique_sources=unique_sources,
            source_diversity=round(source_diversity, 3),
            result_count_factor=round(result_count, 3),
            final_confidence=round(final_confidence, 3),
            processing_time_seconds=round(processing_time, 4),
            score_breakdown={
                'score_component': round(avg_score * score_weight, 3),
                'diversity_component': round(source_diversity * source_diversity_weight, 3),
                'count_component': round(result_count * result_count_weight, 3)
            }
        )
        
        return final_confidence
        
    except Exception as e:
        processing_time = time.time() - start_time
        log_with_context(
            logger, 'error',
            f"CONFIDENCE_ERROR - Confidence calculation failed",
            request_id, "confidence",
            error=str(e),
            traceback=traceback.format_exc(),
            results_count=len(results),
            processing_time_seconds=round(processing_time, 4)
        )
        return 0.0

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        log_with_context(
            logger, 'info',
            f"HEALTH_CHECK_START - Starting health check",
            request_id, "/health"
        )
        
        services = {
            "vector_retriever": VECTOR_RETRIEVER_URL,
            "sql_retriever": SQL_RETRIEVER_URL,
            "web_retriever": WEB_RETRIEVER_URL
        }
        
        health_status = {}
        health_details = {}
        
        async with aiohttp.ClientSession() as session:
            for service, url in services.items():
                service_start_time = time.time()
                
                try:
                    log_with_context(
                        logger, 'debug',
                        f"HEALTH_CHECK_SERVICE - Checking service health",
                        request_id, "/health",
                        service=service,
                        url=url
                    )
                    
                    async with session.get(f"{url}/health", timeout=5) as response:
                        service_time = time.time() - service_start_time
                        
                        if response.status == 200:
                            health_status[service] = "healthy"
                            health_details[service] = {
                                'status': 'healthy',
                                'response_time_seconds': round(service_time, 3),
                                'status_code': response.status
                            }
                            log_with_context(
                                logger, 'info',
                                f"HEALTH_CHECK_SERVICE_SUCCESS - Service is healthy",
                                request_id, "/health",
                                service=service,
                                response_time_seconds=round(service_time, 3)
                            )
                        else:
                            health_status[service] = "unhealthy"
                            health_details[service] = {
                                'status': 'unhealthy',
                                'response_time_seconds': round(service_time, 3),
                                'status_code': response.status
                            }
                            log_with_context(
                                logger, 'error',
                                f"HEALTH_CHECK_SERVICE_ERROR - Service returned error status",
                                request_id, "/health",
                                service=service,
                                status_code=response.status,
                                response_time_seconds=round(service_time, 3)
                            )
                            
                except asyncio.TimeoutError:
                    service_time = time.time() - service_start_time
                    health_status[service] = "timeout"
                    health_details[service] = {
                        'status': 'timeout',
                        'response_time_seconds': round(service_time, 3),
                        'error': 'Request timeout'
                    }
                    log_with_context(
                        logger, 'error',
                        f"HEALTH_CHECK_SERVICE_TIMEOUT - Service health check timed out",
                        request_id, "/health",
                        service=service,
                        response_time_seconds=round(service_time, 3)
                    )
                    
                except Exception as e:
                    service_time = time.time() - service_start_time
                    health_status[service] = "unreachable"
                    health_details[service] = {
                        'status': 'unreachable',
                        'response_time_seconds': round(service_time, 3),
                        'error': str(e)
                    }
                    log_with_context(
                        logger, 'error',
                        f"HEALTH_CHECK_SERVICE_ERROR - Error checking service",
                        request_id, "/health",
                        service=service,
                        error=str(e),
                        response_time_seconds=round(service_time, 3)
                    )
        
        overall_status = "healthy" if all(v == "healthy" for v in health_status.values()) else "unhealthy"
        total_time = time.time() - start_time
        
        response_data = {
            "status": overall_status,
            "services": health_status,
            "details": health_details,
            "metadata": {
                "check_time": datetime.utcnow().isoformat(),
                "total_check_time_seconds": round(total_time, 3),
                "services_checked": len(services),
                "healthy_services": sum(1 for v in health_status.values() if v == "healthy"),
                "unhealthy_services": sum(1 for v in health_status.values() if v != "healthy")
            }
        }
        
        log_with_context(
            logger, 'info',
            f"HEALTH_CHECK_SUCCESS - Health check completed",
            request_id, "/health",
            overall_status=overall_status,
            services_checked=len(services),
            healthy_services=response_data["metadata"]["healthy_services"],
            unhealthy_services=response_data["metadata"]["unhealthy_services"],
            total_time_seconds=round(total_time, 3),
            service_details=health_details
        )
        
        return response_data
        
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
        raise HTTPException(status_code=500, detail=str(e))