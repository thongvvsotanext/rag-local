from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, UploadFile, File, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv
import aiohttp
import json
import redis
import jwt
from functools import wraps
from datetime import datetime, timedelta
import asyncio
import logging
import sys
import time
from pydantic import BaseModel, validator
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api_gateway.log')
    ]
)
logger = logging.getLogger(__name__)

from database import database
import models
from schemas import (
    UserCreate, UserLogin, Token,
    ChatRequest, ChatResponse,
    DocumentUpload, DocumentResponse,
    CrawlRequest, CrawlResponse,
    JobStatus,
    ChatSessionCreate, ChatSession,
    ChatMessageCreate, ChatMessage
)

# Load environment variables
load_dotenv()

app = FastAPI(title="Fizen RAG API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Redis client
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))

# JWT Authentication
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")
JWT_ALGORITHM = "HS256"

def create_access_token(data: dict):
    return jwt.encode(data, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except:
        return None

def get_current_user(token: str = Depends(lambda x: x.headers.get("Authorization"))):
    if not token:
        raise HTTPException(status_code=401, detail="No token provided")
    user = verify_token(token.split(" ")[1])
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

manager = ConnectionManager()

# Service URLs
CONTEXT_MANAGER_URL = os.getenv("CONTEXT_MANAGER_URL", "http://context-manager:8001")
RETRIEVAL_ORCHESTRATOR_URL = os.getenv("RETRIEVAL_ORCHESTRATOR_URL", "http://retrieval-orchestrator:8002")
PROMPT_BUILDER_URL = os.getenv("PROMPT_BUILDER_URL", "http://prompt-builder:8007")
LLM_ORCHESTRATOR_URL = os.getenv("LLM_ORCHESTRATOR_URL", "http://llm-orchestrator:8008")
RESPONSE_FORMATTER_URL = os.getenv("RESPONSE_FORMATTER_URL", "http://response-formatter:8009")
WEB_RETRIEVER_URL = os.getenv("WEB_RETRIEVER_URL", "http://web-retriever:8006")
STORAGE_SERVICE_URL = os.getenv("STORAGE_SERVICE_URL", "http://vector-service:8003")
VECTOR_RETRIEVER_URL = os.getenv("VECTOR_RETRIEVER_URL", "http://vector-service:8003")

# Integration logging helper functions
async def log_integration_call(service_name: str, endpoint: str, request_data: Dict[str, Any], request_id: str = None):
    """Log outgoing integration call details."""
    request_id = request_id or f"req_{int(time.time() * 1000)}"
    logger.info(f"🔄 [{request_id}] INTEGRATION_OUT -> {service_name} at {endpoint}")
    logger.info(f"📤 [{request_id}] REQUEST_PAYLOAD: {json.dumps(request_data, indent=2, default=str)}")

async def log_integration_response(service_name: str, endpoint: str, status_code: int, response_data: Dict[str, Any], request_id: str = None, processing_time: float = None):
    """Log integration response details."""
    request_id = request_id or f"req_{int(time.time() * 1000)}"
    time_str = f" (⏱️ {processing_time:.2f}s)" if processing_time else ""
    logger.info(f"✅ [{request_id}] INTEGRATION_IN <- {service_name} - Status: {status_code}{time_str}")
    logger.info(f"📥 [{request_id}] RESPONSE_PAYLOAD: {json.dumps(response_data, indent=2, default=str)}")

async def log_integration_error(service_name: str, endpoint: str, status_code: int, error_detail: str, request_id: str = None, processing_time: float = None):
    """Log integration error details."""
    request_id = request_id or f"req_{int(time.time() * 1000)}"
    time_str = f" (⏱️ {processing_time:.2f}s)" if processing_time else ""
    logger.error(f"❌ [{request_id}] INTEGRATION_ERROR <- {service_name} - Status: {status_code}{time_str}")
    logger.error(f"💥 [{request_id}] ERROR_DETAIL: {error_detail}")

async def call_service_with_logging(service_name: str, url: str, request_data: Dict[str, Any], request_id: str = None) -> Dict[str, Any]:
    """Make service call with comprehensive logging."""
    request_id = request_id or f"req_{int(time.time() * 1000)}"
    start_time = time.time()
    
    await log_integration_call(service_name, url, request_data, request_id)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=request_data) as response:
                processing_time = time.time() - start_time
                
                if response.status != 200:
                    try:
                        error_body = await response.text()
                        await log_integration_error(service_name, url, response.status, error_body, request_id, processing_time)
                        raise HTTPException(
                            status_code=response.status, 
                            detail=f"{service_name} error: Status {response.status}, Response: {error_body}"
                        )
                    except HTTPException:
                        raise
                    except Exception as e:
                        await log_integration_error(service_name, url, response.status, str(e), request_id, processing_time)
                        raise HTTPException(
                            status_code=response.status, 
                            detail=f"{service_name} error: Status {response.status}"
                        )
                
                response_data = await response.json()
                await log_integration_response(service_name, url, response.status, response_data, request_id, processing_time)
                return response_data
                
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        await log_integration_error(service_name, url, 0, f"Connection error: {str(e)}", request_id, processing_time)
        raise HTTPException(status_code=500, detail=f"{service_name} connection error: {str(e)}")

# Database dependency
def get_db():
    db = database.SessionLocal()
    try:
        logger.debug("📦 Database session created")
        yield db
    finally:
        logger.debug("🔒 Database session closed")
        db.close()

# Rate limiting configuration
RATE_LIMIT_WINDOW = 60  # 1 minute window
RATE_LIMIT_MAX_REQUESTS = 600  # 60 requests per minute

class RateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.window = RATE_LIMIT_WINDOW
        self.max_requests = RATE_LIMIT_MAX_REQUESTS

    async def is_rate_limited(self, key: str) -> bool:
        try:
            current = int(time.time())
            window_key = f"rate_limit:{key}:{current // self.window}"
            
            # Get current count
            count = self.redis.get(window_key)
            if count is None:
                # First request in this window
                self.redis.setex(window_key, self.window, 1)
                return False
            
            count = int(count)
            if count >= self.max_requests:
                return True
            
            # Increment counter
            self.redis.incr(window_key)
            return False
        except redis.RedisError as e:
            logger.warning(f"Redis error in rate limiter: {str(e)}")
            # If Redis is not available, allow the request
            return False

rate_limiter = RateLimiter(redis_client)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    try:
        # Skip rate limiting for health checks
        if request.url.path == "/health":
            return await call_next(request)
        
        # Get client identifier (IP or user ID)
        client_id = request.headers.get("X-User-ID") or request.client.host
        
        # Check rate limit
        if await rate_limiter.is_rate_limited(client_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        return await call_next(request)
    except Exception as e:
        logger.error(f"Error in rate limit middleware: {str(e)}")
        # If there's any error in rate limiting, allow the request
        return await call_next(request)

# Authentication routes
@app.post("/auth/register", response_model=Token)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    logger.info(f"👤 Registering new user: {user.email}")
    try:
        try:
            # Check if user exists
            existing_user = db.query(models.User).filter(
                models.User.email == user.email
            ).first()
            
            if existing_user:
                logger.warning(f"⚠️ User already exists: {user.email}")
                raise HTTPException(
                    status_code=400,
                    detail="Email already registered"
                )
            
            try:
                # Create new user
                hashed_password = get_password_hash(user.password)
                db_user = models.User(
                    email=user.email,
                    hashed_password=hashed_password,
                    is_admin=user.is_admin
                )
                db.add(db_user)
                db.commit()
                db.refresh(db_user)
                
                # Create access token
                access_token = create_access_token(
                    data={"sub": db_user.email, "is_admin": db_user.is_admin}
                )
                logger.info(f"✅ User registered successfully: {user.email}")
                return {"access_token": access_token, "token_type": "bearer"}
            except Exception as e:
                db.rollback()
                logger.error(f"❌ Error creating user: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create user: {str(e)}"
                )
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Database error during registration: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during registration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.post("/auth/login", response_model=Token)
async def login(user: UserLogin, db: Session = Depends(get_db)):
    logger.info(f"🔑 Login attempt for user: {user.email}")
    try:
        try:
            # Authenticate user
            db_user = db.query(models.User).filter(
                models.User.email == user.email
            ).first()
            
            if not db_user or not verify_password(user.password, db_user.hashed_password):
                logger.warning(f"⚠️ Invalid login attempt for: {user.email}")
                raise HTTPException(
                    status_code=401,
                    detail="Incorrect email or password"
                )
            
            try:
                # Create access token
                access_token = create_access_token(
                    data={"sub": db_user.email, "is_admin": db_user.is_admin}
                )
                logger.info(f"✅ User logged in successfully: {user.email}")
                return {"access_token": access_token, "token_type": "bearer"}
            except Exception as e:
                logger.error(f"❌ Error creating access token: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create access token: {str(e)}"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Database error during login: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during login: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

# Chat routes
class ChatRequest(BaseModel):
    user_id: str
    query: str
    session_id: str
    filters: Optional[Dict[str, Any]] = None
    max_results: Optional[int] = 5

    @validator('query')
    def validate_query(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Query cannot be empty')
        if len(v) > 1000:
            raise ValueError('Query is too long (max 1000 characters)')
        return v.strip()

    @validator('session_id')
    def validate_session_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Session ID cannot be empty')
        return v.strip()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """
    Handle chat requests by coordinating between services.
    
    Input:
    {
        "user_id": "abc123",
        "query": "What's our refund policy?",
        "session_id": "session123",
        "filters": {
            "source_types": ["document", "web"],
            "date_range": {"start": "2024-01-01", "end": "2024-12-31"}
        },
        "max_results": 5
    }
    
    Output:
    {
        "response": "Our refund policy allows returns within 30 days...",
        "session_id": "session123",
        "sources": [
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
        "metadata": {
            "processing_time": 1.5,
            "sources_used": ["vector", "web"],
            "confidence_score": 0.92
        }
    }
    """
    request_id = f"chat_{int(time.time() * 1000)}"
    logger.info(f"🚀 [{request_id}] CHAT_REQUEST: user_id={request.user_id}, query='{request.query[:100]}...', session_id={request.session_id}")
    
    try:
        start_time = time.time()
        
        # 1. Get context from Context Manager
        logger.info(f"🔍 [{request_id}] Step 1: Fetching context from Context Manager")
        context_data = await call_service_with_logging(
            "Context Manager",
            f"{CONTEXT_MANAGER_URL}/context",
            {
                "user_id": request.user_id,
                "query": request.query,
                "session_id": request.session_id,
                "filters": request.filters
            },
            request_id
        )

        # 2. Get relevant information from Retrieval Orchestrator
        context_summary = context_data.get("context", "")[:200] + "..." if len(context_data.get("context", "")) > 200 else context_data.get("context", "")
        logger.info(f"🔍 [{request_id}] Step 2: Fetching information from Retrieval Orchestrator with context: '{context_summary}'")
        retrieval_data = await call_service_with_logging(
            "Retrieval Orchestrator",
            f"{RETRIEVAL_ORCHESTRATOR_URL}/search",
            {
                "query": request.query,
                "context": context_data.get("context", ""),
                "filters": request.filters,
                "max_results": request.max_results
            },
            request_id
        )

        # 3. Build prompt with Prompt Builder
        results_count = len(retrieval_data.get("results", []))
        results_summary = []
        for i, result in enumerate(retrieval_data.get("results", [])[:3]):  # Show first 3 results
            content_preview = result.get("content", "")[:100] + "..." if len(result.get("content", "")) > 100 else result.get("content", "")
            score = result.get("score", 0)
            source = result.get("metadata", {}).get("source", "unknown")
            results_summary.append(f"[{i+1}] Score: {score:.3f}, Source: {source}, Content: '{content_preview}'")
        
        results_preview = " | ".join(results_summary) if results_summary else "No results"
        logger.info(f"🔍 [{request_id}] Step 3: Building prompt with Prompt Builder using {results_count} retrieved results: {results_preview}")
        prompt_data = await call_service_with_logging(
            "Prompt Builder",
            f"{PROMPT_BUILDER_URL}/build",
            {
                "query": request.query,
                "context": context_data.get("context", ""),
                "retrieved_data": retrieval_data.get("results", []),
                "filters": request.filters
            },
            request_id
        )

        # 4. Get LLM response from LLM Orchestrator
        prompt_preview = prompt_data.get("prompt", "")[:200] + "..." if len(prompt_data.get("prompt", "")) > 200 else prompt_data.get("prompt", "")
        logger.info(f"🔍 [{request_id}] Step 4: Getting response from LLM Orchestrator with prompt: '{prompt_preview}'")
        llm_data = await call_service_with_logging(
            "LLM Orchestrator",
            f"{LLM_ORCHESTRATOR_URL}/generate",
            {
                "prompt": prompt_data.get("prompt", ""),
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            },
            request_id
        )

        # 5. Format response with Response Formatter
        llm_response_preview = llm_data.get("response", "")[:200] + "..." if len(llm_data.get("response", "")) > 200 else llm_data.get("response", "")
        logger.info(f"🔍 [{request_id}] Step 5: Formatting response: '{llm_response_preview}'")
        format_data = await call_service_with_logging(
            "Response Formatter",
            f"{RESPONSE_FORMATTER_URL}/format",
            {
                "response": llm_data.get("response", ""),
                "sources": retrieval_data.get("results", []),
                "format": "markdown"
            },
            request_id
        )

        # Store message in history
        await store_message(
            user_id=request.user_id,
            text=request.query,
            role="user",
            session_id=request.session_id
        )

        processing_time = time.time() - start_time
        
        response = ChatResponse(
            response=format_data.get("formatted_response", ""),
            session_id=request.session_id,
            sources=retrieval_data.get("results", []),
            metadata={
                "processing_time": processing_time,
                "sources_used": retrieval_data.get("sources_used", []),
                "confidence_score": retrieval_data.get("confidence_score", 0.0)
            }
        )
        
        logger.info(f"✅ [{request_id}] CHAT_COMPLETE: processing_time={processing_time:.2f}s, sources_count={len(retrieval_data.get('results', []))}")
        return response

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [{request_id}] CHAT_ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Document routes
@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[Dict[str, Any]] = None,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"📄 Uploading document: {file.filename}")
    try:
        try:
            # Check if document already exists
            doc_id = f"{current_user['id']}_{file.filename}"
            existing_doc = db.query(models.Document).filter(
                models.Document.doc_id == doc_id
            ).first()
            
            if existing_doc:
                logger.warning(f"⚠️ Document already exists: {file.filename}")
                raise HTTPException(
                    status_code=400,
                    detail="Document already exists"
                )
            
            try:
                # Save file and create document record
                file_path = f"uploads/{doc_id}"
                file_size = 0
                
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    file_size = len(content)
                    buffer.write(content)
                
                db_document = models.Document(
                    doc_id=doc_id,
                    owner_id=current_user['id'],
                    filename=file.filename,
                    file_path=file_path,
                    file_type=file.content_type,
                    size=file_size,
                    status="processing",
                    document_metadata=metadata or {}
                )
                
                db.add(db_document)
                db.commit()
                db.refresh(db_document)
                
                logger.info(f"✅ Document uploaded successfully: {file.filename}")
                return {
                    "doc_id": db_document.doc_id,
                    "filename": db_document.filename,
                    "status": db_document.status
                }
            except Exception as e:
                db.rollback()
                logger.error(f"❌ Error uploading document: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to upload document: {str(e)}"
                )
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Database error during upload: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

# Crawl routes
@app.post("/crawl", response_model=CrawlResponse)
async def start_crawl(
    request: CrawlRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"🕷️ Starting crawl job for URLs: {request.start_urls}")
    try:
        try:
            # Check if job already exists
            job_id = f"{current_user['id']}_{int(time.time())}"
            existing_job = db.query(models.CrawlJob).filter(
                models.CrawlJob.job_id == job_id
            ).first()
            
            if existing_job:
                logger.warning(f"⚠️ Crawl job already exists: {job_id}")
                raise HTTPException(
                    status_code=400,
                    detail="Crawl job already exists"
                )
            
            try:
                # Create crawl job
                db_job = models.CrawlJob(
                    job_id=job_id,
                    user_id=current_user['id'],
                    start_urls=request.start_urls,
                    max_pages=request.max_pages,
                    max_depth=request.max_depth,
                    status="pending",
                    filters=request.filters or {}
                )
                
                db.add(db_job)
                db.commit()
                db.refresh(db_job)
                
                logger.info(f"✅ Crawl job created successfully: {job_id}")
                return {
                    "job_id": db_job.job_id,
                    "status": db_job.status,
                    "start_urls": db_job.start_urls
                }
            except Exception as e:
                db.rollback()
                logger.error(f"❌ Error creating crawl job: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create crawl job: {str(e)}"
                )
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Database error during crawl job creation: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during crawl job creation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/crawl/jobs", response_model=List[CrawlResponse])
async def list_crawl_jobs(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"📋 Listing crawl jobs for user: {current_user['email']}")
    try:
        try:
            jobs = db.query(models.CrawlJob).filter(
                models.CrawlJob.user_id == current_user['id']
            ).all()
            
            logger.info(f"✅ Found {len(jobs)} crawl jobs")
            return [{
                "job_id": job.job_id,
                "status": job.status,
                "start_urls": job.start_urls
            } for job in jobs]
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Database error while listing crawl jobs: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to list crawl jobs: {str(e)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error while listing crawl jobs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"🔍 Getting status for job: {job_id}")
    try:
        try:
            job = db.query(models.CrawlJob).filter(
                models.CrawlJob.job_id == job_id,
                models.CrawlJob.user_id == current_user['id']
            ).first()
            
            if not job:
                logger.warning(f"⚠️ Job not found: {job_id}")
                raise HTTPException(
                    status_code=404,
                    detail="Job not found"
                )
            
            logger.info(f"✅ Job status retrieved: {job_id}")
            return {
                "job_id": job.job_id,
                "status": job.status,
                "pages_crawled": job.pages_crawled,
                "pages_failed": job.pages_failed,
                "chunks_created": job.chunks_created
            }
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Database error while getting job status: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get job status: {str(e)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error while getting job status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages
            await manager.send_message(f"Message received: {data}", client_id)
    except WebSocketDisconnect:
        manager.disconnect(client_id)

# Health check
@app.get("/health")
async def health_check():
    logger.info("🏥 Performing health check")
    try:
        try:
            # Check database connection
            db = SessionLocal()
            try:
                db.execute("SELECT 1")
                db_status = "healthy"
            except Exception as e:
                logger.error(f"❌ Database health check failed: {str(e)}")
                db_status = "unhealthy"
            finally:
                db.close()
            
            # Check Redis connection
            try:
                redis_client.ping()
                redis_status = "healthy"
            except Exception as e:
                logger.error(f"❌ Redis health check failed: {str(e)}")
                redis_status = "unhealthy"
            
            status = {
                "status": "healthy" if db_status == "healthy" and redis_status == "healthy" else "unhealthy",
                "database": db_status,
                "redis": redis_status,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Health check completed: {status}")
            return status
        except Exception as e:
            logger.error(f"❌ Error during health check: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Health check failed: {str(e)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during health check: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

async def store_message(user_id: str, text: str, role: str = "user", session_id: Optional[str] = None):
    """Store a message in the context manager."""
    request_id = f"store_{int(time.time() * 1000)}"
    try:
        logger.info(f"💾 [{request_id}] Storing message for user {user_id} in session {session_id}")
        
        await call_service_with_logging(
            "Context Manager (Store)",
            f"{CONTEXT_MANAGER_URL}/store",
            {
                "user_id": user_id,
                "text": text,
                "metadata": {
                    "role": role,
                    "timestamp": datetime.utcnow().isoformat()
                }
            },
            request_id
        )
        
        logger.info(f"✅ [{request_id}] Successfully stored message for user {user_id}")
    except Exception as e:
        logger.error(f"❌ [{request_id}] Error storing message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error storing message: {str(e)}")

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = datetime.utcnow()
    logger.info(f"🚀 Request started: {request.method} {request.url}")
    logger.info(f"📝 Request headers: {request.headers}")
    
    try:
        response = await call_next(request)
        process_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"✅ Request completed: {request.method} {request.url} - Status: {response.status_code} - Time: {process_time:.2f}s")
        return response
    except Exception as e:
        logger.error(f"❌ Request failed: {request.method} {request.url} - Error: {str(e)}")
        raise 

class APIError(Exception):
    def __init__(self, message: str, status_code: int = 500, error_code: str = "INTERNAL_ERROR"):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(message)

@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"❌ HTTP Exception: {exc.detail} - Status: {exc.status_code}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": exc.detail,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"❌ Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

# Session endpoints
@app.post("/sessions", response_model=ChatSession)
async def create_session(request: Request, db: Session = Depends(get_db)):
    logger.info("📝 Creating new session")
    try:
        # Generate session ID using IP and hourly timestamp
        client_ip = request.client.host
        hourly_timestamp = int(time.time() // 3600)  # Get current hour timestamp
        session_id = f"{client_ip.replace('.', '_')}_{hourly_timestamp}"
        
        # Check if session already exists
        try:
            existing_session = db.query(models.ChatSession).filter(
                models.ChatSession.session_id == session_id
            ).first()
            
            if existing_session:
                logger.info(f"✅ Found existing session: {session_id}")
                # Update last activity
                existing_session.last_activity = datetime.utcnow()
                try:
                    db.commit()
                    db.refresh(existing_session)
                    # Convert SQLAlchemy model to dict before returning
                    session_dict = existing_session.__dict__.copy()
                    return ChatSession.parse_obj(session_dict)
                except Exception as e:
                    db.rollback()
                    logger.error(f"❌ Error updating existing session: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to update session: {str(e)}"
                    )
            
            # Create new session if it doesn't exist
            current_time = datetime.utcnow()
            session = models.ChatSession(
                session_id=session_id,
                session_type="anonymous",
                ip_address=request.client.host,
                user_agent=request.headers.get("user-agent", "unknown"),
                last_activity=current_time,
                created_at=current_time,
                expires_at= current_time + timedelta(hours=1)  # Sessions don't expire by default
            )
            
            try:
                db.add(session)
                db.commit()
                db.refresh(session)
                logger.info(f"✅ New session created: {session.session_id}")
                # Convert SQLAlchemy model to dict before returning
                session_dict = session.__dict__.copy()
                return ChatSession.parse_obj(session_dict)
            except Exception as e:
                db.rollback()
                logger.error(f"❌ Error creating new session: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create session: {str(e)}"
                )
                
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Database error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in session creation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/sessions", response_model=List[ChatSession])
async def list_sessions(db: Session = Depends(get_db)):
    logger.info("📋 Listing all sessions")
    try:
        try:
            sessions = db.query(models.ChatSession).all()
            logger.info(f"✅ Found {len(sessions)} sessions")
            return [ChatSession.parse_obj({
                **session.__dict__,
                'created_at': session.__dict__.get('created_at', datetime.utcnow()),
                'expires_at': session.__dict__.get('expires_at', None)
            }) for session in sessions]
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Database error while listing sessions: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to list sessions: {str(e)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error while listing sessions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/sessions/{session_id}", response_model=ChatSession)
async def get_session(session_id: str, db: Session = Depends(get_db)):
    logger.info(f"🔍 Getting session: {session_id}")
    try:
        try:
            session = db.query(models.ChatSession).filter(
                models.ChatSession.session_id == session_id
            ).first()
            
            if not session:
                logger.warning(f"⚠️ Session not found: {session_id}")
                raise HTTPException(status_code=404, detail="Session not found")
                
            logger.info(f"✅ Session found: {session_id}")
            session_dict = session.__dict__.copy()
            session_dict['created_at'] = session_dict.get('created_at', datetime.utcnow())
            session_dict['expires_at'] = session_dict.get('expires_at', None)
            return ChatSession.parse_obj(session_dict)
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Database error while getting session: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get session: {str(e)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error while getting session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    logger.info(f"🗑️ Deleting session: {session_id}")
    try:
        try:
            session = db.query(models.ChatSession).filter(
                models.ChatSession.session_id == session_id
            ).first()
            
            if not session:
                logger.warning(f"⚠️ Session not found: {session_id}")
                raise HTTPException(status_code=404, detail="Session not found")
                
            try:
                db.delete(session)
                db.commit()
                logger.info(f"✅ Session deleted: {session_id}")
                return {"message": "Session deleted"}
            except Exception as e:
                db.rollback()
                logger.error(f"❌ Error deleting session: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to delete session: {str(e)}"
                )
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Database error while deleting session: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error while deleting session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

# Message endpoints
@app.post("/sessions/{session_id}/messages", response_model=ChatMessage)
async def create_message(
    session_id: str,
    message: ChatMessageCreate,
    db: Session = Depends(get_db)
):
    """
    Handle message creation with unified query processing flow.
    
    Input:
    {
        "message_text": "What's our refund policy?",
        "message_type": "user",
        "sources_used": {
            "source_types": ["document", "web"],
            "date_range": {"start": "2024-01-01", "end": "2024-12-31"}
        }
    }
    """
    request_id = f"msg_{int(time.time() * 1000)}"
    logger.info(f"📨 [{request_id}] MESSAGE_REQUEST: session_id={session_id}, message='{message.message_text[:100]}...', type={message.message_type}")
    
    try:
        start_time = time.time()
        
        # 1. Validate session exists
        session = db.query(models.ChatSession).filter(
            models.ChatSession.session_id == session_id
        ).first()
        
        if not session:
            logger.warning(f"⚠️ [{request_id}] Session not found: {session_id}")
            raise HTTPException(status_code=404, detail="Session not found")

        # 2. Get context from Context Manager
        logger.info(f"🔍 [{request_id}] Step 1: Fetching context from Context Manager")
        context_data = await call_service_with_logging(
            "Context Manager",
            f"{CONTEXT_MANAGER_URL}/context",
            {
                "query": message.message_text,
                "session_id": session_id,
                "session_type": "anonymous",  # Default to anonymous for now
                "user_id": None  # Optional field
            },
            request_id
        )

        # 3. Get relevant information from Retrieval Orchestrator
        context_summary = context_data.get("context", "")[:200] + "..." if len(context_data.get("context", "")) > 200 else context_data.get("context", "")
        logger.info(f"🔍 [{request_id}] Step 2: Fetching information from Retrieval Orchestrator with context: '{context_summary}'")
        retrieval_data = await call_service_with_logging(
            "Retrieval Orchestrator",
            f"{RETRIEVAL_ORCHESTRATOR_URL}/search",
            {
                "query": message.message_text,
                "context": context_data.get("context", ""),
                "filters": message.sources_used
            },
            request_id
        )

        # 4. Build prompt
        results_count = len(retrieval_data.get("results", []))
        results_summary = []
        for i, result in enumerate(retrieval_data.get("results", [])[:3]):  # Show first 3 results
            content_preview = result.get("content", "")[:100] + "..." if len(result.get("content", "")) > 100 else result.get("content", "")
            score = result.get("score", 0)
            source = result.get("metadata", {}).get("source", "unknown")
            results_summary.append(f"[{i+1}] Score: {score:.3f}, Source: {source}, Content: '{content_preview}'")
        
        results_preview = " | ".join(results_summary) if results_summary else "No results"
        logger.info(f"🔍 [{request_id}] Step 3: Building prompt with Prompt Builder using {results_count} retrieved results: {results_preview}")
        prompt_data = await call_service_with_logging(
            "Prompt Builder",
            f"{PROMPT_BUILDER_URL}/build",
            {
                "query": message.message_text,
                "context": context_data.get("context", ""),
                "retrieved_data": retrieval_data.get("results", []),
                "filters": message.sources_used
            },
            request_id
        )

        # 5. Get LLM response
        prompt_preview = prompt_data.get("prompt", "")[:200] + "..." if len(prompt_data.get("prompt", "")) > 200 else prompt_data.get("prompt", "")
        logger.info(f"🔍 [{request_id}] Step 4: Getting response from LLM Orchestrator with prompt: '{prompt_preview}'")
        llm_data = await call_service_with_logging(
            "LLM Orchestrator",
            f"{LLM_ORCHESTRATOR_URL}/generate",
            {
                "prompt": prompt_data.get("prompt", ""),
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            },
            request_id
        )

        # 6. Format response
        llm_response_preview = llm_data.get("response", "")[:200] + "..." if len(llm_data.get("response", "")) > 200 else llm_data.get("response", "")
        logger.info(f"🔍 [{request_id}] Step 5: Formatting response: '{llm_response_preview}'")
        format_data = await call_service_with_logging(
            "Response Formatter",
            f"{RESPONSE_FORMATTER_URL}/format",
            {
                "response": llm_data.get("response", ""),
                "sources": retrieval_data.get("results", []),
                "format": "markdown"
            },
            request_id
        )

        # 7. Store message in context manager for history
        logger.info(f"🔍 [{request_id}] Step 6: Storing message in context manager")
        store_data = await call_service_with_logging(
            "Context Manager (Store)",
            f"{CONTEXT_MANAGER_URL}/store",
            {
                "session_id": session_id,
                "message_text": format_data.get("formatted_response", ""),
                "message_type": "assistant",
                "sources_used": {
                    "sources": retrieval_data.get("results", []),
                    "metadata": {
                        "processing_time": int((time.time() - start_time) * 1000),
                        "sources_used": retrieval_data.get("sources_used", []),
                        "confidence_score": retrieval_data.get("confidence_score", 0.0)
                    }
                }
            },
            request_id
        )

        # Return response in format expected by frontend
        response_time_ms = int((time.time() - start_time) * 1000)
        
        response = ChatMessage(
            id=store_data.get("message_id"),
            session_id=session_id,
            message_text=format_data.get("formatted_response", ""),
            message_type="assistant",
            sources_used={
                "sources": retrieval_data.get("results", []),
                "metadata": {
                    "processing_time": response_time_ms,
                    "sources_used": retrieval_data.get("sources_used", []),
                    "confidence_score": retrieval_data.get("confidence_score", 0.0)
                }
            },
            response_time_ms=response_time_ms,
            created_at=datetime.utcnow()
        )
        
        logger.info(f"✅ [{request_id}] MESSAGE_COMPLETE: processing_time={response_time_ms}ms, sources_count={len(retrieval_data.get('results', []))}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [{request_id}] MESSAGE_ERROR: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/sessions/{session_id}/messages", response_model=List[ChatMessage])
async def get_messages(session_id: str, db: Session = Depends(get_db)):
    logger.info(f"💬 Getting messages for session: {session_id}")
    try:
        try:
            session = db.query(models.ChatSession).filter(
                models.ChatSession.session_id == session_id
            ).first()
            
            if not session:
                logger.warning(f"⚠️ Session not found: {session_id}")
                raise HTTPException(status_code=404, detail="Session not found")

            try:
                messages = db.query(models.ChatMessage).filter(
                    models.ChatMessage.session_id == session.session_id
                ).all()
                logger.info(f"✅ Found {len(messages)} messages for session {session_id}")
                return [ChatMessage.parse_obj(message.__dict__) for message in messages]
            except Exception as e:
                db.rollback()
                logger.error(f"❌ Error getting messages: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get messages: {str(e)}"
                )
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Database error while getting messages: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error while getting messages: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )
