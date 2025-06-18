from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import os
from dotenv import load_dotenv
import aiohttp
import numpy as np
import asyncpg
import logging
import sys
from datetime import datetime
from sentence_transformers import SentenceTransformer
import redis
import json
from contextlib import asynccontextmanager
import asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Fixed import
import tiktoken
import PyPDF2
import docx
import io
import time
import traceback
import uuid
from urllib.parse import urlparse
from faiss_store import M2OptimizedFAISSStore
from functools import lru_cache  # Added missing import
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, Text, text, inspect, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

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
file_handler = logging.FileHandler('vector_retriever.log')
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

# Configuration
# Build database URL from individual environment variables
DB_HOST = os.getenv("POSTGRES_HOST", "db")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_USER = os.getenv("POSTGRES_USER", "fizen_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fizen_password")
DB_NAME = os.getenv("POSTGRES_DB", "fizen_rag")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
VECTOR_DIM = 384  # BGE-small dimension (corrected)
EMBEDDING_DIMENSIONS = 384  # BGE-small dimension (corrected)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 20
DOC_SIMILARITY_THRESHOLD = 0.75  # For document retrieval
CHAT_SIMILARITY_THRESHOLD = 0.7   # For chat context retrieval

# Set HuggingFace home directory
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/app/data/huggingface")

# Redis setup
redis_client = redis.from_url(REDIS_URL)

# Initialize BGE-small model
logger.info("Initializing BGE-small model")
model = SentenceTransformer('BAAI/bge-small-en')  # Using BGE-small for better performance
logger.info("BGE-small model initialized successfully")

# Validate model embedding dimensions on startup
try:
    test_embedding = model.encode("test").tolist()
    actual_dim = len(test_embedding)
    if actual_dim != VECTOR_DIM:
        logger.error(f"Model dimension mismatch! Expected: {VECTOR_DIM}, Actual: {actual_dim}")
        logger.error(f"Please update VECTOR_DIM and EMBEDDING_DIMENSIONS constants to {actual_dim}")
        raise ValueError(f"Model produces {actual_dim}-dimensional embeddings, but code expects {VECTOR_DIM}")
    else:
        logger.info(f"Model dimension validation successful: {actual_dim}")
except Exception as e:
    logger.error(f"Failed to validate model dimensions: {e}")
    raise

# Initialize FAISS store
FAISS_DATA_PATH = os.getenv("FAISS_DATA_PATH", "/app/data/faiss")
vector_store = M2OptimizedFAISSStore(dimension=VECTOR_DIM, index_type="Flat")  # Use correct BGE-small dimension

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

def init_db():
    """Initialize database and create tables if they don't exist."""
    try:
        # Check if tables exist
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        required_tables = ["document_chunks", "web_chunks", "chat_messages"]
        
        missing_tables = [table for table in required_tables if table not in existing_tables]
        if missing_tables:
            logger.info(f"Creating missing tables: {', '.join(missing_tables)}")
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully")
        else:
            logger.info("All required tables exist")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Input/Output Models
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    filters: Optional[Dict[str, Any]] = None
    metadata_boost: Optional[Dict[str, float]] = None

class SearchResult(BaseModel):
    text: str
    source: str
    score: float
    metadata: Dict[str, Any]
    vector_score: float
    keyword_score: float
    metadata_score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]
    metadata: Dict[str, Any]

class DocumentUploadResponse(BaseModel):
    doc_id: str
    status: str
    chunks_created: int
    processing_time: float

class DocumentProcessRequest(BaseModel):
    job_id: str
    doc_id: str
    filename: str
    file_type: str
    metadata: Dict[str, Any]

class DocumentIngestionResponse(BaseModel):
    status: str
    chunks_created: int
    doc_id: str
    faiss_index_updated: bool
    chunks: List[Dict[str, Any]]

class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    doc_id: str
    page: int
    embedding_stored: bool

class DocumentSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class DocumentSearchResult(BaseModel):
    text: str
    source: str
    score: float
    chunk_id: str
    page: int

class ChatContextRequest(BaseModel):
    query: str
    session_id: str
    context_type: str = "chat_history"
    top_k: Optional[int] = 5

class ChatContextResult(BaseModel):
    message: str
    timestamp: str
    score: float
    message_type: str

# Input/Output Models for /embed endpoint
class EmbedRequest(BaseModel):
    """Request model for embedding generation."""
    texts: Union[str, List[str]] = Field(..., description="Text or list of texts to embed")
    batch_size: Optional[int] = Field(8, ge=1, le=32, description="Batch size for processing (max 32 for M2 Mac)")
    normalize_embeddings: Optional[bool] = Field(False, description="Whether to normalize embeddings to unit length")
    include_metadata: Optional[bool] = Field(False, description="Whether to include processing metadata")
    
    @validator('texts')
    def validate_texts(cls, v):
        if isinstance(v, str):
            return [v]  # Convert single string to list
        if isinstance(v, list):
            if not v:
                raise ValueError("texts cannot be empty")
            if len(v) > 32:
                raise ValueError("Maximum 32 texts allowed for M2 Mac")
            # Validate each text
            for i, text in enumerate(v):
                if not isinstance(text, str):
                    raise ValueError(f"Text at index {i} must be a string")
                if len(text.strip()) == 0:
                    raise ValueError(f"Text at index {i} cannot be empty")
                if len(text) > 8192:  # Reasonable limit for BGE-small
                    raise ValueError(f"Text at index {i} too long (max 8192 characters)")
        return v

class EmbedResponse(BaseModel):
    """Response model for embedding generation."""
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    model: str = Field(..., description="Model used for embedding generation")
    dimension: int = Field(..., description="Embedding dimension")
    processing_time_seconds: float = Field(..., description="Total processing time")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional processing metadata")


# Add new model for crawl job statistics
class CrawlJobStats(BaseModel):
    total_pages: int
    processed_pages: int
    failed_pages: int
    pending_pages: int
    avg_processing_time: float
    start_time: datetime
    last_update: datetime

# Add new model for content distribution
class ContentDistribution(BaseModel):
    source: str
    total_chunks: int
    avg_chunk_size: float
    total_tokens: int
    last_updated: datetime

# Database Models
class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Text)  # Store as JSON string
    document_metadata = Column(JSON)  # Renamed from metadata
    created_at = Column(DateTime(timezone=True), server_default=text('now()'))

    # Relationships
    document = relationship("Document", back_populates="chunks")

class WebPage(Base):
    __tablename__ = "web_pages"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String(2048), unique=True, nullable=False)
    title = Column(String(500))
    content = Column(Text)
    document_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=text('now()'))
    updated_at = Column(DateTime(timezone=True), onupdate=text('now()'))

    # Relationships
    chunks = relationship("WebChunk", back_populates="page")

class WebChunk(Base):
    __tablename__ = "web_chunks"

    id = Column(Integer, primary_key=True, index=True)
    page_id = Column(Integer, ForeignKey("web_pages.id"), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Text)  # Store as JSON string
    document_metadata = Column(JSON)  # Renamed from metadata
    created_at = Column(DateTime(timezone=True), server_default=text('now()'))

    # Relationships
    page = relationship("WebPage", back_populates="chunks")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), ForeignKey("chat_sessions.session_id"), nullable=False)
    message_text = Column(Text, nullable=False)
    message_type = Column(String(50), nullable=False)  # user, assistant
    faiss_index = Column(Integer)
    response_time_ms = Column(Integer)
    sources_used = Column(JSON)
    document_metadata = Column(JSON)  # Renamed from metadata
    created_at = Column(DateTime(timezone=True), server_default=text('now()'))

    # Relationships
    session = relationship("ChatSession", back_populates="messages")

# Helper Functions
def get_tokenizer():
    """Get tokenizer for text splitting."""
    return tiktoken.get_encoding("cl100k_base")

def split_text(text: str) -> List[str]:
    """Split text into chunks using recursive character splitting."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

async def extract_text_from_file(file: UploadFile) -> str:
    """Extract text from uploaded file based on file type."""
    content = await file.read()
    
    if file.filename.endswith('.pdf'):
        pdf_file = io.BytesIO(content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    elif file.filename.endswith('.docx'):
        doc_file = io.BytesIO(content)
        doc = docx.Document(doc_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    elif file.filename.endswith('.txt'):
        return content.decode('utf-8')
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

def normalize_text(text: str) -> str:
    """Normalize text by removing headers, footers, and extra whitespace."""
    # Remove common header/footer patterns
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        # Skip common header/footer patterns
        if any(pattern in line.lower() for pattern in [
            'page', 'confidential', 'copyright', 'www.', 'http://',
            '©', 'all rights reserved'
        ]):
            continue
            
        # Skip empty lines or lines with just numbers
        if not line.strip() or line.strip().isdigit():
            continue
            
        filtered_lines.append(line)
    
    # Join lines and normalize whitespace
    normalized = ' '.join(filtered_lines)
    normalized = ' '.join(normalized.split())
    
    return normalized

# Enhanced embedding functions with better error handling and logging
async def embed_text_enhanced(text: str, normalize: bool = False) -> List[float]:
    """Generate embedding for a single text with enhanced error handling."""
    try:
        # Input validation
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Truncate if too long (BGE-small has 512 token limit)
        text = text[:8192]  # Conservative character limit
        
        # Tokenize with error handling
        inputs = tokenizer(
            [text], 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=512
        )

        # Move to appropriate device
        if device.type == "mps":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)

        # Extract embedding (mean pooling)
        embedding = output.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
        
        # Normalize if requested
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        # Convert to list
        embedding_list = embedding.tolist()

        # Clean up GPU memory on M2
        if device.type == "mps":
            torch.mps.empty_cache()

        return embedding_list

    except Exception as e:
        # Clean up on error
        if device.type == "mps":
            torch.mps.empty_cache()
        raise RuntimeError(f"Error generating embedding: {str(e)}")

async def embed_texts_batch_enhanced(
    texts: List[str], 
    batch_size: int = 8,
    normalize: bool = False,
    include_metadata: bool = False
) -> Dict[str, Any]:
    """Generate embeddings for multiple texts with enhanced batching and metadata."""
    start_time = time.time()
    all_embeddings = []
    processing_metadata = {
        "total_texts": len(texts),
        "batch_size": batch_size,
        "batches_processed": 0,
        "avg_batch_time": 0,
        "total_tokens_processed": 0,
        "device_used": str(device),
        "model_name": model_name
    }
    
    batch_times = []

    try:
        # Process in batches for M2 memory management
        for i in range(0, len(texts), batch_size):
            batch_start_time = time.time()
            batch_texts = texts[i:i + batch_size]
            
            # Truncate texts in batch
            batch_texts = [text[:8192] for text in batch_texts]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=512
            )

            # Track token count for metadata
            if include_metadata:
                processing_metadata["total_tokens_processed"] += sum(
                    (inputs["input_ids"] != tokenizer.pad_token_id).sum().item() 
                    for _ in range(len(batch_texts))
                )

            # Move to appropriate device
            if device.type == "mps":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model(**inputs)

            # Extract embeddings (mean pooling)
            batch_embeddings = output.last_hidden_state.mean(dim=1).cpu().numpy()
            
            # Normalize if requested
            if normalize:
                norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                batch_embeddings = batch_embeddings / np.maximum(norms, 1e-8)
            
            # Convert to list and add to results
            batch_embeddings_list = batch_embeddings.tolist()
            all_embeddings.extend(batch_embeddings_list)

            # Clean up memory after each batch
            del output, inputs
            if device.type == "mps":
                torch.mps.empty_cache()
            
            # Record batch timing
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            processing_metadata["batches_processed"] += 1
            
            # Force garbage collection for memory management
            import gc
            gc.collect()

        # Calculate metadata
        total_time = time.time() - start_time
        processing_metadata["avg_batch_time"] = sum(batch_times) / len(batch_times) if batch_times else 0
        
        result = {
            "embeddings": all_embeddings,
            "processing_time_seconds": total_time
        }
        
        if include_metadata:
            result["metadata"] = processing_metadata
            
        return result

    except Exception as e:
        # Clean up on error
        if device.type == "mps":
            torch.mps.empty_cache()
        raise RuntimeError(f"Error in batch embedding generation: {str(e)}")


@lru_cache(maxsize=1000)
async def get_embedding(text: str) -> List[float]:
    """Get embedding for text with caching and validation."""
    try:
        # Check Redis cache first
        cache_key = f"embedding:{hash(text)}"
        cached = redis_client.get(cache_key)
        if cached:
            embedding = json.loads(cached)
            # Validate cached embedding dimension
            if len(embedding) != VECTOR_DIM:
                logger.warning(f"Cached embedding dimension mismatch: expected {VECTOR_DIM}, got {len(embedding)}")
                # Remove invalid cache entry
                redis_client.delete(cache_key)
            else:
                return embedding
        
        # Generate embedding
        embedding = model.encode(text).tolist()
        
        # Validate embedding dimension
        if len(embedding) != VECTOR_DIM:
            raise ValueError(f"Embedding dimension mismatch: expected {VECTOR_DIM}, got {len(embedding)}. "
                           f"Check if the model '{model}' produces {VECTOR_DIM}-dimensional embeddings.")
        
        # Cache the embedding
        redis_client.setex(cache_key, 3600, json.dumps(embedding))
        return embedding
    
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

# Add dimension validation helper
def validate_embedding_dimensions(embeddings: List[List[float]], context: str = ""):
    """Validate that all embeddings have the correct dimension."""
    for i, embedding in enumerate(embeddings):
        if len(embedding) != VECTOR_DIM:
            raise ValueError(
                f"Embedding dimension mismatch at index {i} in {context}: "
                f"expected {VECTOR_DIM}, got {len(embedding)}"
            )

# FastAPI app
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Load FAISS indices on startup"""
    if vector_store.load_indices(FAISS_DATA_PATH):
        logger.info("FAISS indices loaded successfully")
        memory_stats = vector_store.get_memory_usage()
        logger.info(f"Memory usage: {memory_stats}")
    else:
        logger.info("No existing FAISS indices found, starting fresh")
    """Initialize database on startup."""
    if not init_db():
        logger.error("Failed to initialize database")
        raise Exception("Database initialization failed")

@app.on_event("shutdown")
async def shutdown_event():
    """Save FAISS indices on shutdown"""
    vector_store.save_indices(FAISS_DATA_PATH)
    logger.info("FAISS indices saved")

@app.post("/ingest", response_model=DocumentIngestionResponse)
async def ingest_document(
    file: UploadFile = File(...),
    metadata: str = Form(None)
):
    """Document Ingestion Mode: Process and store document chunks with embeddings."""
    request_id = str(uuid.uuid4())
    endpoint_logger = get_contextual_logger(request_id, "/ingest")
    start_time = time.time()
    
    # Initialize variables for logging
    doc_id = None
    job_id = None
    processing_steps = []
    
    try:
        # Log input parameters
        file_info = {
            'filename': file.filename,
            'content_type': file.content_type,
            'file_size': None  # Will be updated after reading
        }
        
        # Parse metadata
        metadata_dict = json.loads(metadata) if metadata else {}
        doc_id = metadata_dict.get('doc_id', file.filename)
        job_id = metadata_dict.get('job_id', f"job_{int(time.time())}")
        
        log_with_context(
            logger, 'info',
            f"INGEST_START - Processing document ingestion",
            request_id, "/ingest",
            file_info=file_info,
            metadata=metadata_dict,
            doc_id=doc_id,
            job_id=job_id
        )
        
        processing_steps.append(f"Started processing at {datetime.utcnow().isoformat()}")
        
        # Extract text from file
        endpoint_logger.info("INGEST_STEP - Extracting text from file")
        text = await extract_text_from_file(file)
        file_info['file_size'] = len(text.encode('utf-8'))
        processing_steps.append(f"Text extracted - {len(text)} characters")
        
        endpoint_logger.info(f"INGEST_STEP - Text extraction completed", extra={
            'text_length': len(text),
            'file_size_bytes': file_info['file_size']
        })
        
        # Normalize text (remove headers/footers)
        endpoint_logger.info("INGEST_STEP - Normalizing text")
        original_length = len(text)
        text = normalize_text(text)
        normalized_length = len(text)
        processing_steps.append(f"Text normalized - {original_length} -> {normalized_length} characters")
        
        endpoint_logger.info(f"INGEST_STEP - Text normalization completed", extra={
            'original_length': original_length,
            'normalized_length': normalized_length,
            'reduction_percent': round(((original_length - normalized_length) / original_length) * 100, 2)
        })
        
        # Split into chunks
        endpoint_logger.info("INGEST_STEP - Splitting text into chunks")
        chunks = split_text(text)
        processing_steps.append(f"Text split into {len(chunks)} chunks")
        
        endpoint_logger.info(f"INGEST_STEP - Text splitting completed", extra={
            'total_chunks': len(chunks),
            'avg_chunk_size': round(sum(len(chunk) for chunk in chunks) / len(chunks), 2) if chunks else 0,
            'chunk_config': {
                'chunk_size': CHUNK_SIZE,
                'chunk_overlap': CHUNK_OVERLAP
            }
        })
        
        # Update job status in Redis if available
        if job_id and job_id != 'unknown':
            try:
                redis_client.hset(
                    job_id,
                    mapping={
                        'status': 'processing',
                        'progress': '10',  # 10% - file processed
                        'chunks_processed': '0',
                        'total_chunks': str(len(chunks)),
                        'started_at': datetime.utcnow().isoformat()
                    }
                )
                endpoint_logger.info(f"INGEST_STEP - Updated Redis job status", extra={'job_id': job_id, 'progress': '10%'})
            except Exception as e:
                log_with_context(
                    logger, 'warning',
                    f"INGEST_WARNING - Failed to update Redis status for job {job_id}",
                    request_id, "/ingest",
                    error=str(e)
                )
        
        # Process chunks
        processed_chunks = []
        chunk_ids = []
        embeddings = []
        
        # Process in smaller batches for M2 Mac
        batch_size = 8  # Smaller batch size for M2
        
        endpoint_logger.info(f"INGEST_STEP - Starting chunk processing with batch size {batch_size}")
        
        # Store document metadata in database
        db = next(get_db())
        try:
            # Store document record
            endpoint_logger.info("INGEST_STEP - Storing document metadata in database")
            await db.execute(
                """
                    INSERT INTO documents 
                    (doc_id, filename, file_type, total_chunks, status, metadata)
                    VALUES 
                    ($1, $2, $3, $4, 'processing', $5)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        total_chunks = $4,
                        status = 'processing',
                        metadata = $5,
                        updated_at = NOW()
                """,
                doc_id, file.filename, file.content_type, len(chunks), json.dumps(metadata_dict)
            )
            processing_steps.append(f"Document metadata stored in database")
            endpoint_logger.info("INGEST_STEP - Document metadata stored successfully")
            
        except Exception as e:
            await db.close()
            error_msg = f"Failed to store document metadata: {str(e)}"
            processing_steps.append(f"ERROR: {error_msg}")
            log_with_context(
                logger, 'error',
                f"INGEST_ERROR - Database error during metadata storage",
                request_id, "/ingest",
                error=str(e),
                traceback=traceback.format_exc(),
                doc_id=doc_id
            )
            if job_id and job_id != 'unknown':
                try:
                    redis_client.hset(job_id, 'status', 'failed')
                    redis_client.hset(job_id, 'error', error_msg)
                except:
                    pass
            raise HTTPException(status_code=500, detail=error_msg)
        
        try:
            # Process chunks in batches
            endpoint_logger.info(f"INGEST_STEP - Processing {len(chunks)} chunks in batches of {batch_size}")
            
            for i in range(0, len(chunks), batch_size):
                batch_start_time = time.time()
                batch_chunks = chunks[i:i + batch_size]
                batch_texts = [chunk for chunk in batch_chunks]
                
                endpoint_logger.info(f"INGEST_STEP - Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}", extra={
                    'batch_start_index': i,
                    'batch_size': len(batch_chunks),
                    'total_chunks': len(chunks)
                })
                
                # Get embeddings for batch
                embeddings_start_time = time.time()
                batch_embeddings = await asyncio.gather(*[get_embedding(text) for text in batch_texts])
                embeddings_time = time.time() - embeddings_start_time
                
                # Validate embedding dimensions
                validate_embedding_dimensions(batch_embeddings, f"batch {i//batch_size + 1}")
                
                endpoint_logger.info(f"INGEST_STEP - Generated embeddings for batch", extra={
                    'batch_size': len(batch_chunks),
                    'embedding_time_seconds': round(embeddings_time, 3),
                    'embedding_dimensions': len(batch_embeddings[0]) if batch_embeddings else 0
                })
                
                # Store chunks metadata and prepare for FAISS
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                    chunk_id = f"{doc_id}_chunk_{i+j:04d}"
                    chunk_ids.append(chunk_id)
                    embeddings.append(embedding)
                    
                    # Store metadata in PostgreSQL
                    await db.execute(
                        """
                            INSERT INTO document_chunks 
                            (chunk_id, doc_id, chunk_text, chunk_index, page_number, section_title, faiss_index, metadata)
                            VALUES 
                            ($1, $2, $3, $4, $5, $6, $7, $8)
                            ON CONFLICT (chunk_id) DO UPDATE SET
                                chunk_text = $3,
                                chunk_index = $4,
                                page_number = $5,
                                section_title = $6,
                                faiss_index = $7,
                                metadata = $8,
                                updated_at = NOW()
                        """,
                        chunk_id, 
                        doc_id, 
                        chunk, 
                        i + j, 
                        i // 3 + 1,  # Simple page estimation
                        "",  # section_title
                        vector_store.document_index.ntotal + len(chunk_ids) - 1,
                        json.dumps(metadata_dict.get('chunk_metadata', {}))
                    )
                    
                    processed_chunks.append({
                        'chunk_id': chunk_id,
                        'text': chunk[:100] + '...' if len(chunk) > 100 else chunk,  # Return preview
                        'doc_id': doc_id,
                        'page': i // 3 + 1,
                        'embedding_stored': True
                    })
                
                batch_time = time.time() - batch_start_time
                
                # Update job progress in Redis
                if job_id and job_id != 'unknown':
                    try:
                        progress = min(100, int((i + len(batch_chunks)) / len(chunks) * 100))
                        redis_client.hset(
                            job_id,
                            mapping={
                                'status': 'processing',
                                'progress': str(progress),
                                'chunks_processed': str(min(i + len(batch_chunks), len(chunks))),
                                'total_chunks': str(len(chunks))
                            }
                        )
                        endpoint_logger.info(f"INGEST_STEP - Updated job progress", extra={
                            'job_id': job_id,
                            'progress': f"{progress}%",
                            'chunks_processed': min(i + len(batch_chunks), len(chunks))
                        })
                    except Exception as e:
                        log_with_context(
                            logger, 'warning',
                            f"INGEST_WARNING - Failed to update Redis progress for job {job_id}",
                            request_id, "/ingest",
                            error=str(e)
                        )
                
                endpoint_logger.info(f"INGEST_STEP - Completed batch processing", extra={
                    'batch_number': i//batch_size + 1,
                    'batch_time_seconds': round(batch_time, 3),
                    'chunks_in_batch': len(batch_chunks)
                })
            
            # Add embeddings to FAISS with M2-optimized batching
            if embeddings:
                endpoint_logger.info(f"INGEST_STEP - Adding {len(embeddings)} embeddings to FAISS index")
                faiss_start_time = time.time()
                vector_store.add_documents(embeddings, chunk_ids, batch_size=16)
                faiss_time = time.time() - faiss_start_time
                processing_steps.append(f"Added {len(embeddings)} embeddings to FAISS index")
                
                endpoint_logger.info(f"INGEST_STEP - FAISS index updated successfully", extra={
                    'embeddings_added': len(embeddings),
                    'faiss_update_time_seconds': round(faiss_time, 3)
                })
                
                # Update document status to completed
                await db.execute(
                    """
                        UPDATE documents 
                        SET status = 'completed', 
                            processed_at = NOW(),
                            total_chunks = $1
                        WHERE doc_id = $2
                    """,
                    len(chunk_ids), doc_id
                )
                processing_steps.append(f"Document status updated to completed")
                endpoint_logger.info("INGEST_STEP - Document status updated to completed")
            else:
                await db.execute(
                    """
                        UPDATE documents 
                        SET status = 'failed',
                            error = 'No chunks processed',
                            updated_at = NOW()
                        WHERE doc_id = $1
                    """,
                    doc_id
                )
                error_msg = "No chunks were processed from the document"
                processing_steps.append(f"ERROR: {error_msg}")
                log_with_context(
                    logger, 'error',
                    f"INGEST_ERROR - {error_msg}",
                    request_id, "/ingest",
                    doc_id=doc_id
                )
                raise HTTPException(status_code=400, detail=error_msg)
                
        except Exception as e:
            # Update document status to failed
            await db.execute(
                """
                    UPDATE documents 
                    SET status = 'failed',
                        error = $1,
                        updated_at = NOW()
                    WHERE doc_id = $2
                """,
                str(e), doc_id
            )
            # Update job status in Redis if job_id is provided
            if job_id and job_id != 'unknown':
                try:
                    redis_client.hset(job_id, 'status', 'failed')
                    redis_client.hset(job_id, 'error', str(e))
                except:
                    pass
            
            error_msg = f"Failed to process document: {str(e)}"
            processing_steps.append(f"ERROR: {error_msg}")
            log_with_context(
                logger, 'error',
                f"INGEST_ERROR - Document processing failed",
                request_id, "/ingest",
                error=str(e),
                traceback=traceback.format_exc(),
                doc_id=doc_id,
                processing_steps=processing_steps
            )
            raise HTTPException(status_code=500, detail=error_msg)
        
        finally:
            await db.close()
        
        # Save FAISS indices
        endpoint_logger.info("INGEST_STEP - Saving FAISS indices")
        save_start_time = time.time()
        vector_store.save_indices(FAISS_DATA_PATH)
        save_time = time.time() - save_start_time
        processing_steps.append(f"FAISS indices saved in {save_time:.2f} seconds")
        
        processing_time = time.time() - start_time
        
        # Update job status in Redis if job_id is provided
        if job_id and job_id != 'unknown':
            try:
                redis_client.hset(
                    job_id,
                    mapping={
                        'status': 'completed',
                        'progress': '100',
                        'chunks_processed': str(len(processed_chunks)),
                        'total_chunks': str(len(processed_chunks)),
                        'completed_at': datetime.utcnow().isoformat(),
                        'processing_time': f"{processing_time:.2f} seconds"
                    }
                )
                endpoint_logger.info(f"INGEST_STEP - Updated job status to completed", extra={'job_id': job_id})
            except Exception as e:
                log_with_context(
                    logger, 'warning',
                    f"INGEST_WARNING - Failed to update Redis completion status for job {job_id}",
                    request_id, "/ingest",
                    error=str(e)
                )
        
        # Prepare response
        response = DocumentIngestionResponse(
            status="success",
            chunks_created=len(processed_chunks),
            doc_id=doc_id,
            faiss_index_updated=True,
            chunks=processed_chunks
        )
        
        # Log successful completion
        log_with_context(
            logger, 'info',
            f"INGEST_SUCCESS - Document ingestion completed successfully",
            request_id, "/ingest",
            doc_id=doc_id,
            chunks_created=len(processed_chunks),
            processing_time_seconds=round(processing_time, 3),
            file_info=file_info,
            processing_steps=processing_steps,
            response_summary={
                'status': response.status,
                'chunks_created': response.chunks_created,
                'faiss_index_updated': response.faiss_index_updated
            }
        )
        
        return response
    
    except HTTPException as he:
        log_with_context(
            logger, 'error',
            f"INGEST_HTTP_ERROR - HTTP exception occurred",
            request_id, "/ingest",
            error_detail=str(he.detail) if hasattr(he, 'detail') else str(he),
            status_code=getattr(he, 'status_code', 'unknown'),
            doc_id=doc_id,
            job_id=job_id,
            processing_steps=processing_steps,
            processing_time_seconds=round(time.time() - start_time, 3)
        )
        raise he
    except Exception as e:
        error_msg = f"Error in document ingestion: {str(e)}"
        processing_steps.append(f"FATAL ERROR: {error_msg}")
        
        log_with_context(
            logger, 'error',
            f"INGEST_FATAL_ERROR - Unexpected error occurred",
            request_id, "/ingest",
            error=str(e),
            traceback=traceback.format_exc(),
            doc_id=doc_id,
            job_id=job_id,
            processing_steps=processing_steps,
            processing_time_seconds=round(time.time() - start_time, 3)
        )
        
        # Update job status in Redis if job_id is provided
        if job_id and job_id != 'unknown':
            try:
                redis_client.hset(job_id, 'status', 'failed')
                redis_client.hset(job_id, 'error', error_msg)
            except Exception as redis_err:
                log_with_context(
                    logger, 'error',
                    f"INGEST_REDIS_ERROR - Failed to update Redis error status",
                    request_id, "/ingest",
                    redis_error=str(redis_err)
                )
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/search", response_model=List[DocumentSearchResult])
async def search_documents(request: DocumentSearchRequest):
    """Document/Web Content Retrieval Mode: Search for relevant document chunks."""
    request_id = str(uuid.uuid4())
    endpoint_logger = get_contextual_logger(request_id, "/search")
    start_time = time.time()
    
    try:
        # Log input parameters
        log_with_context(
            logger, 'info',
            f"SEARCH_START - Starting document search",
            request_id, "/search",
            input_params={
                'query': request.query,
                'top_k': request.top_k,
                'query_length': len(request.query),
                'threshold': DOC_SIMILARITY_THRESHOLD
            }
        )
        
        # Get query embedding
        endpoint_logger.info("SEARCH_STEP - Generating query embedding")
        embedding_start_time = time.time()
        query_embedding = await get_embedding(request.query)
        embedding_time = time.time() - embedding_start_time
        
        # Validate query embedding dimension
        if len(query_embedding) != VECTOR_DIM:
            raise HTTPException(
                status_code=500, 
                detail=f"Query embedding dimension mismatch: expected {VECTOR_DIM}, got {len(query_embedding)}"
            )
        
        endpoint_logger.info(f"SEARCH_STEP - Query embedding generated", extra={
            'embedding_time_seconds': round(embedding_time, 3),
            'embedding_dimension': len(query_embedding)
        })
        
        # Search in FAISS index
        endpoint_logger.info("SEARCH_STEP - Searching FAISS index")
        faiss_start_time = time.time()
        results = vector_store.search_documents(
            query_embedding,
            top_k=request.top_k,
            threshold=DOC_SIMILARITY_THRESHOLD
        )
        faiss_time = time.time() - faiss_start_time
        
        endpoint_logger.info(f"SEARCH_STEP - FAISS search completed", extra={
            'faiss_search_time_seconds': round(faiss_time, 3),
            'raw_results_count': len(results),
            'requested_top_k': request.top_k
        })
        
        if not results:
            endpoint_logger.info("SEARCH_RESULT - No results found", extra={
                'query': request.query,
                'threshold': DOC_SIMILARITY_THRESHOLD,
                'total_time_seconds': round(time.time() - start_time, 3)
            })
            return []
        
        # Get chunk metadata from database
        endpoint_logger.info("SEARCH_STEP - Retrieving chunk metadata from database")
        db_start_time = time.time()
        db = next(get_db())
        try:
            chunk_ids = [r['chunk_id'] for r in results]
            
            endpoint_logger.info(f"SEARCH_STEP - Querying database for {len(chunk_ids)} chunks")
            
            # Get metadata for all chunks in one query
            chunk_data = await db.execute(
                """
                SELECT 
                    chunk_id,
                    chunk_text,
                    doc_id,
                    page_number,
                    section_title,
                    content_type,
                    source_url,
                    domain
                FROM document_chunks 
                WHERE chunk_id = ANY($1::text[])
                """,
                tuple(chunk_ids)
            )
            
            db_time = time.time() - db_start_time
            endpoint_logger.info(f"SEARCH_STEP - Database query completed", extra={
                'db_query_time_seconds': round(db_time, 3),
                'chunks_found_in_db': len(chunk_data)
            })
            
            # Create mapping of chunk_id to metadata
            metadata_map = {row['chunk_id']: row for row in chunk_data}
            
            # Combine FAISS results with metadata
            formatting_start_time = time.time()
            formatted_results = []
            for result in results:
                chunk_id = result['chunk_id']
                if chunk_id in metadata_map:
                    metadata = metadata_map[chunk_id]
                    formatted_results.append(DocumentSearchResult(
                        text=metadata['chunk_text'],
                        source=metadata['doc_id'] or metadata.get('domain', 'unknown'),
                        score=result['score'],
                        chunk_id=chunk_id,
                        page=metadata['page_number']
                    ))
            
            formatting_time = time.time() - formatting_start_time
            total_time = time.time() - start_time
            
            # Log successful completion
            log_with_context(
                logger, 'info',
                f"SEARCH_SUCCESS - Search completed successfully",
                request_id, "/search",
                query=request.query,
                results_returned=len(formatted_results),
                requested_top_k=request.top_k,
                timing={
                    'total_time_seconds': round(total_time, 3),
                    'embedding_time_seconds': round(embedding_time, 3),
                    'faiss_time_seconds': round(faiss_time, 3),
                    'db_time_seconds': round(db_time, 3),
                    'formatting_time_seconds': round(formatting_time, 3)
                },
                result_scores=[round(r.score, 3) for r in formatted_results[:5]],  # Top 5 scores
                result_sources=list(set([r.source for r in formatted_results]))
            )
            
            return formatted_results
            
        finally:
            await db.close()
        
    except Exception as e:
        total_time = time.time() - start_time
        log_with_context(
            logger, 'error', 
            f"SEARCH_ERROR - Search failed",
            request_id, "/search",
            error=str(e),
            traceback=traceback.format_exc(),
            query=request.query,
            top_k=request.top_k,
            total_time_seconds=round(total_time, 3)
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat-context", response_model=List[ChatContextResult])
async def get_chat_context(request: ChatContextRequest):
    """Chat Context Retrieval Mode: Get relevant chat history."""
    request_id = str(uuid.uuid4())
    endpoint_logger = get_contextual_logger(request_id, "/chat-context")
    start_time = time.time()
    
    try:
        # Log input parameters
        log_with_context(
            logger, 'info',
            f"CHAT_CONTEXT_START - Starting chat context retrieval",
            request_id, "/chat-context",
            input_params={
                'query': request.query,
                'session_id': request.session_id,
                'context_type': request.context_type,
                'top_k': request.top_k,
                'query_length': len(request.query),
                'threshold': CHAT_SIMILARITY_THRESHOLD
            }
        )
        
        # Get query embedding
        endpoint_logger.info("CHAT_CONTEXT_STEP - Generating query embedding")
        embedding_start_time = time.time()
        query_embedding = await get_embedding(request.query)
        embedding_time = time.time() - embedding_start_time
        
        # Validate query embedding dimension
        if len(query_embedding) != VECTOR_DIM:
            raise HTTPException(
                status_code=500, 
                detail=f"Query embedding dimension mismatch: expected {VECTOR_DIM}, got {len(query_embedding)}"
            )
        
        endpoint_logger.info(f"CHAT_CONTEXT_STEP - Query embedding generated", extra={
            'embedding_time_seconds': round(embedding_time, 3),
            'embedding_dimension': len(query_embedding)
        })
        
        # Search in FAISS index
        endpoint_logger.info("CHAT_CONTEXT_STEP - Searching FAISS index for chat messages")
        faiss_start_time = time.time()
        results = vector_store.search_chat_messages(
            query_embedding,
            top_k=request.top_k,
            threshold=CHAT_SIMILARITY_THRESHOLD
        )
        faiss_time = time.time() - faiss_start_time
        
        endpoint_logger.info(f"CHAT_CONTEXT_STEP - FAISS search completed", extra={
            'faiss_search_time_seconds': round(faiss_time, 3),
            'raw_results_count': len(results),
            'requested_top_k': request.top_k,
            'session_id': request.session_id
        })
        
        if not results:
            endpoint_logger.info("CHAT_CONTEXT_RESULT - No chat context found", extra={
                'query': request.query,
                'session_id': request.session_id,
                'threshold': CHAT_SIMILARITY_THRESHOLD,
                'total_time_seconds': round(time.time() - start_time, 3)
            })
            return []
        
        # Get message metadata from database
        endpoint_logger.info("CHAT_CONTEXT_STEP - Retrieving message metadata from database")
        db_start_time = time.time()
        db = next(get_db())
        try:
            message_ids = [r['message_id'] for r in results]
            
            endpoint_logger.info(f"CHAT_CONTEXT_STEP - Querying database for {len(message_ids)} messages")
            
            # Get metadata for all messages in one query
            message_data = await db.execute(
                """
                SELECT 
                    message_id,
                    message_text,
                    session_id,
                    message_type,
                    timestamp,
                    metadata
                FROM chat_messages 
                WHERE message_id = ANY($1::text[])
                ORDER BY timestamp DESC
                """,
                tuple(message_ids)
            )
            
            db_time = time.time() - db_start_time
            endpoint_logger.info(f"CHAT_CONTEXT_STEP - Database query completed", extra={
                'db_query_time_seconds': round(db_time, 3),
                'messages_found_in_db': len(message_data)
            })
            
            # Create mapping of message_id to metadata
            metadata_map = {row['message_id']: row for row in message_data}
            
            # Combine FAISS results with metadata
            formatting_start_time = time.time()
            formatted_results = []
            for result in results:
                message_id = result['message_id']
                if message_id in metadata_map:
                    metadata = metadata_map[message_id]
                    formatted_results.append(ChatContextResult(
                        message=metadata['message_text'],
                        timestamp=metadata['timestamp'].isoformat(),
                        score=result['score'],
                        message_type=metadata['message_type']
                    ))
            
            formatting_time = time.time() - formatting_start_time
            total_time = time.time() - start_time
            
            # Log successful completion
            log_with_context(
                logger, 'info',
                f"CHAT_CONTEXT_SUCCESS - Chat context retrieval completed successfully",
                request_id, "/chat-context",
                query=request.query,
                session_id=request.session_id,
                context_type=request.context_type,
                results_returned=len(formatted_results),
                requested_top_k=request.top_k,
                timing={
                    'total_time_seconds': round(total_time, 3),
                    'embedding_time_seconds': round(embedding_time, 3),
                    'faiss_time_seconds': round(faiss_time, 3),
                    'db_time_seconds': round(db_time, 3),
                    'formatting_time_seconds': round(formatting_time, 3)
                },
                result_scores=[round(r.score, 3) for r in formatted_results[:5]],  # Top 5 scores
                message_types=list(set([r.message_type for r in formatted_results]))
            )
            
            return formatted_results
            
        finally:
            await db.close()
    
    except Exception as e:
        total_time = time.time() - start_time
        log_with_context(
            logger, 'error',
            f"CHAT_CONTEXT_ERROR - Chat context retrieval failed",
            request_id, "/chat-context",
            error=str(e),
            traceback=traceback.format_exc(),
            query=request.query,
            session_id=request.session_id,
            context_type=request.context_type,
            top_k=request.top_k,
            total_time_seconds=round(total_time, 3)
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crawl-stats/{job_id}", response_model=CrawlJobStats)
async def get_crawl_stats(job_id: str):
    """Get crawl job statistics."""
    try:
        db = next(get_db())
        try:
            # Get crawl job statistics
            stats = await db.execute(
                """
                WITH page_stats AS (
                    SELECT 
                        COUNT(*) as total_pages,
                        COUNT(*) FILTER (WHERE status = 'processed') as processed_pages,
                        COUNT(*) FILTER (WHERE status = 'failed') as failed_pages,
                        COUNT(*) FILTER (WHERE status = 'pending') as pending_pages,
                        AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_processing_time
                    FROM web_pages
                    WHERE crawl_job_id = $1
                )
                SELECT 
                    ps.*,
                    cj.created_at as start_time,
                    cj.updated_at as last_update
                FROM page_stats ps
                JOIN crawl_jobs cj ON cj.job_id = $1
                """,
                job_id
            )
            
            if not stats:
                raise HTTPException(status_code=404, detail="Crawl job not found")
            
            return CrawlJobStats(
                total_pages=stats['total_pages'],
                processed_pages=stats['processed_pages'],
                failed_pages=stats['failed_pages'],
                pending_pages=stats['pending_pages'],
                avg_processing_time=stats['avg_processing_time'] or 0.0,
                start_time=stats['start_time'],
                last_update=stats['last_update']
            )
        finally:
            await db.close()
        
    except Exception as e:
        logger.error(f"Error getting crawl stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/content-distribution", response_model=List[ContentDistribution])
async def get_content_distribution():
    """Get content distribution by source."""
    try:
        db = next(get_db())
        try:
            # Get content distribution statistics
            distribution = await db.execute(
                """
                WITH source_stats AS (
                    SELECT 
                        COALESCE(doc_id, domain) as source,
                        COUNT(*) as total_chunks,
                        AVG(LENGTH(chunk_text)) as avg_chunk_size,
                        SUM((metadata->>'token_count')::int) as total_tokens,
                        MAX(updated_at) as last_updated
                    FROM document_chunks
                    WHERE metadata->>'token_count' IS NOT NULL
                    GROUP BY COALESCE(doc_id, domain)
                )
                SELECT 
                    source,
                    total_chunks,
                    avg_chunk_size,
                    COALESCE(total_tokens, 0) as total_tokens,
                    last_updated
                FROM source_stats
                ORDER BY total_chunks DESC
                """
            )
            
            return [
                ContentDistribution(
                    source=row['source'],
                    total_chunks=row['total_chunks'],
                    avg_chunk_size=row['avg_chunk_size'],
                    total_tokens=row['total_tokens'],
                    last_updated=row['last_updated']
                )
                for row in distribution
            ]
        finally:
            await db.close()

    except Exception as e:
        logger.error(f"Error getting content distribution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

#--
# Enhanced /embed endpoint with comprehensive functionality
@app.post("/embed", response_model=EmbedResponse)
async def embed_endpoint(request: EmbedRequest):
    """
    Enhanced embedding generation endpoint for Vector Service Container.
    
    Supports both single text and batch text embedding generation with:
    - Input validation and sanitization
    - M2 Mac memory optimization
    - Batch processing with configurable batch size
    - Optional embedding normalization
    - Comprehensive error handling
    - Processing metadata and timing information
    - Proper resource cleanup
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Set up logging context
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"EMBED_REQUEST - Starting embedding generation", extra={
            "request_id": request_id,
            "num_texts": len(request.texts),
            "batch_size": request.batch_size,
            "normalize": request.normalize_embeddings,
            "include_metadata": request.include_metadata
        })
        
        # Input validation
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        # Generate embeddings
        result = await embed_texts_batch_enhanced(
            texts=request.texts,
            batch_size=request.batch_size,
            normalize=request.normalize_embeddings,
            include_metadata=request.include_metadata
        )
        
        # Validate embedding dimensions
        embeddings = result["embeddings"]
        if embeddings:
            expected_dim = 384  # BGE-small dimension
            actual_dim = len(embeddings[0])
            if actual_dim != expected_dim:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}"
                )
        
        # Prepare response
        response = EmbedResponse(
            embeddings=embeddings,
            model=model_name,
            dimension=len(embeddings[0]) if embeddings else 0,
            processing_time_seconds=result["processing_time_seconds"],
            metadata=result.get("metadata") if request.include_metadata else None
        )
        
        logger.info(f"EMBED_SUCCESS - Embedding generation completed", extra={
            "request_id": request_id,
            "num_embeddings": len(embeddings),
            "dimension": response.dimension,
            "processing_time_seconds": response.processing_time_seconds,
            "total_time_seconds": round(time.time() - start_time, 3)
        })
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log error and return 500
        error_msg = f"Embedding generation failed: {str(e)}"
        logger.error(f"EMBED_ERROR - {error_msg}", extra={
            "request_id": request_id,
            "error": str(e),
            "processing_time_seconds": round(time.time() - start_time, 3)
        })
        raise HTTPException(status_code=500, detail=error_msg)

# Health check enhancement for embedding service
@app.get("/embed/health")
async def embed_health_check():
    """Health check specifically for embedding functionality."""
    try:
        # Test embedding generation with a simple text
        test_text = "This is a test sentence for health check."
        test_embedding = await embed_text_enhanced(test_text)
        
        return {
            "status": "healthy",
            "embedding_service": "operational",
            "model": model_name,
            "device": str(device),
            "dimension": len(test_embedding),
            "test_embedding_generated": True,
            "torch_mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "embedding_service": "failed",
            "error": str(e)
        }

# Utility endpoint to get embedding model information
@app.get("/embed/info")
async def embed_model_info():
    """Get information about the embedding model and configuration."""
    return {
        "model_name": model_name,
        "dimension": 384,  # BGE-small dimension
        "max_sequence_length": 512,
        "device": str(device),
        "batch_size_limit": 32,
        "supported_features": [
            "single_text_embedding",
            "batch_text_embedding", 
            "embedding_normalization",
            "processing_metadata",
            "m2_optimization"
        ],
        "hardware_optimization": {
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            "current_device": str(device),
            "memory_optimized": True
        }
    }
#--

@app.get("/health")
async def health_check():
    """Health check endpoint with memory usage info"""
    try:
        # Check database connection
        db = next(get_db())
        try:
            await db.execute("SELECT 1")
        finally:
            await db.close()
        
        # Get FAISS memory usage
        memory_stats = vector_store.get_memory_usage()
        
        # Test embedding generation
        test_embedding = await get_embedding("health check test")
        
        return {
            "status": "healthy",
            "database": "connected",
            "faiss": memory_stats,
            "redis": "connected" if redis_client.ping() else "disconnected",
            "model_info": {
                "model_name": "BAAI/bge-small-en",
                "expected_dimension": VECTOR_DIM,
                "actual_dimension": len(test_embedding),
                "dimension_valid": len(test_embedding) == VECTOR_DIM
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/memory")
async def memory_status():
    """Endpoint to monitor memory usage on M2 Mac"""
    memory_stats = vector_store.get_memory_usage()
    return memory_stats