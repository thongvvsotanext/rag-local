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
from urllib.parse import urlparse
from faiss_store import M2OptimizedFAISSStore
from functools import lru_cache  # Added missing import
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, Text, text, inspect, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('vector_retriever.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
# Build database URL from individual environment variables
DB_HOST = os.getenv("POSTGRES_HOST", "db")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_USER = os.getenv("POSTGRES_USER", "fizen_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fizen_password")
DB_NAME = os.getenv("POSTGRES_DB", "fizen_rag")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
VECTOR_DIM = 512  # BGE-small dimension
EMBEDDING_DIMENSIONS = 512  # BGE-small dimension
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

# Initialize FAISS store
FAISS_DATA_PATH = os.getenv("FAISS_DATA_PATH", "/app/data/faiss")
vector_store = M2OptimizedFAISSStore(dimension=VECTOR_DIM, index_type="Flat")  # Use Flat for M2 Mac

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

@lru_cache(maxsize=1000)
async def get_embedding(text: str) -> List[float]:
    """Get embedding for text with caching and validation."""
    try:
        # Check Redis cache first
        cache_key = f"embedding:{hash(text)}"
        cached = redis_client.get(cache_key)
        if cached:
            embedding = json.loads(cached)
            return embedding
        
        # Generate embedding
        embedding = model.encode(text).tolist()
        
        # Cache the embedding
        redis_client.setex(cache_key, 3600, json.dumps(embedding))
        return embedding
    
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
    try:
        start_time = time.time()
        
        # Parse metadata
        metadata_dict = json.loads(metadata) if metadata else {}
        doc_id = metadata_dict.get('doc_id', file.filename)
        job_id = metadata_dict.get('job_id', f"job_{int(time.time())}")
        
        # Log the start of processing
        logger.info(f"Starting document ingestion for {doc_id} (job: {job_id})")
        
        # Extract text from file
        text = await extract_text_from_file(file)
        
        # Normalize text (remove headers/footers)
        text = normalize_text(text)
        
        # Split into chunks
        chunks = split_text(text)
        
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
            except Exception as e:
                logger.warning(f"Failed to update Redis status for job {job_id}: {str(e)}")
        
        # Process chunks
        processed_chunks = []
        chunk_ids = []
        embeddings = []
        
        # Process in smaller batches for M2 Mac
        batch_size = 8  # Smaller batch size for M2
        
        # Store document metadata in database
        db = next(get_db())
        try:
            # Store document record
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
        except Exception as e:
            await db.close()
            error_msg = f"Failed to store document metadata: {str(e)}"
            logger.error(error_msg)
            if job_id and job_id != 'unknown':
                try:
                    redis_client.hset(job_id, 'status', 'failed')
                    redis_client.hset(job_id, 'error', error_msg)
                except:
                    pass
            raise HTTPException(status_code=500, detail=error_msg)
        try:
            # Process chunks in batches
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_texts = [chunk for chunk in batch_chunks]
                
                # Get embeddings for batch
                batch_embeddings = await asyncio.gather(*[get_embedding(text) for text in batch_texts])
                
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
                    except Exception as e:
                        logger.warning(f"Failed to update Redis progress for job {job_id}: {str(e)}")
            
            # Add embeddings to FAISS with M2-optimized batching
            if embeddings:
                vector_store.add_documents(embeddings, chunk_ids, batch_size=16)
                
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
                raise HTTPException(status_code=400, detail="No chunks were processed from the document")
                
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
            raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
        
        finally:
            await db.close()
        
        # Save FAISS indices
        vector_store.save_indices(FAISS_DATA_PATH)
        
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
            except Exception as e:
                logger.warning(f"Failed to update Redis completion status for job {job_id}: {str(e)}")
        
        logger.info(f"Successfully processed document {doc_id} in {processing_time:.2f}s. Created {len(processed_chunks)} chunks.")
        
        return DocumentIngestionResponse(
            status="success",
            chunks_created=len(processed_chunks),
            doc_id=doc_id,
            faiss_index_updated=True,
            chunks=processed_chunks
        )
    
    except HTTPException as he:
        logger.error(f"HTTP error in document ingestion: {str(he.detail) if hasattr(he, 'detail') else str(he)}")
        raise he
    except Exception as e:
        error_msg = f"Error in document ingestion: {str(e)}"
        logger.error(error_msg, exc_info=True)
        # Update job status in Redis if job_id is provided
        if job_id and job_id != 'unknown':
            try:
                redis_client.hset(job_id, 'status', 'failed')
                redis_client.hset(job_id, 'error', error_msg)
            except Exception as redis_err:
                logger.error(f"Failed to update Redis error status: {str(redis_err)}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/search", response_model=List[DocumentSearchResult])
async def search_documents(request: DocumentSearchRequest):
    """Document/Web Content Retrieval Mode: Search for relevant document chunks."""
    try:
        # Get query embedding
        query_embedding = await get_embedding(request.query)
        
        # Search in FAISS index
        results = vector_store.search_documents(
            query_embedding,
            top_k=request.top_k,
            threshold=DOC_SIMILARITY_THRESHOLD
        )
        
        if not results:
            return []
        
        # Get chunk metadata from database
        db = next(get_db())
        try:
            chunk_ids = [r['chunk_id'] for r in results]
            
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
            
            # Create mapping of chunk_id to metadata
            metadata_map = {row['chunk_id']: row for row in chunk_data}
            
            # Combine FAISS results with metadata
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
            
            return formatted_results
        finally:
            await db.close()
        
    except Exception as e:
        logger.error(f"Error in document search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat-context", response_model=List[ChatContextResult])
async def get_chat_context(request: ChatContextRequest):
    """Chat Context Retrieval Mode: Get relevant chat history."""
    try:
        # Get query embedding
        query_embedding = await get_embedding(request.query)
        
        # Search in FAISS index
        results = vector_store.search_chat_messages(
            query_embedding,
            top_k=request.top_k,
            threshold=CHAT_SIMILARITY_THRESHOLD
        )
        
        if not results:
            return []
        
        # Get message metadata from database
        db = next(get_db())
        try:
            message_ids = [r['message_id'] for r in results]
            
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
            
            # Create mapping of message_id to metadata
            metadata_map = {row['message_id']: row for row in message_data}
            
            # Combine FAISS results with metadata
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
            
            return formatted_results
        finally:
            await db.close()
    
    except Exception as e:
        logger.error(f"Error in chat context retrieval: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
        
        return {
            "status": "healthy",
            "database": "connected",
            "faiss": memory_stats,
            "redis": "connected" if redis_client.ping() else "disconnected"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/memory")
async def memory_status():
    """Endpoint to monitor memory usage on M2 Mac"""
    memory_stats = vector_store.get_memory_usage()
    return memory_stats

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