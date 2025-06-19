from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import os
from dotenv import load_dotenv
import aiohttp
import logging
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, Text, text, inspect, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.exc import SQLAlchemyError  # Added missing import
import redis
import json
from sentence_transformers import SentenceTransformer
import sys
import asyncio
from contextlib import asynccontextmanager
from sqlalchemy.dialects.postgresql import JSONB

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('context_manager.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
VECTOR_SERVICE_URL = os.getenv("VECTOR_SERVICE_URL", "http://vector-service:8003")
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "50"))
TOP_K_CONTEXT = int(os.getenv("TOP_K_CONTEXT", "3"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Configuration
# Build database URL from individual environment variables
DB_HOST = os.getenv("POSTGRES_HOST", "db")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_USER = os.getenv("POSTGRES_USER", "fizen_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fizen_password")
DB_NAME = os.getenv("POSTGRES_DB", "fizen_rag")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

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
        required_tables = ["chat_sessions", "chat_messages"]
        
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

# Redis setup
redis_client = redis.from_url(REDIS_URL)

# Initialize BGE model
logger.info("Initializing BGE model")
model = SentenceTransformer('BAAI/bge-small-en')
logger.info("BGE model initialized successfully")

# Database Models
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, nullable=False)
    session_type = Column(String(50), default="anonymous")
    ip_address = Column(String(50))
    user_agent = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), server_default=text("now() + interval '1 hour'"))

    # Relationships
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), ForeignKey("chat_sessions.session_id", ondelete="CASCADE"), nullable=False)
    message_text = Column(Text, nullable=False)
    message_type = Column(String(50), nullable=False)
    user_id = Column(String(255), nullable=True)  # Added missing field
    embedding = Column(JSONB, nullable=True)  # Added missing field
    faiss_index = Column(Integer)
    response_time_ms = Column(Integer)
    sources_used = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    session = relationship("ChatSession", back_populates="messages")

# Input/Output Models
class ContextRequest(BaseModel):
    query: str
    session_id: str
    session_type: str = "anonymous"

class ContextSource(BaseModel):
    message: str
    timestamp: str
    similarity_score: float

class ContextResponse(BaseModel):
    query: str
    context: str
    session_type: str
    context_sources: List[ContextSource]

class EmbeddingRequest(BaseModel):
    texts: Union[str, List[str]]

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

async def get_embeddings(texts: Union[str, List[str]]) -> List[List[float]]:
    """Get embeddings from Vector Storage & Retrieval Service with caching."""
    # Determine cache key based on input type
    if isinstance(texts, str):
        cache_key = f"embeddings:{hash(texts)}"
    else:
        # Ensure list is hashable for caching
        cache_key = f"embeddings:{hash(tuple(texts))}"

    # Try to get from Redis cache first
    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Redis cache error: {str(e)}")
    
    try:
        async with aiohttp.ClientSession() as session:
            # The request payload is a simple dictionary
            payload = {"texts": texts}
            async with session.post(f"{VECTOR_SERVICE_URL}/embed", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to get embeddings: {response.status} - {error_text}")
                    return []
                
                data = await response.json()
                # The service response is an EmbedResponse model, we just need the embeddings
                embeddings = data.get("embeddings", [])
                
                # Cache the successful response
                try:
                    redis_client.setex(
                        cache_key,
                        3600,  # Cache for 1 hour
                        json.dumps(embeddings)
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache embeddings: {str(e)}")
                
                return embeddings
    except Exception as e:
        logger.error(f"Error getting embeddings: {str(e)}", exc_info=True)
        return []

def get_embedding(text: str) -> List[float]:
    """Get single embedding using the local BGE model."""
    try:
        embedding = model.encode([text])[0]
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return []

async def get_relevant_messages(query: str, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get relevant messages using embedding similarity."""
    try:
        if not history:
            return []
            
        # Get query embedding
        query_embedding = get_embedding(query)
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []
        
        # Calculate similarities
        similarities = []
        for msg in history:
            if msg.get("message_type") == "user":  # Only compare with user messages
                msg_embedding = get_embedding(msg["message"])
                if msg_embedding:
                    similarity = cosine_similarity(query_embedding, msg_embedding)
                    similarities.append((msg, similarity))
        
        # Sort by similarity and get top 3
        similarities.sort(key=lambda x: x[1], reverse=True)
        relevant_messages = [
            {
                "message": msg["message"],
                "timestamp": msg["timestamp"],
                "similarity_score": float(score)
            }
            for msg, score in similarities[:3] if score > SIMILARITY_THRESHOLD
        ]
        
        return relevant_messages
        
    except Exception as e:
        logger.error(f"Error getting relevant messages: {str(e)}")
        return []

def format_context(relevant_messages: List[Dict[str, Any]]) -> str:
    """Format relevant messages into a context block."""
    if not relevant_messages:
        return ""
    
    # Sort messages by timestamp
    sorted_messages = sorted(relevant_messages, key=lambda x: x.get("timestamp", ""))
    
    # Format each message with timestamp
    formatted_messages = []
    for msg in sorted_messages:
        timestamp = msg.get("timestamp", "")
        try:
            # Parse and format timestamp
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            timestamp_str = dt.strftime("%H:%M:%S")
        except:
            timestamp_str = "Unknown"
        
        formatted_messages.append(f"[{timestamp_str}] {msg.get('message', '')}")
    
    return "\n".join(formatted_messages)

async def get_session_history(db: Session, session_id: str) -> List[Dict[str, Any]]:
    """Get message history for a session."""
    try:
        # Get messages from database
        messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.created_at.desc()).limit(MAX_HISTORY_MESSAGES).all()
        
        # Convert to list of dicts
        return [
            {
                "message": msg.message_text,
                "timestamp": msg.created_at.isoformat(),
                "message_type": msg.message_type,
                "sources_used": msg.sources_used
            }
            for msg in messages
        ]
        
    except Exception as e:
        logger.error(f"Error getting session history: {str(e)}")
        return []

async def create_or_update_session(
    db: Session,
    session_id: str,
    session_type: str,
    req: Request
) -> ChatSession:
    """Create or update a chat session."""
    try:
        # Check if session exists
        session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
        
        if session:
            # Update existing session
            session.last_activity = datetime.utcnow()
            session.expires_at = datetime.utcnow() + timedelta(hours=24)
            db.commit()
            return session
        
        # Create new session
        session = ChatSession(
            session_id=session_id,
            session_type=session_type,
            ip_address=req.client.host,
            user_agent=req.headers.get("user-agent", ""),
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=24)
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        return session
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating/updating session: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI app."""
    # Startup
    logger.info("Starting up Context Manager service...")
    try:
        # Test database connection
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        logger.info("Database connection successful")
        
        # Test Redis connection
        redis_client.ping()
        logger.info("Redis connection successful")
        
        # Test Vector Storage & Retrieval Service
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{VECTOR_SERVICE_URL}/health") as response:
                if response.status != 200:
                    raise Exception("Vector Storage & Retrieval Service is not healthy")
                logger.info("Vector Storage & Retrieval Service connection successful")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Context Manager service...")

# Initialize FastAPI app
app = FastAPI(title="Context Manager Service", lifespan=lifespan)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    if not init_db():
        logger.error("Failed to initialize database")
        raise Exception("Database initialization failed")

@app.post("/context", response_model=ContextResponse)
async def get_context(request: ContextRequest, req: Request):
    """
    Get relevant context for a query.
    
    Input:
    {
        "session_id": "anonymous_789",
        "query": "And how do we issue refunds?",
        "session_type": "anonymous"
    }
    
    Output:
    {
        "query": "And how do we issue refunds?",
        "context": "Earlier in this session you asked about customer complaints and return periods...",
        "session_type": "anonymous",
        "context_sources": [
            {
                "message": "What is your return policy?",
                "timestamp": "2024-01-08T14:15:30Z",
                "similarity_score": 0.89
            }
        ]
    }
    """
    try:
        logger.info(f"Received context request for session {request.session_id}")
        
        # Get database session
        db = next(get_db())
        
        try:
            # Create or update session
            session = await create_or_update_session(
                db,
                request.session_id,
                request.session_type,
                req
            )
            
            # Get session's message history
            history = await get_session_history(db, request.session_id)
            logger.info(f"Retrieved {len(history)} messages from history")
            
            # Get relevant messages using embedding similarity
            relevant_messages = await get_relevant_messages(request.query, history)
            logger.info(f"Found {len(relevant_messages)} relevant messages")
            
            # Format context and prepare sources
            context = format_context(relevant_messages)
            context_sources = [
                ContextSource(
                    message=msg.get("message", ""),
                    timestamp=msg.get("timestamp", datetime.utcnow().isoformat()),
                    similarity_score=msg.get("similarity_score", 0.0)
                )
                for msg in relevant_messages
            ]
            
            return ContextResponse(
                query=request.query,
                context=context,
                session_type=request.session_type,
                context_sources=context_sources
            )
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    except Exception as e:
        logger.error(f"Error getting context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class StoreMessageRequest(BaseModel):
    session_id: str
    message_text: str
    message_type: str
    user_id: Optional[str] = None
    sources_used: Optional[Dict[str, Any]] = None

@app.post("/store")
async def store_message(request: StoreMessageRequest):
    """Store a new message in the database."""
    logger.info(f"Storing message for session: {request.session_id}")
    logger.debug(f"Message details - type: {request.message_type}, user_id: {request.user_id}")
    
    db = SessionLocal()
    try:
        # Log the start of embedding generation
        logger.debug("Generating embeddings for message")
        embeddings = await get_embeddings(request.message_text)
        if not embeddings:
            error_msg = "Failed to generate embedding - empty response from embedding service"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        logger.debug("Creating ChatMessage object")
        message = ChatMessage(
            session_id=request.session_id,
            message_text=request.message_text,
            message_type=request.message_type,
            user_id=request.user_id,
            embedding=embeddings[0],  # Store as JSONB directly
            sources_used=request.sources_used
        )
        
        logger.debug("Adding message to database")
        db.add(message)
        db.commit()
        db.refresh(message)
        
        logger.info(f"Successfully stored message with ID: {message.id}")
        return {"status": "success", "message_id": message.id}
        
    except HTTPException as he:
        logger.error(f"HTTP error in store_message: {str(he)}")
        raise
    except (TypeError, ValueError) as je:
        error_msg = f"JSON encoding/decoding error: {str(je)}"
        logger.error(error_msg, exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=error_msg)
    except SQLAlchemyError as se:
        error_msg = f"Database error: {str(se)}"
        logger.error(error_msg, exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Unexpected error in store_message: {str(e)}"
        logger.error(error_msg, exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        try:
            db.close()
        except Exception as e:
            logger.error(f"Error closing database connection: {str(e)}")
            # Don't raise to avoid masking the original error

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        
        # Check Redis
        redis_client.ping()
        
        # Check Vector Storage & Retrieval Service
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{VECTOR_SERVICE_URL}/health") as response:
                if response.status != 200:
                    return {"status": "unhealthy", "error": "Vector Storage & Retrieval Service is not healthy"}
        
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}