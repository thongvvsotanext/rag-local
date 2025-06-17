from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime, JSON, Text, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func, text
from database.database import Base
from sqlalchemy.dialects.postgresql import JSONB

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    documents = relationship("Document", back_populates="owner")
    crawl_jobs = relationship("CrawlJob", back_populates="user")

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    doc_id = Column(String(255), unique=True, nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String)
    file_path = Column(String)  # Path to the file in the shared volume
    file_type = Column(String(50), nullable=False)
    size = Column(Integer)
    total_chunks = Column(Integer, default=0)
    status = Column(String)  # processing, processed, failed
    document_metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    owner = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document")

class CrawlJob(Base):
    __tablename__ = "crawl_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(255), unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"))
    start_urls = Column(ARRAY(Text), nullable=False)
    max_pages = Column(Integer)
    max_depth = Column(Integer)
    status = Column(String)  # pending, processing, completed, failed
    pages_crawled = Column(Integer, default=0)
    pages_failed = Column(Integer, default=0)
    chunks_created = Column(Integer, default=0)
    filters = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    next_scheduled = Column(DateTime(timezone=True))
    error_details = Column(Text)

    # Relationships
    user = relationship("User", back_populates="crawl_jobs")
    web_pages = relationship("WebPage", back_populates="crawl_job")

class WebPage(Base):
    __tablename__ = "web_pages"

    id = Column(Integer, primary_key=True, index=True)
    page_id = Column(String(255), unique=True, nullable=False)
    crawl_job_id = Column(String(255), ForeignKey("crawl_jobs.job_id"), nullable=False)
    url = Column(Text, nullable=False)
    title = Column(String(500))
    domain = Column(String(255))
    crawled_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String(50), default="success")
    content_length = Column(Integer)
    chunk_count = Column(Integer, default=0)

    # Relationships
    crawl_job = relationship("CrawlJob", back_populates="web_pages")
    chunks = relationship("DocumentChunk", back_populates="web_page")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(String(255), unique=True, nullable=False)
    doc_id = Column(String(255), ForeignKey("documents.doc_id"), nullable=False)
    page_id = Column(String(255), ForeignKey("web_pages.page_id"))
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer)
    section_title = Column(String(255))
    content_type = Column(String(50), default="document")
    source_url = Column(Text)
    domain = Column(String(255))
    faiss_index = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    document = relationship("Document", back_populates="chunks")
    web_page = relationship("WebPage", back_populates="chunks")

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

class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(255), unique=True, nullable=False)
    job_type = Column(String(50), nullable=False)
    status = Column(String(50), default="pending")
    input_data = Column(JSONB, nullable=False)
    result_data = Column(JSONB)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    retry_count = Column(Integer, default=0) 