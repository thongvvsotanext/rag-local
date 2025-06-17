from pydantic import BaseModel, EmailStr
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import validator

# User schemas
class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str
    is_admin: bool = False

class UserLogin(UserBase):
    password: str

class User(UserBase):
    id: int
    is_admin: bool
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

# Document schemas
class DocumentBase(BaseModel):
    filename: str
    file_type: str
    size: int
    document_metadata: Optional[Dict[str, Any]] = None

class DocumentCreate(DocumentBase):
    pass

class Document(DocumentBase):
    id: int
    doc_id: str
    owner_id: int
    total_chunks: int
    status: str
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

# Web page schemas
class WebPageBase(BaseModel):
    url: str
    title: Optional[str] = None
    domain: Optional[str] = None

class WebPageCreate(WebPageBase):
    crawl_job_id: str

class WebPage(WebPageBase):
    id: int
    page_id: str
    crawl_job_id: str
    crawled_at: datetime
    status: str
    content_length: Optional[int] = None
    chunk_count: int

    class Config:
        from_attributes = True

# Document chunk schemas
class DocumentChunkBase(BaseModel):
    chunk_text: str
    chunk_index: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    content_type: str = "document"
    source_url: Optional[str] = None
    domain: Optional[str] = None

class DocumentChunkCreate(DocumentChunkBase):
    doc_id: str
    page_id: Optional[str] = None

class DocumentChunk(DocumentChunkBase):
    id: int
    chunk_id: str
    doc_id: str
    page_id: Optional[str]
    faiss_index: int
    created_at: datetime

    class Config:
        from_attributes = True

# Chat session schemas
class ChatSessionBase(BaseModel):
    session_type: str = "anonymous"
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

class ChatSessionCreate(ChatSessionBase):
    user_id: Optional[str] = None

class ChatSession(ChatSessionBase):
    id: int
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    last_activity: datetime
    expires_at: datetime

    class Config:
        from_attributes = True

# Chat message schemas
class ChatMessageBase(BaseModel):
    message_text: str
    message_type: str
    sources_used: Optional[Dict[str, Any]] = None

class ChatMessageCreate(ChatMessageBase):
    pass

class ChatMessage(ChatMessageBase):
    id: int
    session_id: str
    faiss_index: Optional[int]
    response_time_ms: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True

# Crawl job schemas
class CrawlJobBase(BaseModel):
    start_urls: List[str]
    max_pages: int
    max_depth: int
    filters: Optional[Dict[str, Any]] = None

class CrawlJobCreate(CrawlJobBase):
    pass

class CrawlJob(CrawlJobBase):
    id: int
    job_id: str
    user_id: int
    status: str
    pages_crawled: int
    pages_failed: int
    chunks_created: int
    created_at: datetime
    completed_at: Optional[datetime]
    next_scheduled: Optional[datetime]
    error_details: Optional[str]

    class Config:
        from_attributes = True

# Processing job schemas
class ProcessingJobBase(BaseModel):
    job_type: str
    input_data: Dict[str, Any]

class ProcessingJobCreate(ProcessingJobBase):
    pass

class ProcessingJob(ProcessingJobBase):
    id: int
    job_id: str
    status: str
    result_data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    retry_count: int

    class Config:
        from_attributes = True

# Token schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# Document upload schemas
class DocumentResponse(BaseModel):
    document_id: int
    status: str

# Crawl schemas
class CrawlFilters(BaseModel):
    exclude_patterns: List[str] = []
    content_types: List[str] = ["text/html"]
    min_content_length: int = 200

class CrawlRequest(BaseModel):
    url: str
    max_pages: int = 10
    max_depth: int = 2
    filters: CrawlFilters = CrawlFilters()
    respect_robots: bool = True
    rate_limit_delay: float = 1.0
    schedule: Optional[str] = None

class CrawlResponse(BaseModel):
    job_id: str
    status: str
    pages_crawled: int = 0
    pages_failed: int = 0
    chunks_created: int = 0
    processing_time: str = "0s"
    next_scheduled: Optional[str] = None
    failed_urls: List[Dict[str, Any]] = []
    crawl_statistics: Dict[str, Any] = {
        "total_content_size": "0B",
        "average_chunk_size": 0,
        "domains_crawled": [],
        "content_types_found": []
    }

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    result: Optional[Dict[str, Any]] = None

# Chat request/response schemas
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
        return v.strip()

    @validator('session_id')
    def validate_session_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Session ID cannot be empty')
        return v.strip()

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    session_id: str
    processing_time: float

# Document upload schemas
class DocumentUpload(BaseModel):
    filename: str
    file_type: str
    size: int
    document_metadata: Optional[Dict[str, Any]] = None 