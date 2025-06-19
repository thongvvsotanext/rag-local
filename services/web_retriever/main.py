from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends, status, Query, Response
from pydantic import BaseModel, HttpUrl, Field, validator
from typing import List, Dict, Any, Optional, Union
import os
from dotenv import load_dotenv
import logging
import sys
from datetime import datetime
from enum import Enum
from uuid import uuid4
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from readability import Document
import re
from urllib.parse import urlparse, urljoin
import json
from contextlib import asynccontextmanager
import boto3
from sqlalchemy import Column, String, Integer, DateTime, Boolean, JSON, ForeignKey, Text, create_engine, text, MetaData, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
import asyncpg
import hashlib
from ratelimit import limits, sleep_and_retry
from sqlalchemy.orm import sessionmaker, relationship

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('web_retriever.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
# Database
DB_HOST = os.getenv("POSTGRES_HOST", "db")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_USER = os.getenv("POSTGRES_USER", "fizen_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fizen_password")
DB_NAME = os.getenv("POSTGRES_DB", "fizen_rag")

# SearxNG Configuration
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")
SEARXNG_TIMEOUT = int(os.getenv("SEARXNG_TIMEOUT", "10"))
SEARXNG_SAFE_SEARCH = int(os.getenv("SEARXNG_SAFE_SEARCH", "1"))  # 0=off, 1=moderate, 2=strict

if not all([DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME]):
    raise ValueError("Missing required database configuration. Please set all POSTGRES_* environment variables.")

# Initialize database
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Initialize other clients
bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("BEDROCK_REGION", "us-east-1"))
metadata = MetaData()

# Constants
MAX_CRAWL_DEPTH = 5
MAX_PAGES_PER_DOMAIN = 100
MAX_CONTENT_LENGTH = 10000  # characters
MIN_CONTENT_LENGTH = 200    # characters

# Models
class CrawlJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    SCHEDULED = "scheduled"

class ContentType(str, Enum):
    ARTICLE = "article"
    BLOG = "blog"
    DOCUMENTATION = "documentation"
    FORUM = "forum"
    NEWS = "news"
    PRODUCT = "product"
    WIKI = "wiki"
    OTHER = "other"

class CrawlJob(Base):
    __tablename__ = "crawl_jobs"
    
    id = Column(String, primary_key=True, index=True, default=lambda: f"crawl_{uuid4().hex}")
    status = Column(String, default=CrawlJobStatus.PENDING.value, index=True)
    urls = Column(JSON, nullable=False)
    max_pages = Column(Integer, default=50)
    max_depth = Column(Integer, default=3)
    filters = Column(JSON, default={
        "exclude_patterns": [],
        "allowed_domains": [],
        "content_types": ["text/html"],
        "min_content_length": MIN_CONTENT_LENGTH,
        "respect_robots": True,
        "rate_limit_delay": 1.0
    })
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    next_scheduled = Column(DateTime, nullable=True)
    stats = Column(JSON, default={
        "pages_crawled": 0,
        "pages_failed": 0,
        "chunks_created": 0,
        "domains": {}
    })
    error = Column(String, nullable=True)
    created_by = Column(String, nullable=True)
    crawl_metadata = Column('metadata', JSON, default=dict)
    callback_url = Column(String, nullable=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            "urls": self.urls,
            "max_pages": self.max_pages,
            "max_depth": self.max_depth,
            "filters": self.filters,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "next_scheduled": self.next_scheduled.isoformat() if self.next_scheduled else None,
            "stats": self.stats,
            "error": self.error,
            "metadata": self.crawl_metadata,
            "callback_url": self.callback_url
        }

# Input/Output Models
class CrawlFilter(BaseModel):
    """Filtering options for web crawling."""
    exclude_patterns: List[str] = Field(
        default_factory=list,
        description="List of URL patterns to exclude from crawling"
    )
    allowed_domains: List[str] = Field(
        default_factory=list,
        description="List of allowed domains (empty allows all)"
    )
    content_types: List[str] = Field(
        default_factory=lambda: ["text/html"],
        description="Content types to include"
    )
    min_content_length: int = Field(
        default=MIN_CONTENT_LENGTH,
        ge=0,
        description="Minimum content length in characters"
    )
    respect_robots: bool = Field(
        default=True,
        description="Whether to respect robots.txt rules"
    )
    rate_limit_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Delay between requests in seconds"
    )
    max_links_per_page: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of links to follow per page"
    )

class CrawlRequest(BaseModel):
    """Request model for starting a new crawl job."""
    urls: List[HttpUrl] = Field(
        ...,
        min_items=1,
        description="List of seed URLs to start crawling from"
    )
    max_pages: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum number of pages to crawl"
    )
    max_depth: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum link depth to follow"
    )
    filters: CrawlFilter = Field(
        default_factory=CrawlFilter,
        description="Content filtering options"
    )
    schedule: Optional[str] = Field(
        None,
        regex=r'^(@(annually|yearly|monthly|weekly|daily|hourly)|((\*|\d+)(\/\d+)?(\s+\S+){4,5}))$',
        description="Cron expression for scheduled crawls"
    )
    callback_url: Optional[HttpUrl] = Field(
        None,
        description="URL to receive webhook notifications"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the crawl job"
    )
    
    @validator('urls')
    def validate_urls(cls, urls):
        if not urls:
            raise ValueError("At least one URL is required")
        return [str(url) for url in urls]
    
    @validator('filters')
    def validate_filters(cls, filters):
        if isinstance(filters, dict):
            filters = CrawlFilter(**filters)
        if filters.allowed_domains:
            filters.allowed_domains = [d.lower() for d in filters.allowed_domains]
        return filters

class CrawlJobStats(BaseModel):
    """Statistics for a crawl job."""
    pages_crawled: int = 0
    pages_failed: int = 0
    chunks_created: int = 0
    total_content_size: int = 0  # in bytes
    avg_chunk_size: float = 0
    domains: Dict[str, int] = Field(default_factory=dict)  # domain: page_count
    content_types: Dict[str, int] = Field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def processing_time_seconds(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

class CrawlResponse(BaseModel):
    """Response model for crawl job operations."""
    job_id: str = Field(..., description="Unique identifier for the crawl job")
    status: CrawlJobStatus = Field(..., description="Current status of the crawl job")
    urls: List[str] = Field(..., description="List of seed URLs")
    max_pages: int = Field(..., description="Maximum pages to crawl")
    max_depth: int = Field(..., description="Maximum crawl depth")
    filters: CrawlFilter = Field(..., description="Content filtering options")
    created_at: datetime = Field(..., description="When the job was created")
    started_at: Optional[datetime] = Field(None, description="When the job started running")
    completed_at: Optional[datetime] = Field(None, description="When the job completed")
    next_scheduled: Optional[datetime] = Field(None, description="Next scheduled run time")
    stats: CrawlJobStats = Field(default_factory=CrawlJobStats, description="Crawl statistics")
    error: Optional[str] = Field(None, description="Error details if the job failed")
    callback_url: Optional[HttpUrl] = Field(None, description="Webhook URL for notifications")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            HttpUrl: lambda v: str(v) if v else None
        }
        use_enum_values = True

class CrawlJobListResponse(BaseModel):
    """Response model for listing crawl jobs."""
    jobs: List[CrawlResponse] = Field(..., description="List of crawl jobs")
    total: int = Field(..., description="Total number of jobs")
    limit: int = Field(..., description="Number of jobs per page")
    offset: int = Field(..., description="Pagination offset")

class SearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5
    source_types: Optional[List[str]] = ["news", "official"]

class SearchResult(BaseModel):
    url: str
    title: str
    summary: str
    domain: str
    content_type: str
    crawled_at: str
    relevance_score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]

# Rate limiting decorator
ONE_MINUTE = 60
MAX_REQUESTS_PER_MINUTE = 30

@sleep_and_retry
@limits(calls=MAX_REQUESTS_PER_MINUTE, period=ONE_MINUTE)
async def rate_limited_request(session: aiohttp.ClientSession, url: str, **kwargs) -> aiohttp.ClientResponse:
    """Make a rate-limited HTTP request."""
    return await session.get(url, **kwargs)

async def fetch_robots_txt(session: aiohttp.ClientSession, domain: str) -> str:
    """Fetch and parse robots.txt for a domain."""
    try:
        robots_url = f"https://{domain}/robots.txt"
        async with rate_limited_request(session, robots_url) as response:
            if response.status == 200:
                return await response.text()
    except Exception as e:
        logger.warning(f"Error fetching robots.txt for {domain}: {str(e)}")
    return ""

async def is_allowed_by_robots(url: str, robots_txt: str) -> bool:
    """Check if URL is allowed by robots.txt."""
    try:
        from urllib.robotparser import RobotFileParser
        rp = RobotFileParser()
        rp.parse(robots_txt.splitlines())
        return rp.can_fetch("*", url)
    except Exception as e:
        logger.warning(f"Error parsing robots.txt: {str(e)}")
        return True

async def extract_main_content(html: str) -> str:
    """Extract main content from HTML using readability."""
    try:
        doc = Document(html)
        return doc.summary()
    except Exception as e:
        logger.error(f"Error extracting content: {str(e)}")
        return ""

async def clean_text(text: str) -> str:
    """Clean extracted text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

async def generate_content_hash(text: str) -> str:
    """Generate hash for content deduplication."""
    return hashlib.md5(text.encode()).hexdigest()

# Vector Retriever Service Configuration
VECTOR_RETRIEVER_URL = os.getenv("VECTOR_RETRIEVER_URL", "http://vector-service:8003")
MAX_RETRIES = 3
RETRY_DELAY = 1.0

async def store_content_in_db(content: Dict[str, Any]) -> List[str]:
    """
    Store extracted content in the vector database via the vector-retriever service.
    
    Args:
        content: Dictionary containing the content and metadata to store
        
    Returns:
        List of chunk IDs that were created
    """
    if not content or "content" not in content:
        logger.error("No content provided to store in database")
        return []
    
    try:
        # Prepare metadata for vector retriever
        metadata = {
            "doc_id": content.get("doc_id"),
            "url": content.get("url"),
            "title": content.get("title", ""),
            "domain": content.get("domain", ""),
            "language": content.get("language", "en"),
            "content_type": content.get("content_type", "webpage"),
            "source": "web_crawler",
            "crawled_at": content.get("crawled_at", datetime.utcnow().isoformat()),
            **content.get("metadata", {})
        }
        
        # Clean and preprocess the content
        text = clean_text(content["content"])
        
        # Skip if text is too short
        if not text or len(text) < MIN_CONTENT_LENGTH:
            logger.warning(f"Content too short or empty, skipping storage")
            return []
            
        # Prepare the request payload for vector-retriever
        payload = {
            "text": text,
            "metadata": metadata,
            "chunk_size": content.get("chunk_size", CHUNK_SIZE),
            "chunk_overlap": content.get("chunk_overlap", CHUNK_OVERLAP)
        }
        
        # Call vector-retriever service to store the content
        async with aiohttp.ClientSession() as session:
            for attempt in range(MAX_RETRIES):
                try:
                    # First check if the vector-retriever is healthy
                    health_url = f"{VECTOR_RETRIEVER_URL}/health"
                    async with session.get(health_url) as response:
                        if response.status != 200:
                            raise HTTPException(
                                status_code=503,
                                detail=f"Vector retriever service is not healthy: {await response.text()}"
                            )
                    
                    # Store the content
                    store_url = f"{VECTOR_RETRIEVER_URL}/ingest/web"
                    async with session.post(store_url, json=payload) as store_response:
                        if store_response.status == 200:
                            result = await store_response.json()
                            return result.get('chunk_ids', [])
                        else:
                            error_detail = await store_response.text()
                            logger.error(f"Error storing content in vector store (attempt {attempt + 1}): {error_detail}")
                            
                            # If not the last attempt, wait before retrying
                            if attempt < MAX_RETRIES - 1:
                                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                                continue
                                
                            raise HTTPException(
                                status_code=store_response.status,
                                detail=f"Failed to store content in vector store: {error_detail}"
                            )
                            
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.error(f"Error connecting to vector-retriever service (attempt {attempt + 1}): {str(e)}")
                    if attempt == MAX_RETRIES - 1:  # Last attempt
                        raise HTTPException(
                            status_code=503,
                            detail=f"Failed to connect to vector-retriever service after {MAX_RETRIES} attempts: {str(e)}"
                        )
    except Exception as e:
        logger.error(f"Error in store_content_in_db: {str(e)}", exc_info=True)
        raise

async def crawl_url(
    session: aiohttp.ClientSession,
    url: str,
    depth: int,
    max_depth: int,
    filters: Dict[str, Any],
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Crawl a single URL and extract content with enhanced error handling and metadata.
    
    Args:
        session: aiohttp client session
        url: URL to crawl
        depth: Current crawl depth
        max_depth: Maximum allowed crawl depth
        filters: Content filtering options
        timeout: Request timeout in seconds
        
    Returns:
        Dict containing extracted content and metadata, or error information
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    base_url = f"{parsed_url.scheme}://{domain}"
    
    try:
        # Validate URL against allowed domains if specified
        if filters.get('allowed_domains') and domain not in filters['allowed_domains']:
            return {"url": url, "error": f"Domain {domain} not in allowed domains"}
        
        # Check URL against exclude patterns
        for pattern in filters.get('exclude_patterns', []):
            if re.search(pattern, url, re.IGNORECASE):
                return {"url": url, "error": f"URL matches exclude pattern: {pattern}"}
        
        # Set request headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; FizenRAG/1.0; +https://fizen.io/rag-bot)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Make the request with timeout
        async with session.get(
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
            ssl=False
        ) as response:
            if response.status != 200:
                return {"url": url, "error": f"HTTP {response.status}"}
            
            # Check content type
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                return {"url": url, "error": f"Unsupported content type: {content_type}"}
            
            # Read and decode response
            html = await response.text()
            
            # Extract main content using readability
            doc = Document(html)
            title = doc.title() or url
            content = doc.summary()
            
            # Clean and process content
            cleaned_text = clean_text(content)
            
            # Check content length
            if len(cleaned_text) < filters.get('min_content_length', MIN_CONTENT_LENGTH):
                return {"url": url, "error": f"Content too short: {len(cleaned_text)} chars"}
            
            # Extract metadata
            soup = BeautifulSoup(html, 'html.parser')
            meta = {}
            
            # Get OpenGraph/Twitter metadata
            for meta_tag in soup.find_all('meta'):
                name = meta_tag.get('name', meta_tag.get('property', '')).lower()
                content = meta_tag.get('content', '').strip()
                if name and content:
                    meta[name] = content
            
            # Get language
            lang = soup.html.get('lang', 'en') if soup.html else 'en'
            
            # Extract links
            links = set()
            for a in soup.find_all('a', href=True):
                href = a['href'].strip()
                if not href or href.startswith(('javascript:', 'mailto:', 'tel:', '#')):
                    continue
                    
                # Resolve relative URLs
                if not href.startswith(('http://', 'https://')):
                    if href.startswith('//'):  # Protocol-relative URL
                        href = f"{parsed_url.scheme}:{href}"
                    else:
                        href = urljoin(base_url, href)
                
                # Filter out unwanted URLs
                parsed = urlparse(href)
                if not parsed.netloc or not parsed.scheme or parsed.scheme not in ('http', 'https'):
                    continue
                    
                # Apply domain filtering
                if filters.get('allowed_domains') and parsed.netloc not in filters['allowed_domains']:
                    continue
                
                # Clean up URL
                href = parsed._replace(fragment='', params='', query='').geturl()
                links.add(href)
            
            # Generate content hash for deduplication
            content_hash = hashlib.sha256(cleaned_text.encode('utf-8')).hexdigest()
            
            # Generate document ID
            doc_id = f"web_{hashlib.md5(url.encode()).hexdigest()}"
            
            # Determine content type (article, blog, documentation, etc.)
            content_type = 'other'
            path = parsed_url.path.lower()
            if any(p in path for p in ['/blog/', '/news/', '/articles/']):
                content_type = 'article'
            elif any(p in path for p in ['/docs/', '/documentation/']):
                content_type = 'documentation'
            elif any(p in path for p in ['/forum/', '/discussion/']):
                content_type = 'forum'
            
            # Extract published date if available
            published_date = None
            if 'article:published_time' in meta:
                try:
                    published_date = datetime.fromisoformat(meta['article:published_time'].replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    pass
            
            return {
                "doc_id": doc_id,
                "url": url,
                "title": title,
                "content": cleaned_text,
                "domain": domain,
                "language": lang,
                "content_type": content_type,
                "content_hash": content_hash,
                "metadata": meta,
                "published_date": published_date.isoformat() if published_date else None,
                "crawled_at": datetime.utcnow().isoformat(),
                "depth": depth,
                "links": list(links)[:filters.get('max_links_per_page', 20)],
                "status": "success"
            }
    
    except asyncio.TimeoutError:
        return {"url": url, "error": f"Request timed out after {timeout} seconds"}
    except aiohttp.ClientError as e:
        return {"url": url, "error": f"HTTP client error: {str(e)}"}
    except Exception as e:
        logger.error(f"Error crawling {url}: {str(e)}", exc_info=True)
        return {"url": url, "error": f"Crawling error: {str(e)}"}

async def process_crawl_job(job_id: str, request: CrawlRequest):
    """
    Process a crawl job with enhanced error handling, rate limiting, and progress tracking.
    
    Args:
        job_id: Unique identifier for the crawl job
        request: CrawlRequest with configuration
    """
    start_time = datetime.utcnow()
    visited_urls = set()
    queue = asyncio.Queue()
    failed_urls = []
    chunks_created = 0
    domains = {}
    content_types = {}
    
    # Initialize job status
    await update_job_status(
        job_id=job_id,
        status=CrawlJobStatus.RUNNING,
        started_at=start_time
    )
    
    try:
        # Initialize queue with seed URLs
        for url in request.urls:
            if url not in visited_urls:
                await queue.put((url, 0))  # (url, depth)
                visited_urls.add(url)
        
        # Process URLs until queue is empty or max pages reached
        async with aiohttp.ClientSession() as session:
            while not queue.empty() and len(visited_urls) < request.max_pages:
                url, depth = await queue.get()
                
                try:
                    # Check if we should respect robots.txt for this domain
                    domain = urlparse(url).netloc
                    if domain not in domains:
                        domains[domain] = {
                            'robots_txt': await fetch_robots_txt(session, domain) if request.filters.respect_robots else None,
                            'count': 0
                        }
                    
                    robots_txt = domains[domain]['robots_txt']
                    
                    # Check if URL is allowed by robots.txt
                    if robots_txt and not await is_allowed_by_robots(url, robots_txt):
                        logger.info(f"Skipping {url} - disallowed by robots.txt")
                        continue
                    
                    # Rate limiting
                    await asyncio.sleep(request.filters.rate_limit_delay)
                    
                    # Fetch and process the page
                    content = await crawl_url(
                        session=session,
                        url=url,
                        depth=depth,
                        max_depth=request.max_depth,
                        filters=request.filters
                    )
                    
                    if content and "error" not in content:
                        # Store content in vector database
                        chunk_ids = await store_content_in_db(content)
                        chunks_created += len(chunk_ids)
                        
                        # Update domain and content type stats
                        domains[domain]['count'] += 1
                        content_type = content.get('content_type', 'unknown')
                        content_types[content_type] = content_types.get(content_type, 0) + 1
                        
                        # Add new links to queue if we haven't reached max depth
                        if depth < request.max_depth:
                            for link in content.get('links', [])[:request.filters.max_links_per_page]:
                                if link not in visited_urls:
                                    await queue.put((link, depth + 1))
                                    visited_urls.add(link)
                    
                    # Update job progress periodically
                    if len(visited_urls) % 10 == 0:
                        await update_job_progress(
                            job_id=job_id,
                            pages_crawled=len(visited_urls) - len(failed_urls),
                            pages_failed=len(failed_urls),
                            chunks_created=chunks_created,
                            domains=domains,
                            content_types=content_types
                        )
                
                except Exception as e:
                    logger.error(f"Error processing {url}: {str(e)}")
                    failed_urls.append({
                        'url': url,
                        'error': str(e),
                        'timestamp': datetime.utcnow().isoformat()
                    })
                
                finally:
                    queue.task_done()
        
        # Calculate next scheduled time if this is a recurring job
        next_scheduled = None
        if request.schedule:
            next_scheduled = calculate_next_schedule(request.schedule)
        
        # Mark job as completed
        await update_job_status(
            job_id=job_id,
            status=CrawlJobStatus.COMPLETED,
            completed_at=datetime.utcnow(),
            next_scheduled=next_scheduled,
            stats={
                'pages_crawled': len(visited_urls) - len(failed_urls),
                'pages_failed': len(failed_urls),
                'chunks_created': chunks_created,
                'domains': {d: info['count'] for d, info in domains.items()},
                'content_types': content_types,
                'start_time': start_time,
                'end_time': datetime.utcnow()
            }
        )
        
        # Trigger callback if configured
        if request.callback_url:
            await trigger_callback(request.callback_url, job_id, 'completed')
    
    except Exception as e:
        logger.error(f"Fatal error in crawl job {job_id}: {str(e)}", exc_info=True)
        await update_job_status(
            job_id=job_id,
            status=CrawlJobStatus.FAILED,
            error=str(e),
            completed_at=datetime.utcnow(),
            stats={
                'pages_crawled': len(visited_urls) - len(failed_urls),
                'pages_failed': len(failed_urls),
                'chunks_created': chunks_created,
                'domains': {d: info['count'] for d, info in domains.items()},
                'content_types': content_types,
                'start_time': start_time,
                'end_time': datetime.utcnow()
            }
        )
        
        if 'request' in locals() and request.callback_url:
            await trigger_callback(request.callback_url, job_id, 'failed', str(e))
    
    finally:
        # Clean up resources
        if 'session' in locals():
            await session.close()

async def update_job_status(
    job_id: str,
    status: CrawlJobStatus,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
    next_scheduled: Optional[datetime] = None,
    error: Optional[str] = None,
    stats: Optional[Dict] = None
):
    """Update the status of a crawl job in the database."""
    update_data = {
        'status': status.value,
        'updated_at': datetime.utcnow()
    }
    
    if started_at:
        update_data['started_at'] = started_at
    if completed_at:
        update_data['completed_at'] = completed_at
    if next_scheduled:
        update_data['next_scheduled'] = next_scheduled
    if error is not None:
        update_data['error'] = error
    if stats:
        update_data['stats'] = stats
    
    async with engine.begin() as conn:
        await conn.execute(
            text("""
                UPDATE crawl_jobs 
                SET 
                    status = :status,
                    started_at = COALESCE(:started_at, started_at),
                    completed_at = COALESCE(:completed_at, completed_at),
                    next_scheduled = :next_scheduled,
                    error = COALESCE(:error, error),
                    stats = COALESCE(:stats::jsonb, stats),
                    updated_at = :updated_at
                WHERE job_id = :job_id
            """),
            {'job_id': job_id, **update_data}
        )

async def update_job_progress(
    job_id: str,
    pages_crawled: int,
    pages_failed: int,
    chunks_created: int,
    domains: Dict,
    content_types: Dict
):
    """Update the progress of a crawl job in the database."""
    stats = {
        'pages_crawled': pages_crawled,
        'pages_failed': pages_failed,
        'chunks_created': chunks_created,
        'domains': {d: info['count'] if isinstance(info, dict) else info for d, info in domains.items()},
        'content_types': content_types,
        'updated_at': datetime.utcnow().isoformat()
    }
    
    async with engine.begin() as conn:
        await conn.execute(
            text("""
                UPDATE crawl_jobs 
                SET 
                    stats = :stats::jsonb,
                    updated_at = NOW()
                WHERE job_id = :job_id
            """),
            {'job_id': job_id, 'stats': json.dumps(stats)}
        )

async def trigger_callback(url: str, job_id: str, status: str, error: Optional[str] = None):
    """Trigger a callback to the specified URL with job status."""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                'job_id': job_id,
                'status': status,
                'timestamp': datetime.utcnow().isoformat()
            }
            if error:
                payload['error'] = error
                
            async with session.post(url, json=payload) as response:
                if response.status >= 400:
                    logger.error(f"Callback to {url} failed with status {response.status}")
    except Exception as e:
        logger.error(f"Error triggering callback to {url}: {str(e)}")

def calculate_next_schedule(schedule: str) -> datetime:
    """Calculate the next scheduled run time from a cron expression."""
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.util import datetime_repr
    
    try:
        trigger = CronTrigger.from_crontab(schedule)
        next_run = trigger.get_next_fire_times(None, datetime.utcnow(), None)[0]
        return next_run
    except Exception as e:
        logger.error(f"Error calculating next schedule for {schedule}: {str(e)}")
        return None

async def search_searxng(query: str, max_results: int = 5, categories: list = None):
    """Search using SearxNG metasearch engine.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        categories: List of search categories (e.g., ['general', 'news', 'science'])
        
    Returns:
        List of search results with metadata
    """
    if not categories:
        categories = ["general"]
        
    params = {
        "q": query,
        "format": "json",
        "pageno": 1,
        "safesearch": SEARXNG_SAFE_SEARCH,
        "categories": ",".join(categories),
        "language": "en",
        "timeout": SEARXNG_TIMEOUT
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{SEARXNG_URL}/search",
                params=params,
                timeout=SEARXNG_TIMEOUT + 5  # Add buffer for network
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"SearxNG API error: {error_text}")
                    return []
                
                data = await response.json()
                results = []
                
                # Process results
                for result in data.get("results", [])[:max_results]:
                    if not result.get("url"):
                        continue
                        
                    results.append({
                        "url": result.get("url"),
                        "title": result.get("title", ""),
                        "summary": result.get("content", ""),
                        "domain": urlparse(result.get("url")).netloc,
                        "content_type": result.get("category", "webpage"),
                        "crawled_at": datetime.utcnow().isoformat(),
                        "relevance_score": result.get("score", 0.0)
                    })
                    logger.info(f"Found title: {result.get('title', '')}, result: {result.get('content', '')}")
                
                return results
    
    except asyncio.TimeoutError:
        logger.error("SearxNG search timed out")
        return []
    except Exception as e:
        logger.error(f"Error searching with SearxNG: {str(e)}", exc_info=True)
        return []

async def search_web(request: SearchRequest):
    """Search web for current information using SearxNG.
    
    This endpoint provides real-time web search capabilities by querying multiple
    search engines through the SearxNG metasearch engine. It respects the source_types
    filter to return only results matching the specified content categories.
    """
    try:
        # Map source_types to SearxNG categories
        category_map = {
            "news": ["news"],
            "official": ["general"],  # Official sites often rank high in general search
            "academic": ["science", "academic"],
            "forum": ["social media", "qna"],
            "video": ["videos"],
            "image": ["images"],
            "map": ["map"]
        }
        
        # Determine categories based on source_types
        categories = []
        if request.source_types:
            for source_type in request.source_types:
                if source_type in category_map:
                    categories.extend(category_map[source_type])
        
        # Ensure we have at least one category
        if not categories:
            categories = ["general"]
            
        # Perform the search
        results = await search_searxng(
            query=request.query,
            max_results=request.max_results,
            categories=list(set(categories))  # Remove duplicates
        )
        
        # Filter results by domain if needed (e.g., for official sources)
        if "official" in (request.source_types or []):
            official_domains = [".gov", ".edu", ".org"]
            results = [r for r in results if any(d in r["domain"] for d in official_domains)]
        
        return SearchResponse(results=results)
    
    except Exception as e:
        logger.error(f"Error searching web: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to perform web search")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI app."""
    # Startup
    logger.info("Starting up Web Retriever service...")
    try:
        # Test database connection
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        
        # Test Bedrock connection
        bedrock.list_foundation_models()
        logger.info("Bedrock connection successful")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Web Retriever service...")
    try:
        await engine.dispose()
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

# Database Models
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
    document_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=text('now()'))

    # Relationships
    page = relationship("WebPage", back_populates="chunks")

def init_db():
    """Initialize database and create tables if they don't exist."""
    try:
        # Check if tables exist
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        required_tables = ["web_pages", "web_chunks"]
        
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

# Initialize FastAPI app
app = FastAPI(title="Web Retriever Service", lifespan=lifespan)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    if not init_db():
        logger.error("Failed to initialize database")
        raise Exception("Database initialization failed")

@app.get("/health", status_code=200)
async def health_check():
    """Health check endpoint for the web-retriever service."""
    return {"status": "healthy"}

@app.post("/crawl/start", response_model=CrawlResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_crawl(
    request: CrawlRequest,
    background_tasks: BackgroundTasks,
):
    """
    Create and start a new web crawling job.
    
    This endpoint accepts a crawl configuration and starts a background job to crawl
    the specified URLs according to the provided parameters.
    """
    try:
        # Validate request parameters
        if not request.urls and not request.seed_urls:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one URL or seed URL is required"
            )
        
        # Combine and validate URLs
        urls = set()
        for url in (request.urls or []) + (request.seed_urls or []):
            try:
                parsed = urlparse(str(url))
                if not all([parsed.scheme, parsed.netloc]):
                    raise ValueError(f"Invalid URL: {url}")
                urls.add(url.strip('/'))  # Normalize URLs by removing trailing slashes
            except Exception as e:
                logger.warning(f"Invalid URL {url}: {str(e)}")
        
        if not urls:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid URLs provided"
            )
        
        # Normalize filters
        filters = request.filters.dict() if request.filters else {}
        if 'allowed_domains' in filters and filters['allowed_domains']:
            filters['allowed_domains'] = [d.lower().strip() for d in filters['allowed_domains']]
        
        # Generate job ID and metadata
        job_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        
        # Create crawl job record
        crawl_job = {
            "job_id": job_id,
            "status": CrawlJobStatus.PENDING.value,
            "urls": list(urls),
            "filters": filters,
            "max_depth": min(request.max_depth, 10),  # Enforce reasonable limits
            "max_pages": min(request.max_pages, 10000) if request.max_pages else 1000,
            "schedule": request.schedule,
            "callback_url": request.callback_url,
            "created_at": created_at,
            "updated_at": created_at,
            "metadata": {
                "user_agent": "Mozilla/5.0 (compatible; FizenRAG/1.0; +https://fizen.io/rag-bot)",
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "embedding_model": EMBEDDING_MODEL_NAME
            }
        }
        
        # Store in database
        async with engine.begin() as conn:
            await conn.execute(
                text("""
                    INSERT INTO crawl_jobs 
                    (job_id, status, urls, filters, max_depth, max_pages, 
                     schedule, callback_url, created_at, updated_at, metadata)
                    VALUES 
                    (:job_id, :status, :urls, :filters, :max_depth, :max_pages, 
                     :schedule, :callback_url, :created_at, :updated_at, :metadata)
                """),
                crawl_job
            )
        
        # Start background task
        background_tasks.add_task(process_crawl_job, job_id, request)
        
        # Log job creation
        logger.info(f"Crawl job {job_id} created with {len(urls)} URLs")
        
        return {
            "job_id": job_id,
            "status": CrawlJobStatus.PENDING.value,
            "message": f"Crawl job {job_id} started",
            "created_at": created_at.isoformat(),
            "urls": list(urls)[:5],  # Return first 5 URLs as sample
            "url_count": len(urls)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating crawl job: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create crawl job: {str(e)}"
        )

@app.get("/crawl/list", response_model=List[CrawlResponse])
async def list_crawl_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip for pagination"),
    sort_by: str = Query("-created_at", description="Sort field with optional - prefix for descending"),
    domain: Optional[str] = Query(None, description="Filter by domain")
):
    """
    List crawl jobs with optional filtering and pagination.
    
    Returns a paginated list of crawl jobs, with support for filtering by status, domain, and other criteria.
    """
    try:
        # Build base query
        query = """
            SELECT 
                job_id, status, urls, max_depth, max_pages, schedule,
                started_at, completed_at, created_at, updated_at,
                (SELECT COUNT(*) FROM web_pages WHERE web_pages.crawl_job_id = crawl_jobs.job_id) as pages_crawled,
                (SELECT COUNT(*) FROM web_pages WHERE web_pages.crawl_job_id = crawl_jobs.job_id AND status = 'error') as pages_failed,
                (SELECT COUNT(*) FROM document_chunks WHERE document_chunks.metadata->>'crawl_job_id' = crawl_jobs.job_id) as chunks_created
            FROM crawl_jobs
            WHERE 1=1
        """
        
        # Add status filter
        if status:
            query += " AND status = :status"
        
        # Add domain filter
        if domain:
            query += " AND EXISTS (SELECT 1 FROM jsonb_array_elements_text(urls) url WHERE url LIKE :domain_pattern)"
        
        # Add sorting
        sort_field = sort_by.lstrip('-')
        sort_dir = "DESC" if sort_by.startswith('-') else "ASC"
        
        # Validate sort field to prevent SQL injection
        valid_sort_fields = {
            "job_id", "status", "created_at", "updated_at", "started_at", "completed_at",
            "max_depth", "max_pages", "pages_crawled", "pages_failed", "chunks_created"
        }
        
        if sort_field not in valid_sort_fields:
            sort_field = "created_at"
            sort_dir = "DESC"
        
        query += f" ORDER BY {sort_field} {sort_dir} NULLS LAST"
        
        # Add pagination
        query += " LIMIT :limit OFFSET :offset"
        
        # Get total count for pagination
        count_query = "SELECT COUNT(*) as total FROM crawl_jobs WHERE 1=1"
        if status:
            count_query += " AND status = :status"
        if domain:
            count_query += " AND EXISTS (SELECT 1 FROM jsonb_array_elements_text(urls) url WHERE url LIKE :domain_pattern)"
        
        async with engine.begin() as conn:
            # Get total count
            count_result = await conn.execute(text(count_query), {"status": status, "domain_pattern": f"%{domain}%" if domain else None})
            total_count = count_result.scalar()
            
            # Get paginated results
            result = await conn.execute(text(query), {"status": status, "limit": limit, "offset": offset, "domain_pattern": f"%{domain}%" if domain else None})
            jobs = [dict(row) for row in result.mappings()]
            
            # Format response
            response = []
            for job in jobs:
                response.append({
                    "job_id": job["job_id"],
                    "status": job["status"],
                    "created_at": job["created_at"].isoformat() if job["created_at"] else None,
                    "started_at": job["started_at"].isoformat() if job["started_at"] else None,
                    "completed_at": job["completed_at"].isoformat() if job["completed_at"] else None,
                    "url_count": len(job["urls"]) if job["urls"] else 0,
                    "pages_crawled": job["pages_crawled"] or 0,
                    "pages_failed": job["pages_failed"] or 0,
                    "chunks_created": job["chunks_created"] or 0,
                    "max_depth": job["max_depth"],
                    "max_pages": job["max_pages"],
                    "schedule": job["schedule"],
                    "sample_urls": job["urls"][:2] if job["urls"] else []
                })
            
            return {
                "jobs": response,
                "pagination": {
                    "total": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_more": (offset + len(response)) < total_count
                }
            }
    
    except Exception as e:
        logger.error(f"Error listing crawl jobs: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list crawl jobs: {str(e)}"
        )

@app.get("/crawl/status/{job_id}", response_model=CrawlResponse)
async def get_crawl_status(
    job_id: str,
):
    """
    Get the current status and details of a crawl job.
    
    This endpoint returns detailed information about a crawl job including its current status,
    progress statistics, and any errors encountered during crawling.
    """
    try:
        # Get job details from database
        async with engine.begin() as conn:
            result = await conn.execute(
                text("""
                    SELECT 
                        job_id, status, urls, filters, max_depth, max_pages,
                        schedule, callback_url, started_at, completed_at, error,
                        stats, created_at, updated_at, next_scheduled, metadata
                    FROM crawl_jobs 
                    WHERE job_id = :job_id
                """),
                {"job_id": job_id}
            )
            job_row = result.mappings().first()
            
            if not job_row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Crawl job {job_id} not found"
                )
            
            # Convert row to dict for easier manipulation
            job = dict(job_row)
            
            # Calculate processing time
            processing_time = None
            if job["started_at"]:
                end_time = job["completed_at"] or datetime.utcnow()
                if isinstance(end_time, str):
                    end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                if isinstance(job["started_at"], str):
                    job["started_at"] = datetime.fromisoformat(job["started_at"].replace('Z', '+00:00'))
                delta = end_time - job["started_at"]
                processing_seconds = int(delta.total_seconds())
                processing_time = f"{processing_seconds // 60}m {processing_seconds % 60}s"
            
            # Prepare response data
            response_data = {
                "job_id": job_id,
                "status": job["status"],
                "created_at": job["created_at"].isoformat() if job["created_at"] else None,
                "started_at": job["started_at"].isoformat() if job["started_at"] else None,
                "completed_at": job["completed_at"].isoformat() if job["completed_at"] else None,
                "processing_time": processing_time,
                "next_scheduled": job["next_scheduled"].isoformat() if job["next_scheduled"] else None,
                "url_count": len(job["urls"]) if job["urls"] else 0,
                "max_depth": job["max_depth"],
                "max_pages": job["max_pages"],
                "schedule": job["schedule"],
                "callback_url": job["callback_url"],
                "metadata": job["metadata"] or {}
            }
            
            # Add statistics if available
            stats = job.get("stats") or {}
            if stats:
                response_data.update({
                    "pages_crawled": stats.get("pages_crawled", 0),
                    "pages_failed": stats.get("pages_failed", 0),
                    "chunks_created": stats.get("chunks_created", 0),
                    "domains_crawled": stats.get("domains", {}),
                    "content_types": stats.get("content_types", {}),
                    "total_content_size": stats.get("total_content_size", 0),
                    "average_chunk_size": stats.get("average_chunk_size", 0)
                })
            
            # Add error details if available
            if job["error"]:
                response_data["error"] = {
                    "message": job["error"],
                    "timestamp": job["completed_at"].isoformat() if job["completed_at"] else None
                }
            
            # Add sample URLs if available
            if job["urls"]:
                response_data["sample_urls"] = job["urls"][:5]
            
            return response_data
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for crawl job {job_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get crawl job status: {str(e)}"
        )

@app.post("/search", response_model=SearchResponse)
async def search_web(request: SearchRequest):
    """Search web for current information using SearxNG.
    
    This endpoint provides real-time web search capabilities by querying multiple
    search engines through the SearxNG metasearch engine. It respects the source_types
    filter to return only results matching the specified content categories.
    """
    try:
        # Map source_types to SearxNG categories
        category_map = {
            "news": ["news"],
            "official": ["general"],  # Official sites often rank high in general search
            "academic": ["science", "academic"],
            "forum": ["social media", "qna"],
            "video": ["videos"],
            "image": ["images"],
            "map": ["map"]
        }
        
        # Determine categories based on source_types
        categories = []
        if request.source_types:
            for source_type in request.source_types:
                if source_type in category_map:
                    categories.extend(category_map[source_type])
        
        # Ensure we have at least one category
        if not categories:
            categories = ["general"]
            
        # Perform the search
        results = await search_searxng(
            query=request.query,
            max_results=request.max_results,
            categories=list(set(categories))  # Remove duplicates
        )
        
        # Filter results by domain if needed (e.g., for official sources)
        if "official" in (request.source_types or []):
            official_domains = [".gov", ".edu", ".org"]
            results = [r for r in results if any(d in r["domain"] for d in official_domains)]
        
        return SearchResponse(results=results)
    
    except Exception as e:
        logger.error(f"Error searching web: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to perform web search")

@app.post("/crawl/stop", response_model=Dict[str, Any])
async def stop_crawl_job(
    job_id: str,
):
    """
    Stop a running crawl job.
    
    This will attempt to gracefully stop a crawl job that is currently in progress.
    The job status will be updated to 'stopping' and any in-progress operations
    will be allowed to complete.
    """
    try:
        async with engine.begin() as conn:
            # Get job details and verify ownership
            result = await conn.execute(
                text("""
                    SELECT job_id, user_id, status, started_at, completed_at 
                    FROM crawl_jobs 
                    WHERE job_id = :job_id
                    FOR UPDATE
                """),
                {"job_id": job_id}
            )
            job = result.mappings().first()
            
            if not job:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Crawl job {job_id} not found"
                )
            
            # Check permissions
            if not current_user.get("is_admin") and str(job["user_id"]) != str(current_user.get("user_id")):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have permission to stop this crawl job"
                )
            
            # Check if job can be stopped
            if job["status"] not in [CrawlJobStatus.PENDING.value, CrawlJobStatus.RUNNING.value]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cannot stop job with status '{job['status']}'"
                )
            
            # Update job status to stopping (will be updated to stopped by the worker)
            now = datetime.utcnow()
            await conn.execute(
                text("""
                    UPDATE crawl_jobs 
                    SET 
                        status = 'stopping',
                        updated_at = :now,
                        completed_at = CASE 
                            WHEN :completed IS NOT NULL THEN :completed 
                            ELSE completed_at 
                        END
                    WHERE job_id = :job_id
                """),
                {
                    "job_id": job_id,
                    "now": now,
                    "completed": now if job["status"] == CrawlJobStatus.RUNNING.value else None
                }
            )
            
            # TODO: Implement actual job cancellation logic
            # This would involve tracking the background task and cancelling it
            
            logger.info(f"Crawl job {job_id} stop requested by user {current_user.get('user_id')}")
            
            return {
                "job_id": job_id,
                "status": "stopping",
                "message": f"Crawl job {job_id} is being stopped"
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping crawl job {job_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop crawl job: {str(e)}"
        )

@app.delete("/crawl/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_crawl_job(
    job_id: str,
):
    """
    Delete a crawl job and its associated data.
    
    This will permanently remove the crawl job record and any associated
    crawled content from the vector store.
    """
    try:
        async with engine.begin() as conn:
            # Get job details and verify ownership
            result = await conn.execute(
                text("""
                    SELECT job_id, user_id, status 
                    FROM crawl_jobs 
                    WHERE job_id = :job_id
                    FOR UPDATE
                """),
                {"job_id": job_id}
            )
            job = result.mappings().first()
            
            if not job:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Crawl job {job_id} not found"
                )
            
            # Check permissions
            if not current_user.get("is_admin") and str(job["user_id"]) != str(current_user.get("user_id")):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have permission to delete this crawl job"
                )
            
            # Prevent deletion of running jobs
            if job["status"] == CrawlJobStatus.RUNNING.value:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot delete a running crawl job. Stop the job first."
                )
            
            # Delete associated data in the background
            asyncio.create_task(delete_crawl_job_data(job_id, current_user.get("user_id")))
            
            # Delete the job record
            await conn.execute(
                text("DELETE FROM crawl_jobs WHERE job_id = :job_id"),
                {"job_id": job_id}
            )
            
            logger.info(f"Crawl job {job_id} deleted by user {current_user.get('user_id')}")
            
            return Response(status_code=status.HTTP_204_NO_CONTENT)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting crawl job {job_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete crawl job: {str(e)}"
        )

async def delete_crawl_job_data(job_id: str, user_id: str):
    """
    Delete all data associated with a crawl job.
    
    This includes:
    - Web pages from the database
    - Document chunks from the vector store
    - Any temporary files or cached data
    """
    try:
        logger.info(f"Starting cleanup for crawl job {job_id}")
        
        # Delete web pages from database
        async with engine.begin() as conn:
            # Get page IDs first to delete associated chunks
            result = await conn.execute(
                text("""
                    DELETE FROM web_pages 
                    WHERE crawl_job_id = :job_id
                    RETURNING page_id
                """),
                {"job_id": job_id}
            )
            page_ids = [row[0] for row in result]
            
            # Delete associated document chunks
            if page_ids:
                await conn.execute(
                    text("""
                        DELETE FROM document_chunks 
                        WHERE metadata->>'crawl_job_id' = :job_id
                        OR metadata->>'page_id' = ANY(:page_ids)
                    """),
                    {"job_id": job_id, "page_ids": page_ids}
                )
        
        # TODO: Clean up any temporary files or cached data
        
        logger.info(f"Successfully cleaned up data for crawl job {job_id}")
        
    except Exception as e:
        logger.error(f"Error cleaning up data for crawl job {job_id}: {str(e)}", exc_info=True)
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        
        # Check Bedrock
        bedrock.list_foundation_models()
        
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/crawl")
async def crawl_endpoint(request: CrawlRequest):
    """Crawl a web page and store its content."""
    try:
        # Get database session
        db = next(get_db())
        
        try:
            # Check if page already exists
            existing_page = await get_web_page(db, request.url)
            if existing_page and not request.force_refresh:
                return CrawlResponse(
                    url=existing_page.url,
                    title=existing_page.title,
                    chunks=len(existing_page.chunks),
                    status="existing"
                )
            
            # Crawl the page
            content = await crawl_page(request.url)
            if not content:
                raise HTTPException(status_code=400, detail="Failed to crawl page")
            
            # Extract title and clean content
            title = extract_title(content)
            cleaned_content = clean_content(content)
            
            # Split into chunks
            chunks = split_into_chunks(cleaned_content)
            
            # Store page
            page = await store_web_page(
                db,
                request.url,
                title,
                cleaned_content,
                {"crawl_date": datetime.utcnow().isoformat()}
            )
            
            # Store chunks with embeddings
            for chunk in chunks:
                embedding = get_embedding(chunk)
                await store_web_chunk(
                    db,
                    page.id,
                    chunk,
                    embedding,
                    {"chunk_index": chunks.index(chunk)}
                )
            
            return CrawlResponse(
                url=request.url,
                title=title,
                chunks=len(chunks),
                status="crawled"
            )
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error crawling page: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 