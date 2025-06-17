# Fizen RAG Deep Technical Specification - Upgraded (FAISS + Ollama)

## Objective

Fizen RAG (Retrieval-Augmented Generation) is an enterprise-ready architecture that enables intelligent question-answering from both static and live data sources. It combines LLM capabilities with structured retrieval, allowing businesses to:

* Extract precise, explainable answers from large knowledge bases
* Integrate document, API, SQL, and live web information
* Maintain context-rich, conversational interfaces for users

This specification is intended for developers building the system and testers verifying correctness and performance.

## Technology Stack

### Frontend

* **React** – Interactive UI for users and admins
* **Tailwind CSS** – Modern, responsive styling
* **Axios** – API communication handler

### Backend

* **FastAPI** – High-performance Python backend API
* **aiohttp** – Async HTTP client for live web crawling
* **SQLAlchemy** – ORM for database access
* **Pandas/Numpy** – Data transformation & tabular logic

### AI/ML

* **BGE-Small (via Hugging Face)** – Lightweight sentence embedding model
* **Ollama** – Local LLM inference (Llama2, Mistral, etc.)
* **Optional: LangChain** – Prompt orchestration and chaining

### Storage & Indexing

* **FAISS** – High-performance vector similarity search library
* **PostgreSQL** – Database for structured data and metadata
* **Local File System** – Storage for uploaded files and FAISS indices

### DevOps

* **Docker & Docker Compose** – Local containerization and orchestration
* **Prometheus + Grafana** – Monitoring and alerting

## 1. Frontend

### 1.1 Admin Dashboard (Admin Access Only)
- Built with React, Material-UI, and TypeScript.
- **Administrative Access Control**: JWT-based authentication with admin role validation
- Features:
  - **Document Upload Management**:
    - File/document upload interface (PDF, DOCX, TXT)
    - Upload progress tracking and validation
    - Document metadata management
    - Bulk upload support
  - **Web Crawling Management**:
    - URL submission with crawl parameters (max pages, depth, filters)
    - Domain whitelist/blacklist configuration
    - Real-time crawl progress monitoring
    - Crawl scheduling and automation
  - **Data Management**:
    - Real-time job status monitoring (uploads, crawls, ingestion)
    - FAISS index statistics and health monitoring
    - Content search and preview capabilities
    - Data cleanup and maintenance tools
  - **User Management**: Authentication and role-based access control
- Uses a centralized API service for all backend communication.

### 1.2 Traveler Interface (Public Access)
- **Open Access**: Available to all users without authentication requirements
- Chat-based UI for anyone to interact with the RAG system
- Features:
  - **Interactive Chat**: Natural language conversation interface
  - **Session Management**: Automatic session creation and history tracking
  - **Real-time Responses**: WebSocket-based streaming responses
  - **Search Capabilities**: Query across all ingested content (documents + web)
  - **Source Citations**: Transparent source attribution for all answers
  - **Mobile Responsive**: Optimized for desktop and mobile devices
- **Read-Only Access**: Users can only query and receive information
- **No Content Management**: Users cannot upload, modify, or delete content
- Real-time typing indicators and response streaming.

## 2. Access Control & Session Management

### 2.1 Public User Access
- **No Authentication Required**: Anyone can access the chat interface
- **Anonymous Sessions**: Automatically created for each visitor
- **Session Duration**: 24-hour expiration with activity-based extension
- **Rate Limiting**: Applied per IP address to prevent abuse
- **Read-Only Access**: Users can only query existing content
- **Session Tracking**: IP address and user agent logged for analytics
- **Privacy**: No personal data collection beyond session management

### 2.2 Admin Access Control
- **JWT Authentication**: Secure token-based authentication required
- **Role-Based Permissions**: Admin role required for content management
- **Secure Endpoints**: All admin operations require valid authentication
- **Token Expiration**: Configurable session timeouts for security
- **Audit Logging**: All admin actions logged for compliance
- **Multi-Admin Support**: Multiple admin accounts with same permissions

### 2.3 Content Access Policy
- **Universal Content Access**: All ingested content available to public users
- **Source Attribution**: All responses include proper source citations
- **Content Filtering**: Admin-configurable content filters (if needed)
- **No Data Exposure**: Raw documents/sources not directly accessible
- **Processed Content Only**: Users only see processed, chunked responses

## 3. API Gateway
- Built with FastAPI.
- Exposes REST and WebSocket endpoints for chat, document upload, crawling, and job management.
- **Role-Based Access Control**: JWT-based authentication with admin role validation for administrative endpoints.
- Forwards requests to appropriate microservices and aggregates responses.
- Example endpoints:
  - **Public Endpoints (No Authentication Required)**:
    - `/chat` (POST): Chat with the system
    - `/chat/stream` (WebSocket): Streaming chat responses
    - `/sessions/history/{session_id}` (GET): Retrieve public session history
    - `/health` (GET): Health check for all services
  - **Admin-Only Endpoints (Requires Admin Authentication)**:
    - `/admin/documents/upload` (POST): Upload documents for processing
    - `/admin/documents/status/{doc_id}` (GET): Check processing status
    - `/admin/documents/list` (GET): List all processed documents
    - `/admin/documents/delete/{doc_id}` (DELETE): Remove document and chunks
    - `/admin/crawl/start` (POST): Start a web crawl job
    - `/admin/crawl/status/{job_id}` (GET): Check crawl progress
    - `/admin/crawl/jobs` (GET): List all crawl jobs
    - `/admin/crawl/stop/{job_id}` (POST): Stop active crawl
    - `/admin/stats` (GET): Database and FAISS index statistics
    - `/admin/metrics` (GET): System performance metrics
    - `/admin/users` (GET/POST): User management endpoints
  - **Authentication Endpoints**:
    - `/auth/login` (POST): Admin login
    - `/auth/refresh` (POST): Token refresh
    - `/auth/logout` (POST): Admin logout

## 4. Component Overview with Input, Output, Internal Logic & Example

### 4.1 Context Manager

**Purpose**:
To maintain natural conversation flow and enrich queries with relevant previous interactions for both authenticated admin users and anonymous public users. This ensures continuity (e.g., follow-up questions) and improves retrieval accuracy.

**Internal Logic:**
1. Query latest N messages from the session (PostgreSQL) - supports both user-based and anonymous sessions.
2. Extract relevant messages using Vector Storage & Retrieval Service to query embedded chat history.
3. Concatenate top matches into a summarized history block.
4. Return context+query for retrieval.

**How it works**:
1. Fetch previous messages for session from PostgreSQL (anonymous sessions auto-created).
2. Send current query to Vector Storage & Retrieval Service for similarity search against chat_messages table.
3. Vector Storage & Retrieval Service:
   - Embeds current query using BGE-small
   - Performs FAISS similarity search against stored chat message embeddings
   - Returns top-k most relevant previous messages (e.g. k=3) with similarity scores
4. Format relevant chat history into a context block to attach to current query.

**Input (Anonymous Public User)**:
```json
{ "session_id": "anonymous_789", "query": "And how do we issue refunds?" }
```

**Output**:
```json
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
```

### 4.2 Retrieval Orchestrator

**Purpose**:
To unify diverse knowledge sources (documents, databases, web) into a coherent evidence set for generation. It ensures multi-source fact gathering.

**Internal Logic:**
1. Calls Vector Storage & Retrieval Service with query embedding.
2. Calls SQL Retriever with natural query and schema metadata.
3. Calls Web Retriever using predefined trusted domain filters.
4. Deduplicates results using fingerprint hash or sentence similarity.
5. Sorts based on semantic relevance and recency.
6. Returns unified evidence pool for prompt building.

**How it works**:
1. Forwards enhanced query to:
   - Vector Storage & Retrieval Service (docs)
   - SQL Retriever (databases)
   - Web Retriever (external trusted info)
2. Collects, deduplicates (hash/Jaccard), ranks evidence.
3. Returns clean, sorted result list.

**Input**:
```json
{ "query": "How many returns were made in Q1?", "context": "..." }
```

**Output**:
```json
[
  { "type": "sql", "data": [{ "month": "Jan", "returns": 130 }] },
  { "type": "vector", "text": "Returns report for Q1 shows Jan-Mar..." }
]
```

### 4.3 Vector Storage & Retrieval Service

**Purpose**:
Unified service that handles document ingestion (chunking + embedding), real-time vector retrieval, and chat context retrieval using FAISS. Manages the complete lifecycle of all vectorized content from upload to query response.

**Internal Logic:**

**For Document Ingestion Mode:**
1. Extract raw text using Tika, PDFMiner, or Textract.
2. Normalize whitespace, remove headers/footers.
3. Use recursive character splitting with overlap (LangChain/TextSplitter).
4. Embed each chunk with BGE-small.
5. Store in PostgreSQL (metadata) and FAISS index (vectors): text, embedding, doc_id, chunk_id, page, section, timestamp.

**For Query Retrieval Mode:**
1. Compute query embedding with BGE-small.
2. Perform vector search using FAISS index across document chunks.
3. Apply confidence threshold (e.g., score > 0.75).
4. Attach metadata from PostgreSQL: document_id, source_url, section_id.
5. Return top-k ranked chunks.

**For Chat Context Retrieval Mode:**
1. Compute query embedding with BGE-small.
2. Perform vector search using FAISS index across chat messages.
3. Filter by session_id and message_type = 'user'.
4. Apply confidence threshold (e.g., score > 0.7).
5. Return top-k relevant previous messages with timestamps.

**How it works:**

**Ingestion Flow:**
1. Extract full text from file (PDF, DOCX, TXT).
2. Normalize (remove headers, footers).
3. Use recursive chunk splitter (e.g. 500 tokens with 20 overlap).
4. Embed using BGE-small (CPU).
5. Store chunk metadata in PostgreSQL and embeddings in FAISS index.

**Document/Web Content Retrieval Flow:**
1. Embed input query using BGE-small.
2. Perform top-k similarity search using FAISS index on document embeddings.
3. Filter results with score > 0.75.
4. Fetch chunk metadata from PostgreSQL using chunk IDs.

**Chat Context Retrieval Flow:**
1. Embed input query using BGE-small.
2. Perform top-k similarity search using FAISS index on chat message embeddings.
3. Filter by session_id and message_type = 'user' from PostgreSQL.
4. Filter results with score > 0.7.
5. Return relevant chat history with timestamps.

**Input (Ingestion Mode)**:
PDF with 15 pages

**Output (Ingestion Mode)**:
```json
{
  "status": "success",
  "chunks_created": 45,
  "doc_id": "returns_policy_v2.pdf",
  "faiss_index_updated": true,
  "chunks": [
    { 
      "chunk_id": "returns_policy_v2.pdf_chunk_001", 
      "text": "Returns must be made in 30 days...", 
      "doc_id": "returns_policy_v2.pdf", 
      "page": 3,
      "embedding_stored": true
    }
  ]
}
```

**Input (Document/Web Content Retrieval Mode)**:
```json
{ "query": "When can customers return products?" }
```

**Output (Document/Web Content Retrieval Mode)**:
```json
[
  {
    "text": "Customers may return items within 30 days of purchase.",
    "source": "returns_policy_v2.pdf",
    "score": 0.91,
    "chunk_id": "returns_policy_v2.pdf_chunk_001",
    "page": 3
  },
  {
    "text": "Items must be in original packaging for returns.",
    "source": "returns_policy_v2.pdf", 
    "score": 0.87,
    "chunk_id": "returns_policy_v2.pdf_chunk_012",
    "page": 5
  }
]
```

**Input (Chat Context Retrieval Mode)**:
```json
{ 
  "query": "And how do we issue refunds?",
  "session_id": "anonymous_789",
  "context_type": "chat_history"
}
```

**Output (Chat Context Retrieval Mode)**:
```json
[
  {
    "message": "What is your return policy?",
    "timestamp": "2024-01-08T14:15:30Z",
    "score": 0.89,
    "message_type": "user"
  },
  {
    "message": "How long do customers have to return items?",
    "timestamp": "2024-01-08T14:12:15Z", 
    "score": 0.82,
    "message_type": "user"
  }
]
```

### 4.4 SQL Retriever

**Purpose**:
Allow non-technical users to extract structured information from internal databases via natural language. Prevents manual SQL writing.

**Internal Logic:**
1. Fetch schema map for the target database.
2. Use prompt template to convert natural query to SQL via Ollama.
3. Parse and validate SQL syntax using sqlparse or similar.
4. Execute SQL query using SQLAlchemy (read-only mode).
5. Transform result into JSON + attach table metadata for LLM.

**How it works**:
1. Extract table schema.
2. Prompt Ollama (Llama2/Mistral) to generate SQL.
3. Validate using `sqlparse` or AST.
4. Execute SQL (read-only).
5. Return JSON + schema info.

**Input**:
```
"What were the top 5 products by revenue last month?"
```

**Output**:
```json
[
  { "product": "XPhone", "revenue": 12000 },
  { "product": "TabletZ", "revenue": 9800 }
]
```

### 4.5 Web Retriever

**Purpose**:
Dual-mode service that handles both automated web crawling for knowledge base building and real-time web search for current information. Manages the complete web content lifecycle from crawl configuration to search integration.

**Internal Logic:**

**Mode 1: Automated Web Crawling (Knowledge Base Building)**
1. **Crawl Configuration**: Accept crawl job parameters from admin dashboard
2. **Content Retrieval**: Systematically crawl target websites using aiohttp
3. **Content Processing**: Extract, clean, and process web content for storage
4. **Deduplication & Storage**: Store unique content in FAISS via Vector Storage Service
5. **Monitoring**: Track crawl progress and provide real-time status updates

**Mode 2: Real-Time Web Search (Query Enhancement)**
1. Query Bing/GNews API for current information related to user queries
2. Filter results against trusted domain whitelist
3. Extract and summarize content for immediate use in responses
4. Return fresh data to supplement knowledge base content

**How it works:**

**Automated Crawling Flow:**
1. **Job Setup**: Receive crawl configuration from API Gateway
   - Target URLs or domains
   - Max pages and crawl depth limits
   - Content filters and exclusion patterns
   - Crawl frequency (one-time or scheduled)
2. **Content Retrieval**: For each URL in crawl queue:
   - Fetch HTML content with proper user-agent headers
   - Respect robots.txt and implement rate limiting
   - Extract main text using readability-lxml
   - Filter out navigation, ads, and boilerplate content
   - Validate content against domain whitelist
3. **Content Processing**: 
   - Send extracted text to Vector Storage & Retrieval Service
   - Apply same processing pipeline as document uploads:
     - Text cleaning and normalization
     - Semantic chunking with overlap
     - BGE-small embedding generation
   - Enrich with web-specific metadata (URL, crawl timestamp, page title)
4. **Deduplication & Storage**:
   - Check for existing content using text similarity
   - Store only new unique chunks in FAISS index
   - Update web-specific metadata tables in PostgreSQL
5. **Progress Monitoring**:
   - Track crawl progress in real-time
   - Log failed URLs with detailed error reasons
   - Make successful pages immediately available for search
   - Update admin dashboard with crawl statistics

**Real-Time Search Flow:**
1. Call Bing API with user query for current information
2. Apply domain filters (gov, org, edu, whitelist)
3. Download HTML, parse with BeautifulSoup
4. Extract main text using readability-lxml
5. If content > 2,000 words, summarize with Ollama
6. Return URLs and relevant text snippets

**Input (Crawl Configuration Mode)**:
```json
{
  "urls": ["https://company.com/help", "https://company.com/policies"],
  "max_pages": 50,
  "max_depth": 3,
  "filters": {
    "exclude_patterns": ["/admin", "/private"],
    "content_types": ["text/html"],
    "min_content_length": 200
  },
  "schedule": "weekly",
  "respect_robots": true,
  "rate_limit_delay": 1.0
}
```

**Output (Crawl Configuration Mode)**:
```json
{
  "crawl_job_id": "crawl_002",
  "status": "completed",
  "pages_crawled": 47,
  "pages_failed": 3,
  "chunks_created": 234,
  "processing_time": "12m 34s",
  "next_scheduled": "2024-01-15T09:00:00Z",
  "failed_urls": [
    {
      "url": "https://company.com/restricted",
      "error": "403 Forbidden",
      "timestamp": "2024-01-08T14:23:45Z"
    }
  ],
  "crawl_statistics": {
    "total_content_size": "2.4MB",
    "average_chunk_size": 512,
    "domains_crawled": ["company.com"],
    "content_types_found": ["text/html", "application/pdf"]
  }
}
```

**Input (Real-Time Search Mode)**:
```json
{ 
  "query": "Latest GDPR updates in 2024",
  "max_results": 5,
  "source_types": ["news", "official"]
}
```

**Output (Real-Time Search Mode)**:
```json
[
  {
    "url": "https://gdpr.eu/update-2024",
    "title": "GDPR Updates March 2024",
    "summary": "As of March 2024, companies must notify users within 24 hours of breach.",
    "domain": "gdpr.eu",
    "content_type": "official",
    "crawled_at": "2024-01-08T15:30:00Z",
    "relevance_score": 0.92
  }
]
```

### 4.6 Prompt Builder

**Purpose**:
Build an optimal LLM prompt that blends query + top evidence + instructions.

**Internal Logic:**
1. Template includes:
   - User query
   - Top context summary (if exists)
   - Top 3-5 evidence chunks (with source)
2. Chunks are concatenated in order of relevance.
3. Adds formatting instructions (e.g., JSON answer, citations).
4. Applies truncation to stay under token limit.

**How it works**:
1. Add system role: "You are a helpful enterprise assistant."
2. Add context: top user history lines.
3. Add top-3 evidence:
```text
[1] "Returns must be made within 30 days"
[2] "SQL: 130 items returned in Jan"
```
4. Format instructions: "Reply in markdown. Add citations like [1]".

**Input**:
```json
{
  "query": "How many items were returned?",
  "evidence": [...]
}
```

**Output (prompt)**:
```
Context: Customer asked about refunds.
Evidence:
[1] "130 returns in Jan"
[2] "Return period: 30 days"
Q: How many were returned?
A: In January, there were 130 returns. Customers can return within 30 days [1][2].
```

### 4.7 LLM Orchestrator

**Purpose**:
Manage LLM access (Llama2/Mistral via Ollama), handle retries, latency, and failover.

**Internal Logic:**
1. Sends prompt to Ollama model (e.g., Llama2, Mistral).
2. Handles retries on connection errors.
3. Monitors token usage and logs latency.
4. Supports multi-step chaining (follow-up Qs, refinement).
5. Returns raw LLM output.

**How it works**:
1. Send prompt to Ollama endpoint.
2. Retry up to 3 times on failure (connection timeout).
3. Log latency, token usage to local metrics.
4. Support chaining or follow-up memory state.

**Input**:
```
Prompt with query + context + evidence
```

**Output**:
```
"In January, there were 130 returned items. Policy allows 30-day returns [1]."
```

### 4.8 Response Formatter

**Purpose**:
Format the LLM's raw response for display (Markdown, JSON, citations).

**Internal Logic:**
1. Parse for citation markers like [1], [2] if present.
2. Map citations to original chunks or URLs.
3. Escape/format content for frontend rendering.
4. If requested format = JSON, run JSON validator.
5. Return final answer with references and style hints.

**How it works**:
1. Parse references like [1], [2].
2. Link citations to metadata (chunk source or URL).
3. Render as Markdown or safe HTML.
4. If needed JSON: Validate with schema.

**Input**:
```
"In Jan, 130 items were returned [1]."
```

**Output**:
```json
{
  "answer": "In Jan, 130 items were returned [1].",
  "citations": [{ "ref": "[1]", "doc": "sql", "value": "return stats Jan 2024" }]
}
```

## 5. Complete Data Ingestion Flows

### 5.1 Document Processing Flow

**Purpose**: Convert uploaded documents into searchable vector representations stored in FAISS.

**Flow Steps**:

1. **Document Upload**
   - Admin uploads document via React dashboard
   - API Gateway validates file type (PDF, DOCX, TXT, etc.)
   - File is stored in local file system
   - Upload job is queued with unique job_id

2. **Document Processing**
   - Vector Storage & Retrieval Service receives processing request
   - Service extracts raw text using appropriate parser:
     - PDF: PDFMiner or PyPDF2
     - DOCX: python-docx
     - TXT: Direct text extraction
   - Text is cleaned and normalized (remove headers, footers, extra whitespace)

3. **Chunking & Embedding**
   - Service applies recursive character splitting (500 tokens, 20 overlap)
   - Each chunk is processed with BGE-small to generate 384-dim embeddings
   - Chunks are enriched with metadata:
     - Document source, page numbers
     - Section titles and headers
     - Processing timestamp

4. **Storage in FAISS**
   - Document record created in `documents` table (PostgreSQL)
   - Each chunk metadata stored in `document_chunks` table (PostgreSQL)
   - Embeddings added to FAISS index with unique chunk IDs
   - FAISS index persisted to disk for durability

5. **Completion & Notification**
   - Document status updated to 'completed'
   - Admin dashboard receives real-time update
   - Vector search immediately available for new content

**Input Example**:
```json
{
  "file": "company_policies.pdf",
  "metadata": {
    "department": "HR",
    "version": "2.1",
    "classification": "internal"
  }
}
```

**Output Example**:
```json
{
  "job_id": "doc_001",
  "status": "completed",
  "doc_id": "company_policies_v21.pdf",
  "chunks_created": 156,
  "processing_time": "45.2s",
  "embedding_dimensions": 384,
  "faiss_index_size": 156
}
```

### 5.2 Web Crawling Flow

**Purpose**: Automatically crawl and index web content from trusted sources into the knowledge base.

**Flow Steps**:

1. **Crawl Configuration**
   - Admin configures crawl job via dashboard:
     - Target URLs or domains
     - Max pages and crawl depth limits
     - Content filters and exclusion patterns
     - Crawl frequency (one-time or scheduled)
   - API Gateway validates parameters and checks rate limits

2. **Content Retrieval**
   - Web Retriever initiates crawl using aiohttp
   - For each URL:
     - Fetch HTML content with user-agent headers
     - Respect robots.txt and rate limiting
     - Extract main text using readability-lxml
     - Filter out navigation, ads, and boilerplate
   - Content is validated against domain whitelist

3. **Content Processing**
   - Extracted text sent to Vector Storage & Retrieval Service
   - Service applies same processing as document upload:
     - Text cleaning and normalization
     - Semantic chunking with overlap
     - BGE-small embedding generation
   - Metadata includes URL, crawl timestamp, page title

4. **Deduplication & Storage**
   - Service checks for existing content using text similarity
   - New unique chunks stored in FAISS index
   - Web-specific metadata stored in PostgreSQL:
     - Source URL and domain
     - Last crawled timestamp
     - Content freshness indicators

5. **Monitoring & Updates**
   - Crawl progress tracked in real-time
   - Failed URLs logged with error reasons
   - Successful pages immediately available for search
   - Admin dashboard shows crawl statistics

**Input Example**:
```json
{
  "urls": ["https://company.com/help", "https://company.com/policies"],
  "max_pages": 50,
  "max_depth": 3,
  "filters": {
    "exclude_patterns": ["/admin", "/private"],
    "content_types": ["text/html"]
  },
  "schedule": "weekly"
}
```

**Output Example**:
```json
{
  "crawl_job_id": "crawl_002",
  "status": "completed",
  "pages_crawled": 47,
  "pages_failed": 3,
  "chunks_created": 234,
  "processing_time": "12m 34s",
  "next_scheduled": "2024-01-15T09:00:00Z",
  "faiss_index_updated": true
}
```

### 5.3 Unified Query Processing Flow

**Purpose**: Handle user queries by searching across all ingested content (documents + web) using unified FAISS retrieval.

**Flow Steps**:

1. **Query Reception**
   - User submits query via chat interface
   - API Gateway validates request and applies rate limiting
   - Query includes user context and optional filters

2. **Context Integration**
   - Context Manager directly calls Vector Storage & Retrieval Service to retrieve relevant conversation history
   - Vector Storage & Retrieval Service embeds current query and performs FAISS similarity search against chat message embeddings
   - Context Manager receives relevant previous messages and formats them into context block
   - Context enriches current query for better search results

3. **Parallel Multi-Source Retrieval**
   - Context Manager passes enriched query (with context) to Retrieval Orchestrator
   - Retrieval Orchestrator coordinates multiple searches:
     - **Vector Search**: FAISS similarity search across all document chunks via Vector Storage & Retrieval Service
     - **SQL Search**: Structured data queries for specific metrics
     - **Web Search**: Real-time search for current information
   - Each source returns ranked results with confidence scores

4. **Result Unification**
   - Retrieval Orchestrator processes all results:
     - Deduplicates using content similarity (>0.9 threshold)
     - Applies multi-factor ranking:
       * Original relevance scores
       * Content recency (web content weighted higher)
       * Source authority (internal docs > web content)
       * User context relevance
     - Combines top-k results from each source

5. **Response Generation**
   - Prompt Builder formats unified results with source citations
   - LLM Orchestrator processes prompt via Ollama
   - Response Formatter adds proper citations and metadata
   - Final answer includes evidence from multiple sources

**Query Input Example (Public User)**:
```json
{
  "query": "What is our remote work policy for international employees?",
  "session_id": "anonymous_session_456",
  "filters": {
    "source_types": ["documents", "web"],
    "departments": ["HR", "Legal"]
  }
}
```

**Response Output Example (Public User)**:
```json
{
  "answer": "Our remote work policy allows international employees to work remotely with manager approval [1]. Additional visa considerations apply for work in different countries [2].",
  "sources": [
    {
      "id": "[1]",
      "text": "International remote work requires manager approval and HR review",
      "source": "HR_Policy_2024.pdf",
      "type": "document",
      "score": 0.94
    },
    {
      "id": "[2]", 
      "text": "Visa requirements vary by country for remote work",
      "source": "https://company.com/legal/remote-work",
      "type": "web",
      "score": 0.87
    }
  ],
  "processing_time": "1.2s",
  "session_id": "anonymous_session_456"
}
```

## 6. System Architecture Diagrams

### 6.1 Overall Data Flow Diagram (ASCII)

```
                    +-----------------------+
                    |    User Interfaces    |
                    | - Public Chat (Open)  |
                    | - Admin Dashboard     |
                    |   (Auth Required)     |
                    +----------+------------+
                               |
                               v
                  +------------+-------------+
                  |       API Gateway        |
                  | - Public Endpoints       |
                  | - Admin Authentication   |
                  | - Rate Limiting          |
                  | - Request Routing        |
                  +------------+-------------+
                               |
        +----------------------+----------------------+
        |                      |                      |
        v                      v                      v
+-------+-------+    +---------+---------+    +-------+-------+
|Public Chat    |    |Admin Content Mgmt |    |Admin Web Mgmt |
|Pipeline       |    |Pipeline           |    |Pipeline       |
|(No Auth)      |    |(Admin Only)       |    |(Admin Only)   |
+-------+-------+    +---------+---------+    +-------+-------+
        |                      |                      |
        v                      v                      v
+-------+-------+    +---------+---------+    +-------+-------+
|Context Manager|    |   File Upload     |    | Web Crawler   |
|(Anonymous OK) |    |   (Admin Only)    |    | (Admin Only)  |
+-------+-------+    +---------+---------+    +-------+-------+
        |                      |                      |
        v                      |                      |
        |              +-------+-------+              |
        |              |Vector Storage |              |
        |              |& Retrieval    |              |
        |              |Service        |              |
        |              |(Chat Context) |              |
        |              +-------+-------+              |
        |                      |                      |
        v                      v                      v
+-------+-------+    +---------+---------+    +-------+-------+
|Retrieval Orch.|    |Vector Storage &   |    |Vector Storage |
|(Public Access)|<-->|Retrieval Service  |<-->|& Retrieval    |
|(w/ Context)   |    |(Ingestion Mode)   |    |(Web Mode)     |
+-------+-------+    |(Admin Only)       |    |(Admin Only)   |
        |            +-------------------+    +---------------+
        |                      |                      |
        |                      v                      |
        |            +---------+---------+            |
        |            | PostgreSQL        |            |
        |            | Database          |            |
        |            | (Metadata +       |            |
        |            |  Chat History)    |            |
        |            +---------+---------+            |
        |                      |                      |
        |                      v                      |
        |            +---------+---------+            |
        |            | FAISS Vector      |            |
        |            | Index             |            |
        |            | (All Embeddings)  |            |
        |            +---------+---------+            |
        |                      |                      |
        +----------------------+----------------------+
                               |
                               v
                  +------------+-------------+
                  |    Prompt Builder        |
                  |    (Public Access)       |
                  +------------+-------------+
                               |
                               v
                  +------------+-------------+
                  |   LLM Orchestrator       |
                  |   (Ollama)               |
                  |   (Public Access)        |
                  +------------+-------------+
                               |
                               v
                  +------------+-------------+
                  |   Response Formatter     |
                  |   (Public Access)        |
                  +------------+-------------+
                               |
                               v
                  +------------+-------------+
                  |     Final Answer         |
                  |  (Available to Public)   |
                  +-------------------------+
```

### 6.2 Document Processing Lifecycle Diagram (ASCII)

```
Admin Upload → API Gateway → File Validation → Local File Storage
                    |
                    v
           Document Processing Queue
                    |
                    v
    +---------------+---------------+
    |     Vector Storage &          |
    |     Retrieval Service         |
    |   (Document Ingestion Mode)   |
    +---------------+---------------+
                    |
    +---------------+---------------+
    | Text Extraction               |
    | - PDF: PDFMiner              |
    | - DOCX: python-docx          |
    | - TXT: Direct read           |
    +---------------+---------------+
                    |
                    v
    +---------------+---------------+
    | Text Processing               |
    | - Normalize whitespace        |
    | - Remove headers/footers      |
    | - Extract metadata           |
    +---------------+---------------+
                    |
                    v
    +---------------+---------------+
    | Semantic Chunking             |
    | - 500 token chunks            |
    | - 20 token overlap           |
    | - Preserve context           |
    +---------------+---------------+
                    |
                    v
    +---------------+---------------+
    | BGE-Small Embedding           |
    | - 384-dimensional vectors     |
    | - Batch processing           |
    +---------------+---------------+
                    |
                    v
    +---------------+---------------+
    | FAISS Index Storage           |
    | - Add embeddings to index     |
    | - PostgreSQL metadata        |
    | - Persist index to disk      |
    +---------------+---------------+
                    |
                    v
    +---------------+---------------+
    | Status Update & Notification  |
    | - Admin dashboard update      |
    | - Available for search        |
    +-------------------------------+
```

### 6.3 Web Crawling Lifecycle Diagram (ASCII)

```
Admin Config → API Gateway → Crawl Validation → Job Queue
                    |
                    v
           Web Crawling Scheduler
                    |
                    v
    +---------------+---------------+
    |        Web Retriever          |
    |     (Content Fetching)        |
    +---------------+---------------+
                    |
    +---------------+---------------+
    | URL Processing                |
    | - Respect robots.txt         |
    | - Rate limiting              |
    | - User-agent headers         |
    +---------------+---------------+
                    |
                    v
    +---------------+---------------+
    | Content Extraction            |
    | - HTML parsing               |
    | - readability-lxml           |
    | - Remove boilerplate         |
    +---------------+---------------+
                    |
                    v
    +---------------+---------------+
    | Content Validation            |
    | - Domain whitelist check     |
    | - Content type filter        |
    | - Duplicate detection        |
    +---------------+---------------+
                    |
                    v
    +---------------+---------------+
    |     Vector Storage &          |
    |     Retrieval Service         |
    |    (Web Content Mode)         |
    +---------------+---------------+
                    |
                    v
    +---------------+---------------+
    | Same Processing Pipeline      |
    | - Text normalization         |
    | - Semantic chunking          |
    | - BGE-Small embedding        |
    | - FAISS index storage        |
    +---------------+---------------+
                    |
                    v
    +---------------+---------------+
    | Crawl Completion              |
    | - Statistics logging         |
    | - Schedule next crawl        |
    | - Admin notification         |
    +-------------------------------+
```

### 6.4 Query Processing Lifecycle Diagram (ASCII)

```
User Query → API Gateway → [Context Manager]
                            ↓
                    [Vector Storage Service]
                            ↓
                    [Chat History FAISS Search]
                            ↓
                    [Relevant Context Retrieved]
                            ↓
                    [Context Manager + Query] → [Retrieval Orchestrator]
                                                        ↓
                                                [Multi-Source Search]
                                                        ↓
                                                [Vector Storage & Retrieval] 
                                                [SQL Retriever] 
                                                [Web Retriever] 
                                                [FAISS Index + PostgreSQL]
                                                        ↓
                                                [Result Deduplication]
                                                        ↓
                                                [Multi-Factor Ranking]
                                                        ↓
                                                [Source Unification]
                                                        ↓
                                                [Prompt Builder] 
                                                        ↓
                                                [LLM Orchestrator] 
                                                        ↓
                                                [Ollama (Llama2/Mistral)]
                                                        ↓
                                                [Response Formatter] 
                                                        ↓
                                                [Final Answer + Citations]
```

## 7. PostgreSQL Database Schema + FAISS Configuration

### 7.1 PostgreSQL Tables (Metadata Only)

```sql
-- Main documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(255) UNIQUE NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_chunks INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'processing',
    metadata JSONB DEFAULT '{}'
);

-- Web crawl jobs table
CREATE TABLE crawl_jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) UNIQUE NOT NULL,
    start_urls TEXT[] NOT NULL,
    max_pages INTEGER DEFAULT 50,
    max_depth INTEGER DEFAULT 3,
    status VARCHAR(50) DEFAULT 'pending',
    pages_crawled INTEGER DEFAULT 0,
    pages_failed INTEGER DEFAULT 0,
    chunks_created INTEGER DEFAULT 0,
    filters JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    next_scheduled TIMESTAMP,
    error_details TEXT
);

-- Web pages table
CREATE TABLE web_pages (
    id SERIAL PRIMARY KEY,
    page_id VARCHAR(255) UNIQUE NOT NULL,
    crawl_job_id VARCHAR(255) REFERENCES crawl_jobs(job_id),
    url TEXT NOT NULL,
    title VARCHAR(500),
    domain VARCHAR(255),
    crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'success',
    content_length INTEGER,
    chunk_count INTEGER DEFAULT 0
);

-- Document chunks metadata (no embeddings - those go to FAISS)
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255) UNIQUE NOT NULL,
    doc_id VARCHAR(255) REFERENCES documents(doc_id),
    page_id VARCHAR(255) REFERENCES web_pages(page_id),
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    section_title VARCHAR(255),
    content_type VARCHAR(50) DEFAULT 'document', -- 'document' or 'web'
    source_url TEXT,
    domain VARCHAR(255),
    faiss_index INTEGER NOT NULL, -- Index in FAISS vector database
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chat sessions and context (supports both authenticated and anonymous users)
CREATE TABLE chat_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255), -- NULL for anonymous sessions
    session_type VARCHAR(50) DEFAULT 'anonymous', -- 'authenticated' or 'anonymous'
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP + INTERVAL '24 hours')
);

CREATE TABLE chat_messages (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL REFERENCES chat_sessions(session_id),
    message_text TEXT NOT NULL,
    message_type VARCHAR(50) NOT NULL, -- 'user' or 'assistant'
    faiss_index INTEGER, -- Index in FAISS chat embeddings database
    response_time_ms INTEGER,
    sources_used JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processing jobs queue
CREATE TABLE processing_jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) UNIQUE NOT NULL,
    job_type VARCHAR(50) NOT NULL, -- 'document', 'crawl', 'chunk'
    status VARCHAR(50) DEFAULT 'pending',
    input_data JSONB NOT NULL,
    result_data JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    retry_count INTEGER DEFAULT 0
);
```

### 7.2 FAISS Configuration

```python
import faiss
import numpy as np
import pickle
import os

class FAISSVectorStore:
    def __init__(self, dimension=384, index_type="IVFFlat"):
        self.dimension = dimension
        self.index_type = index_type
        self.document_index = None
        self.chat_index = None
        self.document_id_map = []  # Maps FAISS indices to chunk_ids
        self.chat_id_map = []      # Maps FAISS indices to message_ids
        
        # Initialize indices
        self._init_document_index()
        self._init_chat_index()
    
    def _init_document_index(self):
        """Initialize FAISS index for documents"""
        if self.index_type == "IVFFlat":
            # For production: IndexIVFFlat with 100 clusters
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.document_index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "HNSW":
            # For high accuracy: IndexHNSWFlat
            self.document_index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            # For development: Simple flat index
            self.document_index = faiss.IndexFlatIP(self.dimension)
    
    def _init_chat_index(self):
        """Initialize FAISS index for chat messages"""
        # Simpler index for chat messages (typically smaller dataset)
        self.chat_index = faiss.IndexFlatIP(self.dimension)
    
    def add_documents(self, embeddings, chunk_ids):
        """Add document embeddings to FAISS index"""
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Train index if needed (for IVF indices)
        if isinstance(self.document_index, faiss.IndexIVFFlat) and not self.document_index.is_trained:
            if len(embeddings) >= 100:  # Need enough data to train
                self.document_index.train(embeddings)
        
        # Add embeddings
        start_idx = self.document_index.ntotal
        self.document_index.add(embeddings)
        
        # Update ID mapping
        for i, chunk_id in enumerate(chunk_ids):
            self.document_id_map.append(chunk_id)
        
        return start_idx
    
    def add_chat_messages(self, embeddings, message_ids):
        """Add chat message embeddings to FAISS index"""
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        
        start_idx = self.chat_index.ntotal
        self.chat_index.add(embeddings)
        
        # Update ID mapping
        for i, message_id in enumerate(message_ids):
            self.chat_id_map.append(message_id)
        
        return start_idx
    
    def search_documents(self, query_embedding, top_k=5):
        """Search document embeddings"""
        query_embedding = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.document_index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid result
                results.append({
                    'chunk_id': self.document_id_map[idx],
                    'score': float(score),
                    'faiss_index': int(idx)
                })
        
        return results
    
    def search_chat_messages(self, query_embedding, top_k=3):
        """Search chat message embeddings"""
        query_embedding = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.chat_index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid result
                results.append({
                    'message_id': self.chat_id_map[idx],
                    'score': float(score),
                    'faiss_index': int(idx)
                })
        
        return results
    
    def save_indices(self, base_path="/app/data/faiss"):
        """Save FAISS indices to disk"""
        os.makedirs(base_path, exist_ok=True)
        
        # Save FAISS indices
        faiss.write_index(self.document_index, f"{base_path}/document_index.faiss")
        faiss.write_index(self.chat_index, f"{base_path}/chat_index.faiss")
        
        # Save ID mappings
        with open(f"{base_path}/document_id_map.pkl", 'wb') as f:
            pickle.dump(self.document_id_map, f)
        
        with open(f"{base_path}/chat_id_map.pkl", 'wb') as f:
            pickle.dump(self.chat_id_map, f)
    
    def load_indices(self, base_path="/app/data/faiss"):
        """Load FAISS indices from disk"""
        try:
            # Load FAISS indices
            self.document_index = faiss.read_index(f"{base_path}/document_index.faiss")
            self.chat_index = faiss.read_index(f"{base_path}/chat_index.faiss")
            
            # Load ID mappings
            with open(f"{base_path}/document_id_map.pkl", 'rb') as f:
                self.document_id_map = pickle.load(f)
            
            with open(f"{base_path}/chat_id_map.pkl", 'rb') as f:
                self.chat_id_map = pickle.load(f)
            
            return True
        except Exception as e:
            print(f"Failed to load indices: {e}")
            return False
```

### 7.3 Sample Queries

```sql
-- Get document chunk metadata by chunk_id (after FAISS search)
SELECT 
    chunk_text,
    doc_id,
    page_number,
    content_type,
    source_url
FROM document_chunks 
WHERE chunk_id = ANY($1::text[]);

-- Get chat message metadata by message IDs (after FAISS search)
SELECT 
    message_text,
    created_at,
    message_type,
    session_id
FROM chat_messages 
WHERE id = ANY($1::integer[]);

-- Get crawl job statistics
SELECT 
    job_id,
    status,
    pages_crawled,
    pages_failed,
    chunks_created,
    (pages_crawled::float / GREATEST(max_pages, 1)) * 100 as completion_percentage
FROM crawl_jobs 
WHERE status IN ('running', 'completed')
ORDER BY created_at DESC;

-- Get content distribution by source
SELECT 
    content_type,
    COUNT(*) as chunk_count,
    COUNT(DISTINCT COALESCE(doc_id, domain)) as source_count
FROM document_chunks
GROUP BY content_type;

-- Get session activity summary
SELECT 
    session_type,
    COUNT(*) as session_count,
    COUNT(DISTINCT session_id) as unique_sessions,
    AVG(EXTRACT(EPOCH FROM (last_activity - created_at))/60) as avg_session_duration_minutes
FROM chat_sessions 
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY session_type;
```

## 8. Deployment Guide: Local Docker + FAISS + Ollama

### 8.1 Docker Compose Configuration

```yaml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: fizen_rag
      POSTGRES_USER: fizen_user
      POSTGRES_PASSWORD: fizen_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - fizen_rag_network

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    networks:
      - fizen_rag_network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    # For CPU-only deployment, remove the deploy section above

  vector_service:
    build:
      context: ./vector_service
      dockerfile: Dockerfile
    ports:
      - "8001:80"
    environment:
      - DATABASE_URL=postgresql://fizen_user:fizen_password@postgres:5432/fizen_rag
      - FAISS_DATA_PATH=/app/data/faiss
    volumes:
      - faiss_data:/app/data/faiss
      - uploaded_files:/app/data/files
    depends_on:
      - postgres
    networks:
      - fizen_rag_network

  api_gateway:
    build:
      context: ./api_gateway
      dockerfile: Dockerfile
    ports:
      - "8000:80"
    environment:
      - DATABASE_URL=postgresql://fizen_user:fizen_password@postgres:5432/fizen_rag
      - VECTOR_SERVICE_URL=http://vector_service:80
      - OLLAMA_URL=http://ollama:11434
    depends_on:
      - postgres
      - vector_service
      - ollama
    networks:
      - fizen_rag_network

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - api_gateway
    networks:
      - fizen_rag_network

volumes:
  postgres_data:
  ollama_data:
  faiss_data:
  uploaded_files:

networks:
  fizen_rag_network:
    driver: bridge
```

### 8.2 Vector Service with FAISS

#### Dockerfile:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    transformers \
    fastapi \
    uvicorn \
    sqlalchemy \
    psycopg2-binary \
    asyncpg \
    faiss-cpu \
    numpy \
    pandas

WORKDIR /app
COPY . /app

# Create data directories
RUN mkdir -p /app/data/faiss /app/data/files

CMD ["uvicorn", "vector_service:app", "--host", "0.0.0.0", "--port", "80"]
```

#### Code Example

```python
from fastapi import FastAPI, Request, HTTPException
from transformers import AutoTokenizer, AutoModel
import torch
import asyncio
import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from typing import List, Dict, Any
import numpy as np
import json
import os
from faiss_store import FAISSVectorStore

# BGE-Small model setup
model_name = "BAAI/bge-small-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://fizen_user:fizen_password@postgres:5432/fizen_rag")
engine = create_async_engine(DATABASE_URL, echo=True)

# FAISS Vector Store
FAISS_DATA_PATH = os.getenv("FAISS_DATA_PATH", "/app/data/faiss")
vector_store = FAISSVectorStore(dimension=384, index_type="IVFFlat")

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Load FAISS indices on startup"""
    if vector_store.load_indices(FAISS_DATA_PATH):
        print("FAISS indices loaded successfully")
    else:
        print("No existing FAISS indices found, starting fresh")

@app.on_event("shutdown")
async def shutdown_event():
    """Save FAISS indices on shutdown"""
    vector_store.save_indices(FAISS_DATA_PATH)
    print("FAISS indices saved")

async def embed_text(text: str) -> List[float]:
    """Generate embedding for a single text"""
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        output = model(**inputs)
    embedding = output.last_hidden_state.mean(dim=1).numpy().tolist()[0]
    return embedding

async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts"""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        output = model(**inputs)
    embeddings = output.last_hidden_state.mean(dim=1).numpy().tolist()
    return embeddings

@app.post("/embed")
async def embed_endpoint(req: Request):
    data = await req.json()
    texts = data.get("texts", [])
    embeddings = await embed_texts(texts)
    return {"embeddings": embeddings}

@app.post("/ingest")
async def ingest_document(req: Request):
    data = await req.json()
    doc_id = data.get("doc_id")
    filename = data.get("filename", "")
    file_type = data.get("file_type", "")
    chunks = data.get("chunks", [])
    
    try:
        async with engine.begin() as conn:
            # Insert document record
            await conn.execute(
                text("""
                    INSERT INTO documents (doc_id, filename, file_type, total_chunks, status)
                    VALUES (:doc_id, :filename, :file_type, :total_chunks, 'processing')
                    ON CONFLICT (doc_id) DO UPDATE SET
                        total_chunks = :total_chunks,
                        status = 'processing'
                """),
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "file_type": file_type,
                    "total_chunks": len(chunks)
                }
            )
            
            # Process chunks in batches
            batch_size = 10
            chunk_ids = []
            embeddings = []
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_texts = [chunk["text"] for chunk in batch_chunks]
                batch_embeddings = await embed_texts(batch_texts)
                
                # Insert chunks metadata into PostgreSQL
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                    chunk_id = f"{doc_id}_chunk_{i+j:03d}"
                    chunk_ids.append(chunk_id)
                    embeddings.append(embedding)
                    
                    # Store metadata in PostgreSQL (no embedding)
                    await conn.execute(
                        text("""
                            INSERT INTO document_chunks 
                            (chunk_id, doc_id, chunk_text, chunk_index, page_number, section_title, faiss_index)
                            VALUES (:chunk_id, :doc_id, :chunk_text, :chunk_index, :page_number, :section_title, :faiss_index)
                        """),
                        {
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "chunk_text": chunk["text"],
                            "chunk_index": i + j,
                            "page_number": chunk.get("page", 0),
                            "section_title": chunk.get("section", ""),
                            "faiss_index": vector_store.document_index.ntotal + len(chunk_ids) - 1
                        }
                    )
            
            # Add embeddings to FAISS
            vector_store.add_documents(embeddings, chunk_ids)
            
            # Update document status
            await conn.execute(
                text("UPDATE documents SET status = 'completed' WHERE doc_id = :doc_id"),
                {"doc_id": doc_id}
            )
        
        # Save FAISS indices
        vector_store.save_indices(FAISS_DATA_PATH)
        
        return {
            "status": "success", 
            "chunks_stored": len(chunks),
            "doc_id": doc_id,
            "faiss_index_updated": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/retrieve")
async def retrieve_chunks(req: Request):
    data = await req.json()
    query = data.get("query")
    top_k = data.get("top_k", 5)
    similarity_threshold = data.get("similarity_threshold", 0.75)
    
    try:
        # Generate query embedding
        query_embedding = await embed_text(query)
        
        # Search FAISS index
        faiss_results = vector_store.search_documents(query_embedding, top_k)
        
        # Filter by similarity threshold
        filtered_results = [r for r in faiss_results if r['score'] >= similarity_threshold]
        
        if not filtered_results:
            return []
        
        # Get chunk metadata from PostgreSQL
        chunk_ids = [r['chunk_id'] for r in filtered_results]
        
        async with engine.begin() as conn:
            result = await conn.execute(
                text("""
                    SELECT 
                        chunk_id,
                        chunk_text,
                        doc_id,
                        page_number,
                        section_title,
                        content_type,
                        source_url
                    FROM document_chunks 
                    WHERE chunk_id = ANY(:chunk_ids)
                """),
                {"chunk_ids": chunk_ids}
            )
            
            # Create mapping of chunk_id to metadata
            metadata_map = {}
            for row in result:
                metadata_map[row.chunk_id] = {
                    "text": row.chunk_text,
                    "source": row.doc_id or row.source_url,
                    "page": row.page_number,
                    "section": row.section_title,
                    "content_type": row.content_type
                }
            
            # Combine FAISS results with metadata
            chunks = []
            for faiss_result in filtered_results:
                chunk_id = faiss_result['chunk_id']
                if chunk_id in metadata_map:
                    metadata = metadata_map[chunk_id]
                    chunks.append({
                        **metadata,
                        "chunk_id": chunk_id,
                        "score": faiss_result['score']
                    })
            
            return chunks
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

@app.post("/store_chat_context")
async def store_chat_context(req: Request):
    data = await req.json()
    session_id = data.get("session_id")
    user_id = data.get("user_id")
    message_text = data.get("message_text")
    message_type = data.get("message_type", "user")
    
    try:
        # Generate embedding for the message
        message_embedding = await embed_text(message_text)
        
        async with engine.begin() as conn:
            # Ensure session exists
            await conn.execute(
                text("""
                    INSERT INTO chat_sessions (session_id, user_id, last_activity)
                    VALUES (:session_id, :user_id, CURRENT_TIMESTAMP)
                    ON CONFLICT (session_id) DO UPDATE SET
                        last_activity = CURRENT_TIMESTAMP
                """),
                {"session_id": session_id, "user_id": user_id}
            )
            
            # Store message metadata in PostgreSQL
            result = await conn.execute(
                text("""
                    INSERT INTO chat_messages (session_id, message_text, message_type, faiss_index)
                    VALUES (:session_id, :message_text, :message_type, :faiss_index)
                    RETURNING id
                """),
                {
                    "session_id": session_id,
                    "message_text": message_text,
                    "message_type": message_type,
                    "faiss_index": vector_store.chat_index.ntotal
                }
            )
            
            message_id = result.fetchone()[0]
        
        # Add embedding to FAISS chat index
        vector_store.add_chat_messages([message_embedding], [message_id])
        
        # Save FAISS indices
        vector_store.save_indices(FAISS_DATA_PATH)
        
        return {"status": "success", "message": "Chat context stored"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context storage failed: {str(e)}")

@app.post("/get_chat_context")
async def get_chat_context(req: Request):
    """Retrieve relevant chat history via FAISS similarity search"""
    data = await req.json()
    session_id = data.get("session_id")
    query = data.get("query")
    top_k = data.get("top_k", 3)
    
    try:
        # Generate query embedding
        query_embedding = await embed_text(query)
        
        # Search FAISS chat index
        faiss_results = vector_store.search_chat_messages(query_embedding, top_k)
        
        if not faiss_results:
            return {"context": []}
        
        # Get message metadata from PostgreSQL
        message_ids = [r['message_id'] for r in faiss_results]
        
        async with engine.begin() as conn:
            result = await conn.execute(
                text("""
                    SELECT 
                        id,
                        message_text,
                        created_at,
                        message_type
                    FROM chat_messages 
                    WHERE id = ANY(:message_ids)
                        AND session_id = :session_id 
                        AND message_type = 'user'
                """),
                {
                    "message_ids": message_ids,
                    "session_id": session_id
                }
            )
            
            # Create mapping of message_id to metadata
            metadata_map = {}
            for row in result:
                metadata_map[row.id] = {
                    "message": row.message_text,
                    "timestamp": row.created_at.isoformat(),
                    "message_type": row.message_type
                }
            
            # Combine FAISS results with metadata
            context_messages = []
            for faiss_result in faiss_results:
                message_id = faiss_result['message_id']
                if message_id in metadata_map and faiss_result['score'] > 0.7:
                    metadata = metadata_map[message_id]
                    context_messages.append({
                        **metadata,
                        "score": faiss_result['score']
                    })
            
            return {"context": context_messages}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context retrieval failed: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        
        faiss_status = {
            "document_index_size": vector_store.document_index.ntotal,
            "chat_index_size": vector_store.chat_index.ntotal
        }
        
        return {
            "status": "healthy", 
            "database": "connected",
            "faiss": faiss_status
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### 8.3 LLM Service with Ollama

#### Setup and Code Example:

```python
import requests
import json
from typing import Dict, Any

class OllamaClient:
    def __init__(self, base_url: str = "http://ollama:11434"):
        self.base_url = base_url
        self.default_model = "llama2"  # or "mistral", "codellama", etc.
    
    def generate(self, prompt: str, model: str = None, **kwargs) -> str:
        """Generate response using Ollama"""
        model = model or self.default_model
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama request failed: {str(e)}")
    
    def chat(self, messages: list, model: str = None, **kwargs) -> str:
        """Chat using Ollama (for models that support chat format)"""
        model = model or self.default_model
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama chat request failed: {str(e)}")
    
    def list_models(self) -> list:
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            result = response.json()
            return [model["name"] for model in result.get("models", [])]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to list models: {str(e)}")

# Usage in LLM Orchestrator
def query_llm(prompt: str, model: str = "llama2") -> str:
    ollama = OllamaClient()
    return ollama.generate(prompt, model=model)
```

### 8.4 Ollama Model Setup

#### Initial Setup Script:

```bash
#!/bin/bash
# setup_ollama.sh

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
sleep 10

# Pull required models
echo "Pulling Llama2 model..."
docker exec fizen-rag-ollama-1 ollama pull llama2

echo "Pulling Mistral model..."
docker exec fizen-rag-ollama-1 ollama pull mistral

echo "Pulling Code Llama for SQL generation..."
docker exec fizen-rag-ollama-1 ollama pull codellama

echo "Models ready!"
```

### 8.5 Deployment Architecture Summary

```
          +-------------+          +--------------------+
          |  React UI   +---------> FastAPI Backend     |
          | (Port 3000) |          | (Port 8000)        |
          +-------------+          | - Retrieval Logic  |
                                   | - Prompt Builder   |
                                   +--------------------+
                                           |
                 +-------------------------+--------------------------+
                 |                                                    |
      +------------------------+                          +----------------------+
      | Vector Storage &       | <---- Bidirectional ----> | PostgreSQL           |
      | Retrieval Service      |       Data Flow           | (Port 5432)          |
      | (Port 8001)            |                          | - Metadata Storage   |
      | - BGE-Small Embedding  |                          | - Chat Context       |
      | - Document Ingestion   |                          | - Job Management     |
      | - FAISS Search         |                          +----------------------+
      | - Context Management   |
      +------------------------+
                 |
                 v
      +------------------------+           +----------------------+
      | FAISS Vector Indices   |           | Ollama LLM Service   |
      | - Document Embeddings  |           | (Port 11434)         |
      | - Chat Embeddings      |           | - Llama2/Mistral     |
      | - Persistent Storage   |           | - Local Inference    |
      +------------------------+           +----------------------+
```

## 9. Performance Optimization

### 9.1 FAISS Index Optimization

```python
# Production FAISS configuration for better performance
class OptimizedFAISSStore(FAISSVectorStore):
    def __init__(self, dimension=384):
        super().__init__(dimension, index_type="HNSW")
        
    def _init_document_index(self):
        """Initialize optimized FAISS index for documents"""
        # For large datasets (>100k vectors): IndexIVFPQ with quantization
        if os.getenv("FAISS_LARGE_DATASET", "false").lower() == "true":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.document_index = faiss.IndexIVFPQ(
                quantizer, self.dimension, 
                nlist=100,  # Number of clusters
                m=8,        # Number of subquantizers
                nbits=8     # Bits per subquantizer
            )
        else:
            # For medium datasets: IndexHNSWFlat for high accuracy
            self.document_index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.document_index.hnsw.efConstruction = 64
            self.document_index.hnsw.efSearch = 32
```

### 9.2 Docker Performance Tuning

```yaml
# Optimized docker-compose.yml for performance
version: '3.8'
services:
  vector_service:
    # ... other config
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'
    environment:
      - FAISS_LARGE_DATASET=false
      - OMP_NUM_THREADS=4
      - MKL_NUM_THREADS=4
  
  ollama:
    # ... other config
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'
  
  postgres:
    # ... other config
    command: |
      postgres 
      -c shared_buffers=256MB 
      -c max_connections=100
      -c effective_cache_size=1GB
      -c maintenance_work_mem=64MB
      -c checkpoint_completion_target=0.9
      -c wal_buffers=16MB
      -c default_statistics_target=100
```

### 9.3 Connection Pooling and Caching

```python
# Optimized async engine configuration
engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=30,
    pool_timeout=30,
    pool_recycle=3600,
    echo=False
)

# Add caching for embeddings
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embed_text(text: str) -> tuple:
    """Cache embeddings for frequently used queries"""
    embedding = embed_text(text)
    return tuple(embedding)  # Convert to tuple for hashing
```