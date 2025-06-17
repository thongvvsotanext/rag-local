# Fizen RAG Deep Technical Specification - Upgraded (FAISS + Ollama + SearxNG)

## Objective

Fizen RAG (Retrieval-Augmented Generation) is an enterprise-ready architecture that enables intelligent question-answering from both static and live data sources. It combines LLM capabilities with structured retrieval, allowing businesses to:

- Extract precise, explainable answers from large knowledge bases
- Integrate document, API, SQL, and live web information
- Maintain context-rich, conversational interfaces for users

This specification is intended for developers building the system and testers verifying correctness and performance.

## Technology Stack

### Frontend

- **React** – Interactive UI for users and admins
- **Tailwind CSS** – Modern, responsive styling
- **Axios** – API communication handler

### Backend

- **FastAPI** – High-performance Python backend API
- **aiohttp** – Async HTTP client for live web crawling
- **SQLAlchemy** – ORM for database access
- **Pandas/Numpy** – Data transformation & tabular logic

### AI/ML

- **BGE-Small (via Hugging Face)** – Lightweight sentence embedding model
- **Ollama** – Local LLM inference (Llama2, Mistral, etc.)
- **Optional: LangChain** – Prompt orchestration and chaining

### Storage & Indexing

- **FAISS** – High-performance vector similarity search library
- **PostgreSQL** – Database for structured data and metadata
- **Local File System** – Storage for uploaded files and FAISS indices

### DevOps

- **Docker & Docker Compose** – Local containerization and orchestration
- **Prometheus + Grafana** – Monitoring and alerting

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

## 4. Component Overview with Input, Output, Internal Logic, Integration Endpoints & Examples

### 4.1 Context Manager

**Purpose**:
To maintain natural conversation flow and enrich queries with relevant previous interactions for both authenticated admin users and anonymous public users. This ensures continuity (e.g., follow-up questions) and improves retrieval accuracy.

**Internal Logic:**

1. Query latest N messages from the session (PostgreSQL) - supports both user-based and anonymous sessions.
2. Extract relevant messages using Vector Storage & Retrieval Service to query embedded chat history.
3. Concatenate top matches into a summarized history block.
4. Return context+query for retrieval.

**Integration Endpoints (Calls to Other Services)**:

- **Vector Service**: `POST http://vector_service:8001/chat/context` - Performs similarity search against chat message embeddings
- **PostgreSQL**: Direct database connection for session management

**Exposed Endpoints**:

- `POST /context/enhance` - Enhances query with relevant conversation context
- `GET /context/session/{session_id}` - Retrieves session context summary

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

1. Calls Vector Storage & Retrieval Service with query embedding: `POST /search/vector`
2. Calls SQL Retriever with natural query and schema metadata: `POST /sql/query`
3. Calls Web Retriever using predefined trusted domain filters: `POST /search/realtime`
4. Deduplicates results using fingerprint hash or sentence similarity.
5. Sorts based on semantic relevance and recency.
6. Returns unified evidence pool for prompt building.

**Integration Endpoints (Calls to Other Services)**:

- **Vector Service**: `POST http://vector_service:8001/search/vector` - Performs similarity search
- **SQL Retriever**: `POST http://sql_retriever:8004/sql/query` - Converts natural language to SQL
- **Web Retriever**: `POST http://web_retriever:8005/search/realtime` - Gets current web information

**Exposed Endpoints**:

- `POST /orchestrate/retrieve` - Main retrieval orchestration endpoint
- `GET /orchestrate/sources` - Lists available data sources and their status

**How it works**:

1. Forwards enhanced query to:
   - Vector Storage & Retrieval Service (docs): `POST http://vector_service:8001/search/vector`
   - SQL Retriever (databases): `POST http://sql_retriever:8004/sql/query`
   - Web Retriever (external trusted info): `POST http://web_retriever:8005/search/realtime`
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

**Integration Endpoints (Calls to Other Services)**:

- **PostgreSQL**: Direct database connection for metadata storage and retrieval
- **BGE-Small Model**: Local embedding model inference

**Exposed Endpoints**:

- `POST /search/vector` - Document/web content similarity search
- `POST /chat/context` - Chat history similarity search
- `POST /ingest/document` - Document ingestion and processing
- `POST /ingest/web` - Web content ingestion and processing
- `POST /embed` - Text embedding generation
- `GET /health` - Service health check with FAISS statistics
- `GET /memory` - Memory usage and FAISS index statistics
- `GET /stats` - Detailed service statistics and performance metrics

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

**Integration Endpoints (Calls to Other Services)**:

- **Ollama Service**: `POST http://ollama:11434/api/generate` - LLM-powered SQL generation
- **PostgreSQL**: Direct database connection for query execution

**Exposed Endpoints**:

- `POST /sql/query` - Natural language to SQL conversion and execution
- `GET /sql/schema` - Database schema information
- `POST /sql/validate` - SQL query validation
- `GET /sql/health` - Service health check

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

1. Query SearxNG service for current information related to user queries
2. Filter results against trusted domain whitelist
3. Extract and summarize content for immediate use in responses
4. Return fresh data to supplement knowledge base content

**Integration Endpoints (Calls to Other Services)**:

- **SearxNG Service**: `GET http://searxng:8080/search` - Privacy-focused metasearch
- **Vector Service**: `POST http://vector_service:8001/ingest/web` - Store processed web content
- **Ollama Service**: `POST http://ollama:11434/api/generate` - Content summarization

**Exposed Endpoints**:

- `POST /search/realtime` - Real-time web search using SearxNG
- `POST /crawl/start` - Start automated web crawling job
- `GET /crawl/status/{job_id}` - Check crawl job status
- `GET /crawl/jobs` - List all crawl jobs
- `POST /crawl/stop/{job_id}` - Stop active crawl job
- `GET /search/health` - Service health check including SearxNG status

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

1. Call SearxNG service with user query for current information
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

**Integration Endpoints (Calls to Other Services)**:

- None (internal processing only)

**Exposed Endpoints**:

- `POST /prompt/build` - Build optimized prompt from query and evidence
- `POST /prompt/template` - Apply custom prompt templates
- `GET /prompt/health` - Service health check

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

**Integration Endpoints (Calls to Other Services)**:

- **Ollama Service**: `POST http://ollama:11434/api/generate` - Main LLM inference
- **Ollama Service**: `GET http://ollama:11434/api/tags` - Available models
- **Ollama Service**: `POST http://ollama:11434/api/pull` - Model management

**Exposed Endpoints**:

- `POST /llm/generate` - Generate response from prompt
- `POST /llm/chat` - Chat completion with conversation context
- `GET /llm/models` - List available models
- `POST /llm/model/switch` - Switch active model
- `GET /llm/health` - Service health and model status

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

**Integration Endpoints (Calls to Other Services)**:

- None (internal processing only)

**Exposed Endpoints**:

- `POST /format/response` - Format LLM response with citations
- `POST /format/markdown` - Convert response to Markdown
- `POST /format/json` - Structure response as JSON
- `GET /format/health` - Service health check

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
  "citations": [
    { "ref": "[1]", "doc": "sql", "value": "return stats Jan 2024" }
  ]
}
```

## Updated Integration Endpoint Summary Table

| Container                       | Internal Components                                                                           | Exposes Endpoints                                                                                                    | Calls External Services                                                                                                        |
| ------------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **📦 API Gateway Container**    | Context Manager, Retrieval Orchestrator, Prompt Builder, LLM Orchestrator, Response Formatter | `/chat`, `/admin/*`, `/auth/*`, `/sessions/*`                                                                        | Vector Service (`/search/vector`, `/chat/context`), Web Retriever (`/search/realtime`, `/sql/query`), Ollama (`/api/generate`) |
| **📦 Vector Service Container** | Vector Storage & Retrieval Service                                                            | `/search/vector`, `/chat/context`, `/ingest/document`, `/ingest/web`, `/embed`, `/health`, `/memory`, `/stats`       | PostgreSQL (direct), BGE-Small (local)                                                                                         |
| **📦 Web Retriever Container**  | Web Retriever, SQL Retriever                                                                  | `/search/realtime`, `/sql/query`, `/crawl/start`, `/crawl/status/{id}`, `/crawl/jobs`, `/crawl/stop/{id}`, `/health` | SearxNG (`/search`), Vector Service (`/ingest/web`), Ollama (`/api/generate`), PostgreSQL (direct)                             |
| **📦 PostgreSQL Container**     | Database Engine                                                                               | Standard PostgreSQL endpoints                                                                                        | None (base service)                                                                                                            |
| **📦 SearxNG Container**        | Metasearch Engine                                                                             | `/search`, `/healthz`, `/config`                                                                                     | External search engines                                                                                                        |
| **📦 Ollama Container**         | LLM Inference Engine                                                                          | `/api/generate`, `/api/tags`, `/api/pull`                                                                            | None (base service)                                                                                                            |
| **📦 Frontend Container**       | React Application                                                                             | Static file serving                                                                                                  | API Gateway (all endpoints)                                                                                                    |

## Updated Service-to-Service Communication Flow

```
Query Flow (8-Container Architecture):
📦 Frontend Container → 📦 API Gateway Container
                          (Context Manager Module)
                                    ↓
                          📦 Vector Service Container
                                    ↓
                          Chat History FAISS Search
                                    ↓
                    Enhanced Query → 📦 API Gateway Container
                                    ↓
                          📦 Retrieval Orchestrator Container
                                           ↓
                            ┌──────────────┼──────────────┐
                            ↓              ↓              ↓
                  📦 Vector Service   📦 Web Retriever   📦 PostgreSQL Container
                     Container         Container        (Direct SQL)
                          ↓              ↓              ↓
                    FAISS Index    SearxNG + SQL    PostgreSQL
                          ↓              ↓              ↓
                          └──────────────┼──────────────┘
                                         ↓
                              📦 Retrieval Orchestrator Container
                                    (Result Unification)
                                         ↓
                              📦 API Gateway Container
                              (Prompt Builder Module)
                                         ↓
                              📦 API Gateway Container
                              (LLM Orchestrator Module) → 📦 Ollama Container
                                         ↓
                              📦 API Gateway Container
                              (Response Formatter Module)
                                         ↓
                                 Final Answer
```

## 5. Complete Data Ingestion Flows with Integration Endpoints

### 5.1 Document Processing Flow

**Purpose**: Convert uploaded documents into searchable vector representations stored in FAISS.

**Integration Endpoints (Calls to Other Services)**:

- **Vector Service**: `POST http://vector_service:8001/ingest/document` - Document ingestion and processing
- **PostgreSQL**: Direct database connection for metadata storage and job tracking
- **File System**: Local file storage for uploaded documents

**Exposed Endpoints**:

- `POST /admin/documents/upload` - Document upload endpoint (API Gateway)
- `GET /admin/documents/status/{doc_id}` - Check processing status (API Gateway)
- `GET /admin/documents/list` - List all processed documents (API Gateway)
- `DELETE /admin/documents/delete/{doc_id}` - Remove document and chunks (API Gateway)

**Flow Steps**:

1. **Document Upload**

   - Admin uploads document via React dashboard to API Gateway
   - **API Gateway** validates file type, size, and security constraints
   - File stored in **shared volume** mounted by both API Gateway and Vector Service containers
   - Upload job queued with unique job_id in PostgreSQL
   - **Endpoint**: `POST /admin/documents/upload`

2. **Document Processing Initiation**

   - **API Gateway** calls **Vector Service**: `POST http://vector_service:8001/ingest/document`
   - **Vector Service** receives processing request with:
     - **Shared file path**: `/app/data/files/{doc_id}/{filename}` (accessible to both containers)
     - **File metadata**: original filename, content type, upload timestamp
     - **Processing options**: chunk size, overlap, language detection
   - **Vector Service** reads file from shared volume and extracts raw text using appropriate parser:
     - PDF: PDFMiner or PyPDF2
     - DOCX: python-docx
     - TXT: Direct text extraction
   - Text is cleaned and normalized (remove headers, footers, extra whitespace)

3. **Chunking & Embedding**

   - **Vector Service** applies recursive character splitting (500 tokens, 20 overlap)
   - Each chunk processed with BGE-small to generate 384-dim embeddings
   - Chunks enriched with metadata:
     - Document source, page numbers
     - Section titles and headers
     - Processing timestamp

4. **Storage in FAISS**

   - **Vector Service** stores document record in `documents` table (PostgreSQL)
   - Each chunk metadata stored in `document_chunks` table (PostgreSQL)
   - Embeddings added to FAISS index with unique chunk IDs
   - FAISS index persisted to disk for durability

5. **Completion & Notification**
   - **Vector Service** updates document status to 'completed' in PostgreSQL
   - **API Gateway** receives completion notification
   - Admin dashboard receives real-time update via WebSocket
   - Vector search immediately available for new content

**Service Communication Flow**:

```
Frontend → API Gateway → Vector Service → PostgreSQL
                      ↓              ↓
                 File System    FAISS Index
```

**Input Example**:

```json
{
  "shared_file_path": "/app/data/files/doc_001/company_policies.pdf",
  "metadata": {
    "original_filename": "company_policies.pdf",
    "content_type": "application/pdf",
    "department": "HR",
    "version": "2.1",
    "classification": "internal",
    "upload_timestamp": "2024-01-08T14:30:00Z",
    "file_size_bytes": 2048576
  },
  "processing_options": {
    "chunk_size": 500,
    "chunk_overlap": 20,
    "language": "en"
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

**Integration Endpoints (Calls to Other Services)**:

- **Web Retriever**: `POST http://web_retriever:8005/crawl/start` - Start web crawling job
- **Vector Service**: `POST http://vector_service:8001/ingest/web` - Store processed web content
- **PostgreSQL**: Direct database connection for crawl job tracking and web page metadata
- **SearxNG**: `GET http://searxng:8080/search` - Real-time web search capabilities

**Exposed Endpoints**:

- `POST /admin/crawl/start` - Start automated web crawling job (API Gateway)
- `GET /admin/crawl/status/{job_id}` - Check crawl job status (API Gateway)
- `GET /admin/crawl/jobs` - List all crawl jobs (API Gateway)
- `POST /admin/crawl/stop/{job_id}` - Stop active crawl job (API Gateway)
- `GET /admin/crawl/statistics` - Get crawl statistics and performance metrics (API Gateway)

**Flow Steps**:

1. **Crawl Configuration**

   - Admin configures crawl job via dashboard
   - **API Gateway** validates parameters and checks rate limits
   - **API Gateway** calls **Web Retriever**: `POST http://web_retriever:8005/crawl/start`
   - Crawl job created in PostgreSQL with configuration:
     - Target URLs or domains
     - Max pages and crawl depth limits
     - Content filters and exclusion patterns
     - Crawl frequency (one-time or scheduled)

2. **Content Retrieval**

   - **Web Retriever** initiates crawl using aiohttp
   - For each URL in crawl queue:
     - Fetch HTML content with proper user-agent headers
     - Respect robots.txt and implement rate limiting
     - Extract main text using readability-lxml
     - Filter out navigation, ads, and boilerplate content
   - Content validated against domain whitelist
   - Progress tracked in PostgreSQL `crawl_jobs` table

3. **Content Processing**

   - **Web Retriever** sends extracted text to **Vector Service**: `POST http://vector_service:8001/ingest/web`
   - **Vector Service** applies same processing pipeline as document uploads:
     - Text cleaning and normalization
     - Semantic chunking with overlap
     - BGE-small embedding generation
   - Metadata enriched with web-specific information:
     - URL, crawl timestamp, page title
     - Domain classification and content type

4. **Deduplication & Storage**

   - **Vector Service** checks for existing content using text similarity
   - New unique chunks stored in FAISS index
   - Web-specific metadata stored in PostgreSQL:
     - `web_pages` table with URL and domain info
     - `document_chunks` table with content_type = 'web'
     - Source URL and domain tracking
     - Last crawled timestamp and content freshness indicators

5. **Monitoring & Updates**
   - **Web Retriever** tracks crawl progress in real-time
   - Failed URLs logged with detailed error reasons in PostgreSQL
   - Successful pages immediately available for search via FAISS
   - **API Gateway** aggregates statistics for admin dashboard
   - Crawl completion triggers next scheduled crawl if configured

**Service Communication Flow**:

```
Frontend → API Gateway → Web Retriever → Vector Service → PostgreSQL
                                    ↓              ↓
                              External Websites   FAISS Index
```

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

**Integration Endpoints (Calls to Other Services)**:

- **Vector Service**: `POST http://vector_service:8001/chat/context` - Retrieve relevant chat history
- **Retrieval Orchestrator**: `POST http://retrieval_orchestrator:8003/orchestrate/retrieve` - Multi-source data retrieval coordination
- **Ollama**: `POST http://ollama:11434/api/generate` - LLM inference for response generation
- **PostgreSQL**: Direct database connection for session management and result storage

**Exposed Endpoints**:

- `POST /chat` - Main chat endpoint for public users (API Gateway)
- `WebSocket /chat/stream` - Streaming chat responses (API Gateway)
- `GET /sessions/history/{session_id}` - Retrieve session chat history (API Gateway)
- `POST /query/search` - Advanced search with filters (API Gateway)
- `GET /query/sources` - List available content sources (API Gateway)

**Flow Steps**:

1. **Query Reception**

   - User submits query via chat interface to **📦 API Gateway Container**
   - **API Gateway** validates request and applies rate limiting
   - Query includes user context and optional filters
   - Session created or retrieved from PostgreSQL
   - **Endpoint**: `POST /chat` or `WebSocket /chat/stream`

2. **Context Integration (Internal to API Gateway)**

   - **API Gateway** (Context Manager module) calls **📦 Vector Service**: `POST http://vector_service:8001/chat/context`
   - **📦 Vector Service** embeds current query and performs FAISS similarity search against chat message embeddings
   - **📦 Vector Service** returns relevant previous messages
   - **API Gateway** (Context Manager module) formats them into context block
   - Enhanced query with conversation context ready for retrieval

3. **Multi-Source Retrieval Coordination**

   - **📦 API Gateway** calls **📦 Retrieval Orchestrator**: `POST http://retrieval_orchestrator:8003/orchestrate/retrieve`
   - **📦 Retrieval Orchestrator** coordinates multiple parallel searches:
     - **Vector Search**: `POST http://vector_service:8001/search/vector` - FAISS similarity search across all document chunks
     - **SQL Search**: `POST http://web_retriever:8005/sql/query` - Structured data queries for specific metrics
     - **Web Search**: `POST http://web_retriever:8005/search/realtime` - Real-time search for current information
   - Each service returns ranked results with confidence scores to **📦 Retrieval Orchestrator**

4. **Result Unification (Within Retrieval Orchestrator Service)**

   - **📦 Retrieval Orchestrator** processes all results internally:
     - Deduplicates using content similarity (>0.9 threshold)
     - Applies multi-factor ranking:
       - Original relevance scores
       - Content recency (web content weighted higher)
       - Source authority (internal docs > web content)
       - User context relevance
     - Combines top-k results from each source
   - **📦 Retrieval Orchestrator** returns unified evidence set to **📦 API Gateway**

5. **Response Generation (Internal to API Gateway)**
   - **API Gateway** (Prompt Builder module) formats unified results with source citations
   - **API Gateway** (LLM Orchestrator module) calls **📦 Ollama**: `POST http://ollama:11434/api/generate`
   - **API Gateway** (Response Formatter module) processes LLM output, adds proper citations and metadata
   - Final answer includes evidence from multiple sources
   - Response stored in PostgreSQL chat_messages table with source tracking

**Service Communication Flow**:

```
📦 Frontend Container → 📦 API Gateway Container (Context Manager)
                                    ↓
                          📦 Vector Service Container
                            (Chat History Search)
                                    ↓
                       📦 API Gateway Container (Enhanced Query)
                                    ↓
                    📦 Retrieval Orchestrator Container
                                    ↓
          ┌─────────────────────────┼─────────────────────────┐
          ↓                         ↓                         ↓
📦 Vector Service Container  📦 Web Retriever Container   📦 PostgreSQL Container
   (FAISS Document Search)    (SQL + Real-time Web)        (Direct Queries)
          ↓                         ↓                         ↓
          └─────────────────────────┼─────────────────────────┘
                                    ↓
                    📦 Retrieval Orchestrator Container
                           (Result Unification)
                                    ↓
                    📦 API Gateway Container (Prompt Builder)
                                    ↓
                    📦 API Gateway Container (LLM Orchestrator)
                                    ↓
                         📦 Ollama Container
                                    ↓
                    📦 API Gateway Container (Response Formatter)
                                    ↓
                              Final Answer
```

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

## Data Ingestion Flow Summary Table

| Flow Type               | Primary Services                                                           | Integration Endpoints                                                                                                                          | Exposed Endpoints                  |
| ----------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **Document Processing** | API Gateway → Vector Service → PostgreSQL                                  | Vector Service (`/ingest/document`)                                                                                                            | API Gateway (`/admin/documents/*`) |
| **Web Crawling**        | API Gateway → Web Retriever → Vector Service → PostgreSQL                  | Web Retriever (`/crawl/start`), Vector Service (`/ingest/web`)                                                                                 | API Gateway (`/admin/crawl/*`)     |
| **Query Processing**    | API Gateway → Context Manager → Retrieval Orchestrator → Multiple Services | Vector Service (`/search/vector`, `/chat/context`), SQL Retriever (`/sql/query`), Web Retriever (`/search/realtime`), Ollama (`/api/generate`) | API Gateway (`/chat`, `/query/*`)  |

## Complete Service Integration Map

```
Data Ingestion & Query Processing Architecture:

┌─────────────────────────────────────────────────────────────┐
│                    📦 Frontend Container                    │
│                    (React Admin + Chat)                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 📦 API Gateway Container                    │
│  • /admin/documents/*  • /admin/crawl/*  • /chat          │
│  • /query/*           • /sessions/*      • /auth/*        │
└─┬─────────────┬─────────────┬─────────────┬─────────────┬──┘
  │             │             │             │             │
  ▼             ▼             ▼             ▼             ▼
┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐
│📦 Vector│ │📦 Context│ │📦 Web    │ │📦 SQL    │ │📦 Ollama│
│ Service │ │ Manager  │ │Retriever │ │Retriever │ │Container│
│         │ │          │ │          │ │          │ │         │
│• ingest │ │• enhance │ │• crawl   │ │• query   │ │• generate│
│• search │ │• context │ │• realtime│ │• schema  │ │• models │
│• embed  │ │• session │ │• health  │ │• validate│ │• health │
└─────┬───┘ └─────┬────┘ └─────┬────┘ └─────┬────┘ └─────────┘
      │           │            │            │
      ▼           ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────┐
│              📦 PostgreSQL Container                        │
│  • documents  • crawl_jobs  • chat_sessions               │
│  • document_chunks  • web_pages  • chat_messages          │
└─────────────────────────────────────────────────────────────┘
      │                        │
      ▼                        ▼
┌─────────────┐          ┌──────────────┐
│📦 FAISS     │          │📦 SearxNG    │
│ Indices     │          │ Container    │
│ (Vector     │          │ (External    │
│  Storage)   │          │  Search)     │
└─────────────┘          └──────────────┘
```

This comprehensive update ensures that every data ingestion flow clearly documents its service interactions, endpoint mappings, and integration patterns, making the system architecture completely transparent and easy to implement.

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

+-------+-------+ +---------+---------+ +-------+-------+
|Public Chat | |Admin Content Mgmt | |Admin Web Mgmt |
|Pipeline | |Pipeline | |Pipeline |
|(No Auth) | |(Admin Only) | |(Admin Only) |
+-------+-------+ +---------+---------+ +-------+-------+
| | |
v v v
+-------+-------+ +---------+---------+ +-------+-------+
|Context Manager| | File Upload | | Web Crawler |
|(Anonymous OK) | | (Admin Only) | | (Admin Only) |
+-------+-------+ +---------+---------+ +-------+-------+
| | |
v | |
| +-------+-------+ |
| |Vector Storage | |
| |& Retrieval | |
| |Service | |
| |(Chat Context) | |
| +-------+-------+ |
| | |
v v v
+-------+-------+ +---------+---------+ +-------+-------+
|Retrieval Orch.| |Vector Storage & | |Vector Storage |
|(Public Access)|<-->|Retrieval Service |<-->|& Retrieval |
|(w/ Context) | |(Ingestion Mode) | |(Web Mode) |
+-------+-------+ |(Admin Only) | |(Admin Only) |
| +-------------------+ +---------------+
| | |
| v |
| +---------+---------+ |
| | PostgreSQL | |
| | Database | |
| | (Metadata + | |
| | Chat History) | |
| +---------+---------+ |
| | |
| v |
| +---------+---------+ |
| | FAISS Vector | |
| | Index | |
| | (All Embeddings) | |
| +---------+---------+ |
| | |
+----------------------+----------------------+
|
v
+------------+-------------+
| Prompt Builder |
| (Public Access) |
+------------+-------------+
|
v
+------------+-------------+
| LLM Orchestrator |
| (Ollama) |
| (Public Access) |
+------------+-------------+
|
v
+------------+-------------+
| Response Formatter |
| (Public Access) |
+------------+-------------+
|
v
+------------+-------------+
| Final Answer |
| (Available to Public) |
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
| Vector Storage & |
| Retrieval Service |
| (Document Ingestion Mode) |
+---------------+---------------+
|
+---------------+---------------+
| Text Extraction |
| - PDF: PDFMiner |
| - DOCX: python-docx |
| - TXT: Direct read |
+---------------+---------------+
|
v
+---------------+---------------+
| Text Processing |
| - Normalize whitespace |
| - Remove headers/footers |
| - Extract metadata |
+---------------+---------------+
|
v
+---------------+---------------+
| Semantic Chunking |
| - 500 token chunks |
| - 20 token overlap |
| - Preserve context |
+---------------+---------------+
|
v
+---------------+---------------+
| BGE-Small Embedding |
| - 384-dimensional vectors |
| - Batch processing |
+---------------+---------------+
|
v
+---------------+---------------+
| FAISS Index Storage |
| - Add embeddings to index |
| - PostgreSQL metadata |
| - Persist index to disk |
+---------------+---------------+
|
v
+---------------+---------------+
| Status Update & Notification |
| - Admin dashboard update |
| - Available for search |
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
| Web Retriever |
| (Content Fetching) |
+---------------+---------------+
|
+---------------+---------------+
| URL Processing |
| - Respect robots.txt |
| - Rate limiting |
| - User-agent headers |
+---------------+---------------+
|
v
+---------------+---------------+
| Content Extraction |
| - HTML parsing |
| - readability-lxml |
| - Remove boilerplate |
+---------------+---------------+
|
v
+---------------+---------------+
| Content Validation |
| - Domain whitelist check |
| - Content type filter |
| - Duplicate detection |
+---------------+---------------+
|
v
+---------------+---------------+
| Vector Storage & |
| Retrieval Service |
| (Web Content Mode) |
+---------------+---------------+
|
v
+---------------+---------------+
| Same Processing Pipeline |
| - Text normalization |
| - Semantic chunking |
| - BGE-Small embedding |
| - FAISS index storage |
+---------------+---------------+
|
v
+---------------+---------------+
| Crawl Completion |
| - Statistics logging |
| - Schedule next crawl |
| - Admin notification |
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

````

## 8. PostgreSQL Database Schema + FAISS Configuration + Major Implementation Examples

### 8.1 PostgreSQL Tables (Metadata Only)
**📦 Component: PostgreSQL Container Service**

```sql
-- init.sql - PostgreSQL Database Schema for Metadata Storage
-- Deploy to: postgres:15-alpine container

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

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

-- Create indexes for better performance
CREATE INDEX idx_document_chunks_doc_id ON document_chunks(doc_id);
CREATE INDEX idx_document_chunks_content_type ON document_chunks(content_type);
CREATE INDEX idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX idx_chat_messages_type ON chat_messages(message_type);
CREATE INDEX idx_processing_jobs_status ON processing_jobs(status);
CREATE INDEX idx_crawl_jobs_status ON crawl_jobs(status);
````

### 8.2 FAISS Configuration (M2 Mac Optimized)

**📦 Component: Vector Service Container**

```python
# faiss_store.py - FAISS Vector Store Implementation
# Deploy to: Vector Service Container (vector_service:latest)

import faiss
import numpy as np
import pickle
import os

class M2OptimizedFAISSStore:
    """
    FAISS Vector Store optimized for MacBook M2 (16GB RAM)
    Handles document embeddings and chat context embeddings
    Used by: Vector Storage & Retrieval Service Container
    """
    def __init__(self, dimension=384, index_type="Flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.document_index = None
        self.chat_index = None
        self.document_id_map = []  # Maps FAISS indices to chunk_ids
        self.chat_id_map = []      # Maps FAISS indices to message_ids

        # Initialize indices optimized for M2 Mac (16GB RAM)
        self._init_document_index()
        self._init_chat_index()

    def _init_document_index(self):
        """Initialize FAISS index optimized for M2 Mac"""
        if self.index_type == "IVFFlat":
            # For M2 Mac: Smaller clusters to save memory
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.document_index = faiss.IndexIVFFlat(quantizer, self.dimension, 50)  # Reduced from 100
            # Set search parameters for better performance on M2
            self.document_index.nprobe = 5  # Reduced probe count
        elif self.index_type == "HNSW":
            # Conservative HNSW settings for M2 Mac
            self.document_index = faiss.IndexHNSWFlat(self.dimension, 16)  # Reduced M parameter
            self.document_index.hnsw.efConstruction = 32  # Reduced construction
            self.document_index.hnsw.efSearch = 16        # Reduced search
        else:
            # Default: Simple flat index - most memory efficient
            self.document_index = faiss.IndexFlatIP(self.dimension)

    def _init_chat_index(self):
        """Initialize FAISS index for chat messages - always simple for M2"""
        self.chat_index = faiss.IndexFlatIP(self.dimension)

    def add_documents(self, embeddings, chunk_ids, batch_size=32):
        """Add document embeddings in smaller batches for M2 Mac"""
        embeddings = np.array(embeddings).astype('float32')

        # Process in smaller batches to manage memory
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i + batch_size]
            batch_ids = chunk_ids[i:i + batch_size]

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(batch_embeddings)

            # Train index if needed (for IVF indices) with smaller training set
            if isinstance(self.document_index, faiss.IndexIVFFlat) and not self.document_index.is_trained:
                if len(batch_embeddings) >= 50:  # Reduced training requirement
                    self.document_index.train(batch_embeddings)

            # Add embeddings
            self.document_index.add(batch_embeddings)

            # Update ID mapping
            self.document_id_map.extend(batch_ids)

        return len(self.document_id_map) - len(chunk_ids)

    def add_chat_messages(self, embeddings, message_ids):
        """Add chat message embeddings - optimized for M2"""
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)

        start_idx = self.chat_index.ntotal
        self.chat_index.add(embeddings)

        # Update ID mapping
        self.chat_id_map.extend(message_ids)

        return start_idx

    def search_documents(self, query_embedding, top_k=5):
        """Search with M2-optimized parameters"""
        query_embedding = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding)

        # Adjust search parameters for M2 performance
        if isinstance(self.document_index, faiss.IndexIVFFlat):
            # Use fewer probes on M2 to save CPU
            original_nprobe = self.document_index.nprobe
            self.document_index.nprobe = min(5, original_nprobe)

        scores, indices = self.document_index.search(query_embedding, top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1 and idx < len(self.document_id_map):  # Bounds check
                results.append({
                    'chunk_id': self.document_id_map[idx],
                    'score': float(score),
                    'faiss_index': int(idx)
                })

        return results

    def search_chat_messages(self, query_embedding, top_k=3):
        """Search chat messages with M2 optimization"""
        query_embedding = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding)

        scores, indices = self.chat_index.search(query_embedding, top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1 and idx < len(self.chat_id_map):  # Bounds check
                results.append({
                    'message_id': self.chat_id_map[idx],
                    'score': float(score),
                    'faiss_index': int(idx)
                })

        return results

    def save_indices(self, base_path="/app/data/faiss"):
        """Save FAISS indices with compression for M2 storage"""
        os.makedirs(base_path, exist_ok=True)

        try:
            # Save FAISS indices
            faiss.write_index(self.document_index, f"{base_path}/document_index.faiss")
            faiss.write_index(self.chat_index, f"{base_path}/chat_index.faiss")

            # Save ID mappings with compression
            with open(f"{base_path}/document_id_map.pkl", 'wb') as f:
                pickle.dump(self.document_id_map, f, protocol=pickle.HIGHEST_PROTOCOL)

            with open(f"{base_path}/chat_id_map.pkl", 'wb') as f:
                pickle.dump(self.chat_id_map, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"Saved indices: {self.document_index.ntotal} docs, {self.chat_index.ntotal} messages")

        except Exception as e:
            print(f"Error saving indices: {e}")

    def load_indices(self, base_path="/app/data/faiss"):
        """Load FAISS indices with error handling for M2"""
        try:
            if not os.path.exists(f"{base_path}/document_index.faiss"):
                print("No existing indices found, starting fresh")
                return False

            # Load FAISS indices
            self.document_index = faiss.read_index(f"{base_path}/document_index.faiss")
            self.chat_index = faiss.read_index(f"{base_path}/chat_index.faiss")

            # Load ID mappings
            with open(f"{base_path}/document_id_map.pkl", 'rb') as f:
                self.document_id_map = pickle.load(f)

            with open(f"{base_path}/chat_id_map.pkl", 'rb') as f:
                self.chat_id_map = pickle.load(f)

            print(f"Loaded indices: {self.document_index.ntotal} docs, {self.chat_index.ntotal} messages")
            return True

        except Exception as e:
            print(f"Failed to load indices: {e}")
            # Reinitialize on error
            self._init_document_index()
            self._init_chat_index()
            self.document_id_map = []
            self.chat_id_map = []
            return False

    def get_memory_usage(self):
        """Get memory usage statistics for monitoring on M2"""
        doc_size = self.document_index.ntotal * self.dimension * 4  # float32
        chat_size = self.chat_index.ntotal * self.dimension * 4
        return {
            "document_vectors": self.document_index.ntotal,
            "chat_vectors": self.chat_index.ntotal,
            "estimated_memory_mb": (doc_size + chat_size) / (1024 * 1024),
            "document_index_type": type(self.document_index).__name__,
            "chat_index_type": type(self.chat_index).__name__
        }
```

### 8.3 Major Implementation Examples for Each Service

#### 8.3.1 Vector Service Container - Complete Implementation

**📦 Component: Vector Service Container (vector_service:latest)**

```python
# vector_service.py - Complete Vector Service Implementation
# Deploy to: Vector Service Container

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
import gc
from faiss_store import M2OptimizedFAISSStore

app = FastAPI(title="Fizen RAG Vector Service", version="1.0.0")

# BGE-Small model setup with M2 optimization
model_name = "BAAI/bge-small-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Set torch to use MPS (Metal Performance Shaders) on M2 if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    model = model.to(device)
    print("✅ Using MPS (Metal) acceleration on M2 Mac")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU inference - MPS not available")

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://fizen_user:fizen_password@postgres:5432/fizen_rag")
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Disable SQL logging to save memory
    pool_size=5,  # Reduced for M2 Mac
    max_overflow=10,
    pool_timeout=30
)

# FAISS Vector Store (M2 optimized)
FAISS_DATA_PATH = os.getenv("FAISS_DATA_PATH", "/app/data/faiss")
vector_store = M2OptimizedFAISSStore(dimension=384, index_type="Flat")  # Use Flat for M2

@app.on_event("startup")
async def startup_event():
    """Load FAISS indices on startup"""
    print("🚀 Starting Vector Service...")
    if vector_store.load_indices(FAISS_DATA_PATH):
        print("✅ FAISS indices loaded successfully")
        memory_stats = vector_store.get_memory_usage()
        print(f"📊 Memory usage: {memory_stats}")
    else:
        print("🆕 No existing FAISS indices found, starting fresh")

@app.on_event("shutdown")
async def shutdown_event():
    """Save FAISS indices on shutdown"""
    vector_store.save_indices(FAISS_DATA_PATH)
    print("💾 FAISS indices saved")

async def embed_text(text: str) -> List[float]:
    """Generate embedding for a single text with M2 optimization"""
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=512)

    # Move to appropriate device
    if device.type == "mps":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    # Move back to CPU for processing
    embedding = output.last_hidden_state.mean(dim=1).cpu().numpy().tolist()[0]

    # Clean up GPU memory on M2
    if device.type == "mps":
        torch.mps.empty_cache()

    return embedding

@app.post("/search/vector")
async def search_documents(req: Request):
    """Vector search endpoint for documents and web content"""
    data = await req.json()
    query = data.get("query")
    top_k = data.get("top_k", 5)
    similarity_threshold = data.get("similarity_threshold", 0.75)

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        # Generate query embedding
        query_embedding = await embed_text(query)

        # Search FAISS index
        faiss_results = vector_store.search_documents(query_embedding, top_k)

        # Filter by similarity threshold
        filtered_results = [r for r in faiss_results if r['score'] >= similarity_threshold]

        if not filtered_results:
            return {"results": [], "message": "No results above similarity threshold"}

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
            enriched_results = []
            for faiss_result in filtered_results:
                chunk_id = faiss_result['chunk_id']
                if chunk_id in metadata_map:
                    metadata = metadata_map[chunk_id]
                    enriched_results.append({
                        **metadata,
                        "chunk_id": chunk_id,
                        "score": faiss_result['score']
                    })

            return {
                "results": enriched_results,
                "total_found": len(enriched_results),
                "search_time_ms": 0  # Could add timing if needed
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")

@app.post("/chat/context")
async def get_chat_context(req: Request):
    """Chat context retrieval endpoint"""
    data = await req.json()
    query = data.get("query")
    session_id = data.get("session_id")
    top_k = data.get("top_k", 3)
    similarity_threshold = data.get("similarity_threshold", 0.7)

    if not query or not session_id:
        raise HTTPException(status_code=400, detail="Query and session_id are required")

    try:
        # Generate query embedding
        query_embedding = await embed_text(query)

        # Search FAISS chat index
        faiss_results = vector_store.search_chat_messages(query_embedding, top_k)

        # Filter by similarity threshold
        filtered_results = [r for r in faiss_results if r['score'] >= similarity_threshold]

        if not filtered_results:
            return {"context_messages": [], "message": "No relevant context found"}

        # Get message metadata from PostgreSQL
        message_ids = [r['message_id'] for r in filtered_results]

        async with engine.begin() as conn:
            result = await conn.execute(
                text("""
                    SELECT
                        id,
                        message_text,
                        created_at,
                        message_type,
                        session_id
                    FROM chat_messages
                    WHERE id = ANY(:message_ids) AND session_id = :session_id AND message_type = 'user'
                    ORDER BY created_at DESC
                """),
                {"message_ids": message_ids, "session_id": session_id}
            )

            # Format chat context results
            context_messages = []
            for row in result:
                context_messages.append({
                    "message": row.message_text,
                    "timestamp": row.created_at.isoformat(),
                    "score": next((r['score'] for r in filtered_results if r['message_id'] == row.id), 0),
                    "message_type": row.message_type
                })

            return {
                "context_messages": context_messages,
                "session_id": session_id,
                "total_context": len(context_messages)
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat context retrieval failed: {str(e)}")

@app.post("/ingest/document")
async def ingest_document(req: Request):
    """Document ingestion endpoint with comprehensive processing"""
    data = await req.json()
    shared_file_path = data.get("shared_file_path")
    metadata = data.get("metadata", {})
    processing_options = data.get("processing_options", {})

    if not shared_file_path:
        raise HTTPException(status_code=400, detail="shared_file_path is required")

    # Extract doc_id from file path or metadata
    doc_id = metadata.get("doc_id") or os.path.basename(shared_file_path).replace('.', '_')

    try:
        # Read file from shared volume
        if not os.path.exists(shared_file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {shared_file_path}")

        # Process file based on type
        with open(shared_file_path, 'rb') as f:
            file_content = f.read()

        # Extract text (simplified - would use proper parsers in production)
        if shared_file_path.endswith('.txt'):
            text_content = file_content.decode('utf-8')
        else:
            # For PDF/DOCX, would use appropriate libraries
            text_content = file_content.decode('utf-8', errors='ignore')

        # Create chunks (simplified chunking)
        chunk_size = processing_options.get("chunk_size", 500)
        chunk_overlap = processing_options.get("chunk_overlap", 20)

        chunks = []
        words = text_content.split()
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page": i // chunk_size + 1,
                    "section": "",
                    "index": len(chunks)
                })

        if not chunks:
            raise HTTPException(status_code=400, detail="No content could be extracted from file")

        # Limit chunks for M2 Mac
        max_chunks = 500
        if len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]
            print(f"⚠️ Limited to {max_chunks} chunks for M2 Mac memory constraints")

        # Process chunks and store in database + FAISS
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
                    "filename": metadata.get("original_filename", "unknown"),
                    "file_type": metadata.get("content_type", "unknown"),
                    "total_chunks": len(chunks)
                }
            )

            # Process chunks in batches
            batch_size = 8  # Small batches for M2 Mac
            chunk_ids = []
            embeddings = []

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i:03d}"
                chunk_ids.append(chunk_id)

                # Generate embedding
                embedding = await embed_text(chunk["text"])
                embeddings.append(embedding)

                # Store chunk metadata in PostgreSQL
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
                        "chunk_index": i,
                        "page_number": chunk.get("page", 0),
                        "section_title": chunk.get("section", ""),
                        "faiss_index": vector_store.document_index.ntotal + i
                    }
                )

                # Process in batches to manage memory
                if len(embeddings) >= batch_size:
                    vector_store.add_documents(embeddings, chunk_ids[-len(embeddings):], batch_size=batch_size)
                    embeddings = []
                    gc.collect()  # Force garbage collection

            # Process remaining embeddings
            if embeddings:
                vector_store.add_documents(embeddings, chunk_ids[-len(embeddings):], batch_size=batch_size)

            # Update document status
            await conn.execute(
                text("UPDATE documents SET status = 'completed' WHERE doc_id = :doc_id"),
                {"doc_id": doc_id}
            )

        # Save FAISS indices
        vector_store.save_indices(FAISS_DATA_PATH)

        return {
            "status": "success",
            "doc_id": doc_id,
            "chunks_created": len(chunks),
            "processing_time": "0s",  # Could add actual timing
            "faiss_index_updated": True,
            "memory_usage": vector_store.get_memory_usage()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Test database connection
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))

        # Get memory stats
        memory_stats = vector_store.get_memory_usage()

        # Check torch/MPS status
        torch_info = {
            "device": str(device),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        }

        if device.type == "mps" and hasattr(torch.mps, 'current_allocated_memory'):
            torch_info["mps_memory_mb"] = torch.mps.current_allocated_memory() / (1024 * 1024)

        return {
            "status": "healthy",
            "service": "vector_service",
            "version": "1.0.0",
            "database": "connected",
            "faiss": memory_stats,
            "torch": torch_info,
            "model": model_name
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "vector_service",
            "error": str(e)
        }

@app.get("/memory")
async def memory_status():
    """Detailed memory monitoring endpoint"""
    memory_stats = vector_store.get_memory_usage()

    # Add torch memory info if using MPS
    if device.type == "mps" and hasattr(torch.mps, 'current_allocated_memory'):
        memory_stats["torch_mps_memory_mb"] = torch.mps.current_allocated_memory() / (1024 * 1024)

    # Add system memory info if available
    try:
        import psutil
        system_memory = psutil.virtual_memory()
        memory_stats["system_memory"] = {
            "total_gb": round(system_memory.total / (1024**3), 2),
            "available_gb": round(system_memory.available / (1024**3), 2),
            "used_percent": system_memory.percent
        }
    except ImportError:
        pass

    return memory_stats

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
```

#### 8.3.2 Retrieval Orchestrator Container - Complete Implementation

**📦 Component: Retrieval Orchestrator Container (retrieval_orchestrator:latest)**

```python
# orchestrator.py - Complete Retrieval Orchestrator Implementation
# Deploy to: Retrieval Orchestrator Container

from fastapi import FastAPI, Request, HTTPException
import aiohttp
import asyncio
from typing import List, Dict, Any
import hashlib
import time
from collections import defaultdict

app = FastAPI(title="Fizen RAG Retrieval Orchestrator", version="1.0.0")

# Service URLs from environment
VECTOR_SERVICE_URL = "http://vector_service:80"
WEB_RETRIEVER_URL = "http://web_retriever:80"

class ResultUnifier:
    """Handles deduplication and ranking of multi-source results"""

    def __init__(self):
        self.similarity_threshold = 0.9  # For content deduplication

    def calculate_content_hash(self, text: str) -> str:
        """Generate hash for content deduplication"""
        # Normalize text for better deduplication
        normalized = text.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity for text comparison"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate content based on similarity"""
        if not results:
            return results

        deduplicated = []
        seen_hashes = set()

        for result in results:
            text = result.get("text", "")
            content_hash = self.calculate_content_hash(text)

            # Check exact hash match first
            if content_hash in seen_hashes:
                continue

            # Check semantic similarity with existing results
            is_duplicate = False
            for existing in deduplicated:
                similarity = self.calculate_text_similarity(text, existing.get("text", ""))
                if similarity > self.similarity_threshold:
                    # Keep the one with higher score
                    if result.get("score", 0) > existing.get("score", 0):
                        deduplicated.remove(existing)
                        break
                    else:
                        is_duplicate = True
                        break

            if not is_duplicate:
                deduplicated.append(result)
                seen_hashes.add(content_hash)

        return deduplicated

    def apply_multi_factor_ranking(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply sophisticated ranking considering multiple factors"""

        def calculate_composite_score(result: Dict[str, Any]) -> float:
            base_score = result.get("score", 0.0)
            source_type = result.get("type", "unknown")
            content_type = result.get("content_type", "general")

            # Source authority weights
            source_weights = {
                "vector": 1.0,     # Internal documents
                "sql": 1.2,        # Structured data (often most reliable)
                "web": 0.8         # External web content
            }

            # Content type weights
            content_weights = {
                "document": 1.0,
                "government": 1.1,
                "academic": 1.1,
                "encyclopedia": 0.9,
                "general": 0.8
            }

            # Recency bonus for web content
            recency_bonus = 0.0
            if source_type == "web":
                # Could add timestamp-based recency scoring here
                recency_bonus = 0.05

            # Calculate composite score
            source_weight = source_weights.get(source_type, 0.7)
            content_weight = content_weights.get(content_type, 0.8)

            composite_score = (base_score * source_weight * content_weight) + recency_bonus

            return min(composite_score, 1.0)  # Cap at 1.0

        # Calculate composite scores
        for result in results:
            result["composite_score"] = calculate_composite_score(result)

        # Sort by composite score
        return sorted(results, key=lambda x: x["composite_score"], reverse=True)

@app.post("/orchestrate/retrieve")
async def orchestrate_retrieval(req: Request):
    """Main retrieval orchestration endpoint"""
    data = await req.json()
    query = data.get("query")
    context = data.get("context", "")
    max_results = data.get("max_results", 10)
    source_types = data.get("source_types", ["vector", "sql", "web"])

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    print(f"🔍 Orchestrating retrieval for query: {query[:50]}...")

    try:
        # Initialize result unifier
        unifier = ResultUnifier()

        # Prepare parallel requests to different sources
        tasks = []

        # Vector search task
        if "vector" in source_types:
            tasks.append(call_vector_service(query, max_results // 2))

        # SQL search task
        if "sql" in source_types:
            tasks.append(call_sql_service(query))

        # Web search task
        if "web" in source_types:
            tasks.append(call_web_service(query, max_results // 3))

        # Execute all searches in parallel
        start_time = time.time()
        search_results = await asyncio.gather(*tasks, return_exceptions=True)
        search_time = time.time() - start_time

        # Combine and process results
        all_results = []
        source_counts = defaultdict(int)

        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                print(f"⚠️ Source {i} failed: {str(result)}")
                continue

            if isinstance(result, list):
                for item in result:
                    source_type = item.get("type", "unknown")
                    source_counts[source_type] += 1
                    all_results.append(item)

        print(f"📊 Raw results: {dict(source_counts)}")

        # Deduplicate results
        deduplicated_results = unifier.deduplicate_results(all_results)
        print(f"🔄 After deduplication: {len(deduplicated_results)} results")

        # Apply multi-factor ranking
        ranked_results = unifier.apply_multi_factor_ranking(deduplicated_results)

        # Limit final results
        final_results = ranked_results[:max_results]

        return {
            "results": final_results,
            "metadata": {
                "total_sources_queried": len(tasks),
                "source_counts": dict(source_counts),
                "results_before_dedup": len(all_results),
                "results_after_dedup": len(deduplicated_results),
                "final_result_count": len(final_results),
                "search_time_ms": round(search_time * 1000, 2),
                "query": query[:100] + "..." if len(query) > 100 else query
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval orchestration failed: {str(e)}")

async def call_vector_service(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Call Vector Service for document search"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "query": query,
                "top_k": max_results,
                "similarity_threshold": 0.75
            }

            async with session.post(f"{VECTOR_SERVICE_URL}/search/vector", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get("results", [])

                    # Normalize format
                    normalized_results = []
                    for result in results:
                        normalized_results.append({
                            "type": "vector",
                            "text": result.get("text", ""),
                            "source": result.get("source", ""),
                            "score": result.get("score", 0.0),
                            "content_type": result.get("content_type", "document"),
                            "metadata": {
                                "chunk_id": result.get("chunk_id"),
                                "page": result.get("page"),
                                "section": result.get("section")
                            }
                        })

                    return normalized_results
                else:
                    print(f"⚠️ Vector service returned status {response.status}")
                    return []
    except Exception as e:
        print(f"❌ Vector service call failed: {str(e)}")
        return []

async def call_sql_service(query: str) -> List[Dict[str, Any]]:
    """Call SQL service for structured data"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"query": query}

            async with session.post(f"{WEB_RETRIEVER_URL}/sql/query", json=payload) as response:
                if response.status == 200:
                    data = await response.json()

                    # Convert SQL results to normalized format
                    if isinstance(data, list) and data:
                        # Format SQL results as text
                        sql_text = "Query results: " + str(data)
                        return [{
                            "type": "sql",
                            "text": sql_text,
                            "source": "database",
                            "score": 1.0,  # SQL results are considered highly relevant
                            "content_type": "structured_data",
                            "metadata": {"raw_data": data}
                        }]
                    return []
                else:
                    print(f"⚠️ SQL service returned status {response.status}")
                    return []
    except Exception as e:
        print(f"❌ SQL service call failed: {str(e)}")
        return []

async def call_web_service(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Call Web Retriever for real-time search"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "query": query,
                "max_results": max_results
            }

            async with session.post(f"{WEB_RETRIEVER_URL}/search/realtime", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data if isinstance(data, list) else []

                    # Normalize format
                    normalized_results = []
                    for result in results:
                        normalized_results.append({
                            "type": "web",
                            "text": result.get("text", result.get("summary", "")),
                            "source": result.get("source", result.get("url", "")),
                            "score": result.get("score", result.get("relevance_score", 0.0)),
                            "content_type": result.get("content_type", "general"),
                            "metadata": {
                                "url": result.get("url"),
                                "title": result.get("title"),
                                "domain": result.get("domain")
                            }
                        })

                    return normalized_results
                else:
                    print(f"⚠️ Web service returned status {response.status}")
                    return []
    except Exception as e:
        print(f"❌ Web service call failed: {str(e)}")
        return []

@app.get("/orchestrate/sources")
async def get_source_status():
    """Check status of all data sources"""
    sources = {
        "vector_service": VECTOR_SERVICE_URL,
        "web_retriever": WEB_RETRIEVER_URL
    }

    status_results = {}

    for service_name, url in sources.items():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/health", timeout=5) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        status_results[service_name] = {
                            "status": "healthy",
                            "response_time_ms": 0,  # Could add actual timing
                            "details": health_data
                        }
                    else:
                        status_results[service_name] = {
                            "status": "unhealthy",
                            "response_code": response.status
                        }
        except Exception as e:
            status_results[service_name] = {
                "status": "unreachable",
                "error": str(e)
            }

    overall_healthy = all(s.get("status") == "healthy" for s in status_results.values())

    return {
        "overall_status": "healthy" if overall_healthy else "degraded",
        "sources": status_results,
        "timestamp": time.time()
    }

@app.get("/health")
async def health_check():
    """Health check for orchestrator service"""
    return {
        "status": "healthy",
        "service": "retrieval_orchestrator",
        "version": "1.0.0",
        "dependencies": {
            "vector_service": VECTOR_SERVICE_URL,
            "web_retriever": WEB_RETRIEVER_URL
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
```

## 8. Deployment Guide: MacBook M2 (16GB RAM) + Docker + FAISS + Ollama + SearxNG

### 8.1 MacBook M2 Optimized Docker Compose Configuration

**📦 Component: Docker Orchestration (All Services)**

```yaml
# docker-compose.yml - Complete M2 Mac optimized deployment
# Deploys: 6 Container Services (PostgreSQL, SearxNG, Ollama, Vector Service, API Gateway, Frontend)

version: "3.8"
services:
  postgres:
    # 📦 PostgreSQL Database Container
    image: postgres:15-alpine # Lighter alpine image for ARM64
    platform: linux/arm64
    environment:
      POSTGRES_DB: fizen_rag
      POSTGRES_USER: fizen_user
      POSTGRES_PASSWORD: fizen_password
      # Optimize for limited RAM
      POSTGRES_INITDB_ARGS: "--data-checksums"
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - fizen_rag_network
    # Resource limits for M2 Mac (16GB total)
    mem_limit: 1g
    mem_reservation: 512m
    cpus: 1
    command: |
      postgres 
      -c shared_buffers=128MB 
      -c max_connections=50
      -c effective_cache_size=256MB
      -c maintenance_work_mem=32MB
      -c checkpoint_completion_target=0.9
      -c wal_buffers=8MB
      -c default_statistics_target=50
      -c random_page_cost=1.1
      -c effective_io_concurrency=200

  searxng:
    # 📦 SearxNG Search Engine Container
    image: searxng/searxng:latest
    platform: linux/arm64
    ports:
      - "8080:8080"
    environment:
      - SEARXNG_BASE_URL=http://localhost:8080/
    volumes:
      - searxng_data:/etc/searxng
      - ./searxng_settings.yml:/etc/searxng/settings.yml:ro
    networks:
      - fizen_rag_network
    # Lightweight search service
    mem_limit: 512m
    mem_reservation: 256m
    cpus: 1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  ollama:
    # 📦 Ollama LLM Inference Container
    image: ollama/ollama:latest
    platform: linux/arm64 # Explicit ARM64 for M2
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
      # Optimize for M2 Mac
      - OLLAMA_NUM_PARALLEL=1
      - OLLAMA_MAX_LOADED_MODELS=1
      - OLLAMA_FLASH_ATTENTION=1
    networks:
      - fizen_rag_network
    # Resource limits optimized for M2 Mac
    mem_limit: 5g # Reduced from 6GB to accommodate SearxNG
    mem_reservation: 2g
    cpus: 4 # Use 4 cores for LLM inference
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3

  vector_service:
    # 📦 Vector Storage & Retrieval Service Container
    build:
      context: ./vector_service
      dockerfile: Dockerfile.m2
    platform: linux/arm64
    ports:
      - "8001:80"
    environment:
      - DATABASE_URL=postgresql://fizen_user:fizen_password@postgres:5432/fizen_rag
      - FAISS_DATA_PATH=/app/data/faiss
      - SHARED_FILES_PATH=/app/data/files # Shared file storage path
      - SEARXNG_URL=http://searxng:8080
      # ARM64 optimizations
      - OMP_NUM_THREADS=2
      - MKL_NUM_THREADS=2
      - FAISS_LARGE_DATASET=false
      - EMBEDDING_BATCH_SIZE=8 # Smaller batches for limited memory
    volumes:
      - faiss_data:/app/data/faiss
      - uploaded_files:/app/data/files # Shared volume for document storage
    depends_on:
      postgres:
        condition: service_started
      searxng:
        condition: service_healthy
    networks:
      - fizen_rag_network
    # Resource limits for vector service
    mem_limit: 3g
    mem_reservation: 1g
    cpus: 2
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:80/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  retrieval_orchestrator:
    # 📦 Retrieval Orchestrator Container
    build:
      context: ./retrieval_orchestrator
      dockerfile: Dockerfile.m2
    platform: linux/arm64
    ports:
      - "8003:80"
    environment:
      - DATABASE_URL=postgresql://fizen_user:fizen_password@postgres:5432/fizen_rag
      - VECTOR_SERVICE_URL=http://vector_service:80
      - WEB_RETRIEVER_URL=http://web_retriever:80
    depends_on:
      postgres:
        condition: service_started
      vector_service:
        condition: service_healthy
      web_retriever:
        condition: service_healthy
    networks:
      - fizen_rag_network
    mem_limit: 512m
    mem_reservation: 256m
    cpus: 1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:80/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  web_retriever:
    # 📦 Web Retriever & SQL Service Container
    build:
      context: ./web_retriever
      dockerfile: Dockerfile.m2
    platform: linux/arm64
    ports:
      - "8005:80"
    environment:
      - DATABASE_URL=postgresql://fizen_user:fizen_password@postgres:5432/fizen_rag
      - SEARXNG_URL=http://searxng:8080
      - OLLAMA_URL=http://ollama:11434
      - VECTOR_SERVICE_URL=http://vector_service:80
    depends_on:
      postgres:
        condition: service_started
      searxng:
        condition: service_healthy
      ollama:
        condition: service_healthy
    networks:
      - fizen_rag_network
    mem_limit: 1g
    mem_reservation: 256m
    cpus: 1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:80/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  api_gateway:
    # 📦 API Gateway Container (Context Manager + Prompt Builder + LLM Orchestrator + Response Formatter)
    build:
      context: ./api_gateway
      dockerfile: Dockerfile.m2
    platform: linux/arm64
    ports:
      - "8000:80"
    environment:
      - DATABASE_URL=postgresql://fizen_user:fizen_password@postgres:5432/fizen_rag
      - VECTOR_SERVICE_URL=http://vector_service:80
      - RETRIEVAL_ORCHESTRATOR_URL=http://retrieval_orchestrator:80
      - OLLAMA_URL=http://ollama:11434
      - SHARED_FILES_PATH=/app/data/files # Shared file storage path
      # Performance tuning
      - WORKERS=2
      - MAX_REQUESTS=100
    volumes:
      - uploaded_files:/app/data/files # Shared volume for document storage
    depends_on:
      postgres:
        condition: service_started
      vector_service:
        condition: service_healthy
      retrieval_orchestrator:
        condition: service_healthy
      ollama:
        condition: service_healthy
    networks:
      - fizen_rag_network
    # Reduced memory since orchestrator is separate
    mem_limit: 1g
    mem_reservation: 256m
    cpus: 1

  frontend:
    # 📦 React Frontend Container
    build:
      context: ./frontend
      dockerfile: Dockerfile.m2
    platform: linux/arm64
    ports:
      - "3000:80"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - api_gateway
    networks:
      - fizen_rag_network
    # Minimal resources for static frontend
    mem_limit: 512m
    mem_reservation: 128m
    cpus: 0.5

volumes:
  postgres_data:
    driver: local
  ollama_data:
    driver: local
  faiss_data:
    driver: local
  uploaded_files:
    driver: local
  searxng_data:
    driver: local

networks:
  fizen_rag_network:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.name: fizen_rag_br
```

### 8.2 MacBook M2 Setup Instructions

#### Prerequisites:

**📦 Component: System Setup (MacOS Host)**

```bash
# setup_prerequisites_m2.sh - System prerequisites for M2 Mac
# Deploy to: MacBook M2 Host System

# Install Docker Desktop for Mac (Apple Silicon)
# Download from: https://docs.docker.com/desktop/mac/install/

# Verify Docker is using ARM64
docker version --format "{{.Server.Arch}}"
# Should output: arm64

# Configure Docker Desktop for M2 Mac:
# - Go to Docker Desktop > Preferences > Resources
# - Set Memory limit to 12GB (leave 4GB for macOS)
# - Set CPU limit to 6 cores (leave 2 for macOS)
# - Enable "Use Rosetta for x86/amd64 emulation" if needed
```

#### File Storage Architecture for Container Communication:

**📦 Component: Shared Storage Between API Gateway and Vector Service Containers**

**Problem**: Vector Service cannot access files stored in API Gateway's local container file system.

**Solution Options**:

1. **Shared Docker Volume (Recommended)**:

   ```yaml
   volumes: uploaded_files:/app/data/files # Both containers mount same volume
   ```

   - **Pros**: Simple, performant, Docker-native
   - **Cons**: Tied to single Docker host
   - **Use Case**: Single-machine deployments (M2 Mac setup)

2. **File Content Transfer via API**:

   ```python
   # API Gateway reads file and sends content
   file_content = await file.read()
   response = await http_client.post("/ingest/document", {
       "content": base64.encode(file_content),
       "metadata": {...}
   })
   ```

   - **Pros**: No shared storage needed, works across hosts
   - **Cons**: Memory intensive for large files, network overhead
   - **Use Case**: Small files, distributed deployments

3. **External Storage Service**:
   ```python
   # Both containers access S3/MinIO/NFS
   file_url = await upload_to_s3(file)
   response = await http_client.post("/ingest/document", {
       "file_url": file_url,
       "metadata": {...}
   })
   ```
   - **Pros**: Scalable, distributed-ready, persistent
   - **Cons**: Additional infrastructure complexity
   - **Use Case**: Production distributed deployments

**Current Implementation**: Shared Docker Volume for M2 Mac optimization

#### Project Structure:

**📦 Component: Project Organization (All Services)**

```
fizen-rag/
├── docker-compose.yml              # 📦 Orchestration Config
├── init.sql                        # 📦 PostgreSQL Schema
├── setup_ollama_m2.sh             # 📦 Ollama Setup Script
├── searxng_settings.yml           # 📦 SearxNG Configuration
├── setup_searxng.sh               # 📦 SearxNG Setup Script
├── vector_service/                 # 📦 Vector Service Container
│   ├── Dockerfile.m2
│   ├── vector_service.py
│   ├── faiss_store.py
│   └── requirements.txt
├── retrieval_orchestrator/         # 📦 Retrieval Orchestrator Container
│   ├── Dockerfile.m2
│   ├── orchestrator.py            # Multi-source retrieval coordination
│   ├── result_unifier.py          # Result deduplication & ranking
│   └── requirements.txt
├── web_retriever/                  # 📦 Web Retriever & SQL Service Container
│   ├── Dockerfile.m2
│   ├── web_retriever.py           # Web crawling & real-time search
│   ├── sql_retriever.py           # Natural language to SQL
│   ├── main.py                    # FastAPI service entry point
│   └── requirements.txt
├── api_gateway/                    # 📦 API Gateway Container
│   ├── Dockerfile.m2
│   ├── main.py                    # Main FastAPI app
│   ├── context_manager.py         # Context enhancement module
│   ├── prompt_builder.py          # Prompt formatting module
│   ├── llm_orchestrator.py        # LLM interaction module
│   ├── response_formatter.py      # Response formatting module
│   ├── file_upload.py             # File upload handler
│   └── requirements.txt
├── frontend/                       # 📦 Frontend Container
│   ├── Dockerfile.m2
│   ├── package.json
│   └── src/
└── data/                          # Shared Storage (Docker Volumes)
    ├── faiss/                     # FAISS indices (📦 Vector Service only)
    ├── files/                     # Uploaded documents (📦 Shared: API Gateway + Vector Service)
    │   ├── doc_001/
    │   │   └── company_policies.pdf
    │   └── doc_002/
    │       └── employee_handbook.docx
    ├── postgres/                  # PostgreSQL data (📦 PostgreSQL only)
    ├── ollama/                    # Ollama models (📦 Ollama only)
    └── searxng/                   # SearxNG config & cache (📦 SearxNG only)
```

**Container Mount Points**:

- **📦 API Gateway Container**: `/app/data/files` → `uploaded_files` volume (read/write)
- **📦 Vector Service Container**: `/app/data/files` → `uploaded_files` volume (read-only)
- **📦 Vector Service Container**: `/app/data/faiss` → `faiss_data` volume (read/write)

#### API Gateway File Upload Implementation:

**📦 Component: API Gateway Container File Upload Handler**

```python
# api_gateway/file_upload.py - File upload handler with shared volume storage
# Deploy to: API Gateway Container

import os
import uuid
import aiofiles
from fastapi import UploadFile, HTTPException
import aiohttp

SHARED_FILES_PATH = os.getenv("SHARED_FILES_PATH", "/app/data/files")
VECTOR_SERVICE_URL = os.getenv("VECTOR_SERVICE_URL", "http://vector_service:80")

async def handle_document_upload(file: UploadFile, metadata: dict) -> dict:
    """Handle document upload and initiate processing"""

    # Generate unique document ID
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    doc_dir = os.path.join(SHARED_FILES_PATH, doc_id)

    # Create document directory in shared volume
    os.makedirs(doc_dir, exist_ok=True)

    # Save file to shared volume (accessible by Vector Service)
    file_path = os.path.join(doc_dir, file.filename)

    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)

    # Prepare processing request for Vector Service
    processing_request = {
        "shared_file_path": file_path,  # Path accessible to Vector Service
        "metadata": {
            "original_filename": file.filename,
            "content_type": file.content_type,
            "file_size_bytes": len(content),
            **metadata
        },
        "processing_options": {
            "chunk_size": 500,
            "chunk_overlap": 20,
            "language": "en"
        }
    }

    # Call Vector Service for processing
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{VECTOR_SERVICE_URL}/ingest/document",
            json=processing_request
        ) as response:
            if response.status != 200:
                # Clean up file on processing failure
                os.remove(file_path)
                os.rmdir(doc_dir)
                raise HTTPException(status_code=500, detail="Document processing failed")

            result = await response.json()
            return {
                "doc_id": doc_id,
                "status": "processing",
                "shared_file_path": file_path,
                **result
            }
```

```
fizen-rag/
├── docker-compose.yml              # 📦 Orchestration Config
├── init.sql                        # 📦 PostgreSQL Schema
├── setup_ollama_m2.sh             # 📦 Ollama Setup Script
├── searxng_settings.yml           # 📦 SearxNG Configuration
├── setup_searxng.sh               # 📦 SearxNG Setup Script
├── vector_service/                 # 📦 Vector Service Container
│   ├── Dockerfile.m2
│   ├── vector_service.py
│   ├── faiss_store.py
│   └── requirements.txt
├── api_gateway/                    # 📦 API Gateway Container
│   ├── Dockerfile.m2
│   ├── main.py
│   ├── web_retriever.py           # SearxNG Integration
│   └── requirements.txt
└── frontend/                       # 📦 Frontend Container
    ├── Dockerfile.m2
    ├── package.json
    └── src/
```

#### SearxNG Configuration (searxng_settings.yml):

**📦 Component: SearxNG Container Configuration**

```yaml
# searxng_settings.yml - SearxNG configuration optimized for RAG use case
# Deploy to: SearxNG Container (searxng/searxng:latest)

use_default_settings: true

general:
  debug: false
  instance_name: "Fizen RAG Search"
  donation_url: false
  contact_url: false
  enable_metrics: false

search:
  safe_search: 1
  autocomplete: ""
  default_lang: "en"
  ban_time_on_fail: 5
  max_ban_time_on_fail: 120
  formats:
    - html
    - json

server:
  port: 8080
  bind_address: "0.0.0.0"
  secret_key: "fizen-rag-secret-key-change-in-production"
  base_url: "http://localhost:8080/"
  image_proxy: false
  method: "GET"

ui:
  static_use_hash: false
  default_locale: "en"
  theme_args:
    simple_style: "dark"
  results_on_new_tab: false
  advanced_search: true
  search_on_category_select: true
  hotkeys: default

# Enabled engines for RAG (optimized for quality results)
engines:
  - name: bing
    engine: bing
    shortcut: bi
    categories: [general, web]
    disabled: false

  - name: duckduckgo
    engine: duckduckgo
    shortcut: ddg
    categories: [general, web]
    disabled: false

  - name: google
    engine: google
    shortcut: go
    categories: [general, web]
    disabled: false
    use_mobile_ui: false

  - name: startpage
    engine: startpage
    shortcut: sp
    categories: [general, web]
    disabled: false

  - name: wikipedia
    engine: wikipedia
    shortcut: wp
    categories: [general]
    base_url: "https://{language}.wikipedia.org/"
    number_of_results: 5

  - name: arxiv
    engine: arxiv
    shortcut: arx
    categories: [science]
    disabled: false

  - name: github
    engine: github
    shortcut: gh
    categories: [it]
    disabled: false

# Disable engines that might be slow or unreliable for RAG
disabled_engines:
  - "torrentz"
  - "piratebay"
  - "nyaa"
  - "youtube"
  - "dailymotion"
  - "vimeo"
  - "soundcloud"
  - "spotify"
```

#### SearxNG Setup Script (setup_searxng.sh):

**📦 Component: SearxNG Container Setup**

```bash
#!/bin/bash
# setup_searxng.sh - Setup SearxNG for MacBook M2
# Deploy to: SearxNG Container initialization

echo "Setting up SearxNG for Fizen RAG..."

# Create SearxNG settings file if it doesn't exist
if [ ! -f "searxng_settings.yml" ]; then
    echo "Creating default SearxNG configuration..."
    cat > searxng_settings.yml << 'EOF'
use_default_settings: true
general:
  instance_name: "Fizen RAG Search"
  debug: false
search:
  safe_search: 1
  default_lang: "en"
server:
  port: 8080
  bind_address: "0.0.0.0"
  secret_key: "fizen-rag-secret-$(openssl rand -hex 16)"
engines:
  - name: duckduckgo
    disabled: false
  - name: bing
    disabled: false
  - name: google
    disabled: false
  - name: wikipedia
    disabled: false
EOF
fi

# Wait for SearxNG to be ready
echo "Waiting for SearxNG to start..."
sleep 10

# Function to check if SearxNG is ready
check_searxng() {
    curl -s http://localhost:8080/healthz > /dev/null
    return $?
}

# Wait for SearxNG to be responsive
until check_searxng; do
    echo "Waiting for SearxNG to be ready..."
    sleep 5
done

echo "SearxNG is ready!"

# Test search functionality
echo "Testing SearxNG search..."
search_result=$(curl -s "http://localhost:8080/search?q=test&format=json")
if echo "$search_result" | grep -q "results"; then
    echo "✓ SearxNG search is working"
else
    echo "⚠ SearxNG search test failed"
fi

echo "SearxNG setup complete!"
echo "Access SearxNG at: http://localhost:8080"
echo "Search API endpoint: http://localhost:8080/search?q=QUERY&format=json"
```

### 8.3 Vector Service with FAISS (M2 Optimized)

**📦 Component: Vector Service Container**

#### Dockerfile.m2:

**📦 Component: Vector Service Container Build**

```dockerfile
# Dockerfile.m2 - Vector Service Container for M2 Mac
# Deploy to: Vector Service Container (fizen-rag-vector-service)

# Use ARM64-compatible Python base image
FROM python:3.11-slim-bullseye

# Set platform explicitly for M2 Mac
ENV DOCKER_DEFAULT_PLATFORM=linux/arm64

# Install system dependencies optimized for ARM64
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    cmake \
    libblas-dev \
    liblapack-dev \
    gfortran \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for ARM64 optimization
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=2

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with ARM64 optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    transformers==4.35.0 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    sqlalchemy==2.0.23 \
    psycopg2-binary==2.9.9 \
    asyncpg==0.29.0 \
    faiss-cpu==1.7.4 \
    numpy==1.24.3 \
    pandas==2.1.3 \
    sentence-transformers==2.2.2 \
    aiofiles==23.2.1

COPY . /app

# Create data directories
RUN mkdir -p /app/data/faiss /app/data/files

# Set proper permissions
RUN chmod +x /app/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

CMD ["uvicorn", "vector_service:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]
```

#### requirements.txt:

**📦 Component: Vector Service Container Dependencies**

```txt
# requirements.txt - Vector Service Python dependencies
# Deploy to: Vector Service Container

torch==2.1.0
transformers==4.35.0
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
asyncpg==0.29.0
faiss-cpu==1.7.4
numpy==1.24.3
pandas==2.1.3
sentence-transformers==2.2.2
aiofiles==23.2.1
readability-lxml==0.8.1
beautifulsoup4==4.12.2
aiohttp==3.9.1
```

#### Code Example (M2 Optimized):

**📦 Component: Vector Service Container Main Application**

```python
# vector_service.py - Main Vector Service Application
# Deploy to: Vector Service Container (fizen-rag-vector-service)

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
import gc
from faiss_store import M2OptimizedFAISSStore

# BGE-Small model setup with M2 optimization
model_name = "BAAI/bge-small-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Set torch to use MPS (Metal Performance Shaders) on M2 if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    model = model.to(device)
    print("Using MPS (Metal) acceleration on M2 Mac")
else:
    device = torch.device("cpu")
    print("Using CPU inference")

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://fizen_user:fizen_password@postgres:5432/fizen_rag")
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Disable SQL logging to save memory
    pool_size=5,  # Reduced for M2 Mac
    max_overflow=10,
    pool_timeout=30
)

# FAISS Vector Store (M2 optimized)
FAISS_DATA_PATH = os.getenv("FAISS_DATA_PATH", "/app/data/faiss")
vector_store = M2OptimizedFAISSStore(dimension=384, index_type="Flat")  # Use Flat for M2

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Load FAISS indices on startup"""
    if vector_store.load_indices(FAISS_DATA_PATH):
        print("FAISS indices loaded successfully")
        memory_stats = vector_store.get_memory_usage()
        print(f"Memory usage: {memory_stats}")
    else:
        print("No existing FAISS indices found, starting fresh")

@app.on_event("shutdown")
async def shutdown_event():
    """Save FAISS indices on shutdown"""
    vector_store.save_indices(FAISS_DATA_PATH)
    print("FAISS indices saved")

async def embed_text(text: str) -> List[float]:
    """Generate embedding for a single text with M2 optimization"""
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=512)

    # Move to appropriate device
    if device.type == "mps":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    # Move back to CPU for processing
    embedding = output.last_hidden_state.mean(dim=1).cpu().numpy().tolist()[0]

    # Clean up GPU memory on M2
    if device.type == "mps":
        torch.mps.empty_cache()

    return embedding

async def embed_texts(texts: List[str], batch_size: int = 8) -> List[List[float]]:
    """Generate embeddings for multiple texts with M2-optimized batching"""
    all_embeddings = []

    # Process in smaller batches for M2 memory management
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

        # Move to appropriate device
        if device.type == "mps":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)

        # Move back to CPU and convert to list
        batch_embeddings = output.last_hidden_state.mean(dim=1).cpu().numpy().tolist()
        all_embeddings.extend(batch_embeddings)

        # Clean up memory after each batch
        del output, inputs
        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

    return all_embeddings

@app.post("/embed")
async def embed_endpoint(req: Request):
    """Embedding endpoint for Vector Service Container"""
    data = await req.json()
    texts = data.get("texts", [])

    # Limit batch size for M2 Mac
    if len(texts) > 32:
        return {"error": "Batch size too large for M2 Mac, maximum 32 texts"}

    embeddings = await embed_texts(texts)
    return {"embeddings": embeddings}

@app.post("/ingest/document")
async def ingest_document(req: Request):
    """Document ingestion endpoint for Vector Service Container"""
    data = await req.json()
    doc_id = data.get("doc_id")
    filename = data.get("filename", "")
    file_type = data.get("file_type", "")
    chunks = data.get("chunks", [])

    # Limit chunk processing for M2 Mac
    max_chunks = 500  # Reasonable limit for M2 Mac
    if len(chunks) > max_chunks:
        return {"error": f"Too many chunks for M2 Mac, maximum {max_chunks}"}

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

            # Process chunks in smaller batches for M2 Mac
            batch_size = 8  # Smaller batches for limited memory
            chunk_ids = []
            embeddings = []

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_texts = [chunk["text"] for chunk in batch_chunks]
                batch_embeddings = await embed_texts(batch_texts, batch_size=batch_size)

                # Store metadata in PostgreSQL
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

                # Force garbage collection after each batch
                gc.collect()

            # Add embeddings to FAISS with M2-optimized batching
            vector_store.add_documents(embeddings, chunk_ids, batch_size=16)

            # Update document status
            await conn.execute(
                text("UPDATE documents SET status = 'completed' WHERE doc_id = :doc_id"),
                {"doc_id": doc_id}
            )

        # Save FAISS indices
        vector_store.save_indices(FAISS_DATA_PATH)

        # Get memory usage for monitoring
        memory_stats = vector_store.get_memory_usage()

        return {
            "status": "success",
            "chunks_stored": len(chunks),
            "doc_id": doc_id,
            "faiss_index_updated": True,
            "memory_usage_mb": memory_stats["estimated_memory_mb"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/search/vector")
async def retrieve_chunks(req: Request):
    """Vector search endpoint for Vector Service Container"""
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

@app.post("/chat/context")
async def get_chat_context(req: Request):
    """Chat context retrieval endpoint for Vector Service Container"""
    data = await req.json()
    query = data.get("query")
    session_id = data.get("session_id")
    top_k = data.get("top_k", 3)
    similarity_threshold = data.get("similarity_threshold", 0.7)

    try:
        # Generate query embedding
        query_embedding = await embed_text(query)

        # Search FAISS chat index
        faiss_results = vector_store.search_chat_messages(query_embedding, top_k)

        # Filter by similarity threshold
        filtered_results = [r for r in faiss_results if r['score'] >= similarity_threshold]

        if not filtered_results:
            return []

        # Get message metadata from PostgreSQL
        message_ids = [r['message_id'] for r in filtered_results]

        async with engine.begin() as conn:
            result = await conn.execute(
                text("""
                    SELECT
                        id,
                        message_text,
                        created_at,
                        message_type,
                        session_id
                    FROM chat_messages
                    WHERE id = ANY(:message_ids) AND session_id = :session_id AND message_type = 'user'
                """),
                {"message_ids": message_ids, "session_id": session_id}
            )

            # Format chat context results
            context_messages = []
            for row in result:
                context_messages.append({
                    "message": row.message_text,
                    "timestamp": row.created_at.isoformat(),
                    "score": next((r['score'] for r in filtered_results if r['message_id'] == row.id), 0),
                    "message_type": row.message_type
                })

            return context_messages

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat context retrieval failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint for Vector Service Container"""
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))

        memory_stats = vector_store.get_memory_usage()

        return {
            "status": "healthy",
            "database": "connected",
            "faiss": memory_stats,
            "device": str(device),
            "torch_mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/memory")
async def memory_status():
    """Memory monitoring endpoint for Vector Service Container"""
    memory_stats = vector_store.get_memory_usage()

    # Add torch memory info if using MPS
    if device.type == "mps" and hasattr(torch.mps, 'current_allocated_memory'):
        memory_stats["torch_mps_memory_mb"] = torch.mps.current_allocated_memory() / (1024 * 1024)

    return memory_stats
```

### 8.4 Web Retriever with SearxNG Integration

#### web_retriever.py (SearxNG Implementation):

**📦 Component: API Gateway Container Web Retriever Module**

```python
# web_retriever.py - SearxNG Web Retriever Implementation
# Deploy to: API Gateway Container (fizen-rag-api-gateway)

import requests
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from readability import Document
from typing import List, Dict, Any, Optional
import os
import logging
from urllib.parse import urljoin, urlparse
import time

class SearxNGWebRetriever:
    def __init__(self, searxng_url: str = "http://searxng:8080"):
        self.searxng_url = searxng_url
        self.session = None
        self.domain_whitelist = {
            "gov", "edu", "org", "wikipedia.org", "github.com",
            "stackoverflow.com", "arxiv.org", "nature.com", "ieee.org"
        }

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Fizen-RAG/1.0 (Educational Search Bot)'}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def search_current_info(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for current information using SearxNG"""
        try:
            # Prepare search parameters
            search_params = {
                'q': query,
                'format': 'json',
                'categories': 'general',
                'engines': 'duckduckgo,bing,google,wikipedia',
                'safesearch': '1',
                'time_range': None  # Get latest results
            }

            # Make search request to SearxNG
            search_url = f"{self.searxng_url}/search"

            async with self.session.get(search_url, params=search_params) as response:
                if response.status != 200:
                    logging.error(f"SearxNG search failed: {response.status}")
                    return []

                search_data = await response.json()

            # Process search results
            results = []
            search_results = search_data.get('results', [])

            for item in search_results[:max_results * 2]:  # Get extra to filter
                url = item.get('url', '')
                title = item.get('title', '')
                content = item.get('content', '')

                # Apply domain filtering
                if self._is_trusted_domain(url):
                    # Extract more content from the page
                    full_content = await self._extract_page_content(url)

                    if full_content:
                        results.append({
                            'url': url,
                            'title': title,
                            'summary': content[:500] + "..." if len(content) > 500 else content,
                            'full_content': full_content,
                            'domain': urlparse(url).netloc,
                            'content_type': self._classify_content_type(url),
                            'crawled_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                            'relevance_score': self._calculate_relevance_score(query, title, content)
                        })

                # Stop when we have enough quality results
                if len(results) >= max_results:
                    break

            # Sort by relevance score
            results.sort(key=lambda x: x['relevance_score'], reverse=True)

            return results[:max_results]

        except Exception as e:
            logging.error(f"SearxNG search error: {str(e)}")
            return []

    def _is_trusted_domain(self, url: str) -> bool:
        """Check if URL is from a trusted domain"""
        try:
            domain = urlparse(url).netloc.lower()

            # Check whitelist
            for trusted in self.domain_whitelist:
                if trusted in domain:
                    return True

            # Additional checks for government and educational domains
            if domain.endswith('.gov') or domain.endswith('.edu'):
                return True

            return False

        except Exception:
            return False

    async def _extract_page_content(self, url: str) -> Optional[str]:
        """Extract main content from a web page"""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None

                html_content = await response.text()

                # Use readability to extract main content
                doc = Document(html_content)
                clean_content = doc.summary()

                # Parse with BeautifulSoup to get text
                soup = BeautifulSoup(clean_content, 'html.parser')
                text_content = soup.get_text(strip=True, separator=' ')

                # Limit content length for M2 Mac memory efficiency
                if len(text_content) > 5000:
                    text_content = text_content[:5000] + "..."

                return text_content

        except Exception as e:
            logging.warning(f"Failed to extract content from {url}: {str(e)}")
            return None

    def _classify_content_type(self, url: str) -> str:
        """Classify content type based on URL"""
        domain = urlparse(url).netloc.lower()

        if 'wikipedia' in domain:
            return 'encyclopedia'
        elif 'github' in domain:
            return 'code'
        elif 'arxiv' in domain:
            return 'academic'
        elif '.gov' in domain:
            return 'government'
        elif '.edu' in domain:
            return 'academic'
        elif 'stackoverflow' in domain:
            return 'technical'
        else:
            return 'general'

    def _calculate_relevance_score(self, query: str, title: str, content: str) -> float:
        """Calculate relevance score for search result"""
        score = 0.0
        query_terms = query.lower().split()

        title_lower = title.lower()
        content_lower = content.lower()

        # Title matches (higher weight)
        for term in query_terms:
            if term in title_lower:
                score += 2.0

        # Content matches
        for term in query_terms:
            content_count = content_lower.count(term)
            score += min(content_count * 0.5, 3.0)  # Cap contribution per term

        # Normalize score
        return min(score / len(query_terms), 10.0)

# Integration with Web Retriever component
class WebRetriever:
    def __init__(self):
        self.searxng_url = os.getenv("SEARXNG_URL", "http://searxng:8080")

    async def get_real_time_info(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Get real-time information using SearxNG"""
        async with SearxNGWebRetriever(self.searxng_url) as searcher:
            results = await searcher.search_current_info(query, max_results)

            # Format results for RAG pipeline
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "type": "web",
                    "text": result['full_content'] or result['summary'],
                    "source": result['url'],
                    "title": result['title'],
                    "domain": result['domain'],
                    "content_type": result['content_type'],
                    "score": result['relevance_score'],
                    "metadata": {
                        "crawled_at": result['crawled_at'],
                        "domain": result['domain']
                    }
                })

            return formatted_results

# Usage example in API Gateway
async def search_web_content(query: str) -> List[Dict[str, Any]]:
    """Search web content using SearxNG"""
    web_retriever = WebRetriever()
    return await web_retriever.get_real_time_info(query)

# Health check for SearxNG
async def check_searxng_health() -> bool:
    """Check if SearxNG service is healthy"""
    try:
        searxng_url = os.getenv("SEARXNG_URL", "http://searxng:8080")
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{searxng_url}/healthz", timeout=5) as response:
                return response.status == 200
    except Exception:
        return False
```

### 8.5 Ollama Setup for M2 Mac

#### setup_ollama_m2.sh:

**📦 Component: Ollama Container Setup Script**

```bash
#!/bin/bash
# setup_ollama_m2.sh - M2 Mac optimized Ollama setup
# Deploy to: Ollama Container initialization

echo "Setting up Ollama for MacBook M2..."

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
sleep 15

# Function to check if Ollama is ready
check_ollama() {
    curl -s http://localhost:11434/api/tags > /dev/null
    return $?
}

# Wait for Ollama to be responsive
until check_ollama; do
    echo "Waiting for Ollama to be ready..."
    sleep 5
done

echo "Ollama is ready! Pulling models optimized for M2 Mac..."

# Pull lightweight models optimized for M2 Mac (16GB RAM)
echo "Pulling Llama2 7B (recommended for M2 Mac)..."
docker exec fizen-rag-ollama-1 ollama pull llama2:7b

echo "Pulling Mistral 7B (alternative lightweight model)..."
docker exec fizen-rag-ollama-1 ollama pull mistral:7b

echo "Pulling CodeLlama 7B for SQL generation..."
docker exec fizen-rag-ollama-1 ollama pull codellama:7b

# Optional: Pull even smaller models if memory is constrained
echo "Pulling Phi-2 (very lightweight for testing)..."
docker exec fizen-rag-ollama-1 ollama pull phi

echo "Available models:"
docker exec fizen-rag-ollama-1 ollama list

echo "Ollama setup complete for M2 Mac!"
echo "Recommended models for M2 Mac (16GB RAM):"
echo "- llama2:7b (primary choice)"
echo "- mistral:7b (alternative)"
echo "- phi (lightweight for testing)"
echo ""
echo "Memory usage per model (approximate):"
echo "- llama2:7b: ~4GB"
echo "- mistral:7b: ~4GB"
echo "- codellama:7b: ~4GB"
echo "- phi: ~1.5GB"
```

### 8.6 M2 Mac Deployment Instructions

**📦 Component: Host System Setup (MacBook M2)**

#### Step-by-Step Setup:

**📦 Component: All Container Services Orchestration**

```bash
# deploy_fizen_rag_m2.sh - Complete deployment script for M2 Mac
# Deploy to: MacBook M2 Host System (coordinates all containers)

# 1. Clone and setup project
git clone https://github.com/your-org/fizen-rag.git
cd fizen-rag

# 2. Create directory structure
mkdir -p vector_service api_gateway frontend
mkdir -p data/faiss data/uploads

# 3. Copy configuration files
cp docker-compose.yml .
cp setup_ollama_m2.sh .
cp setup_searxng.sh .
cp searxng_settings.yml .
chmod +x setup_ollama_m2.sh setup_searxng.sh

# 4. Build and start services (M2 optimized)
echo "🚀 Starting Fizen RAG on MacBook M2..."
docker-compose up --build -d

# 5. Wait for services to be healthy
echo "⏳ Waiting for services to start..."
sleep 30

# 6. Setup SearxNG search engine
echo "🔍 Setting up SearxNG..."
./setup_searxng.sh

# 7. Setup Ollama models for M2 Mac
echo "🤖 Setting up Ollama LLM..."
./setup_ollama_m2.sh

# 8. Verify all services are running
echo "📊 Checking service status..."
docker-compose ps

# 9. Check health endpoints
echo "🏥 Checking service health..."
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8080/healthz

# 10. Test SearxNG integration
echo "🔍 Testing SearxNG..."
curl "http://localhost:8080/search?q=test&format=json"

# 11. Test memory usage
echo "💾 Checking memory usage..."
curl http://localhost:8001/memory

echo "✅ Fizen RAG with SearxNG is ready on MacBook M2!"
echo "Frontend: http://localhost:3000"
echo "API: http://localhost:8000"
echo "Vector Service: http://localhost:8001"
echo "SearxNG: http://localhost:8080"
```

#### M2 Mac Specific Optimizations Applied:

**📦 Component: All Container Services**

✅ **ARM64 Architecture**: All Docker images use `platform: linux/arm64`  
✅ **Memory Optimization**: Services limited to fit within 16GB total RAM  
✅ **SearxNG Integration**: Privacy-focused search without API keys  
✅ **MPS Acceleration**: Vector service uses Metal Performance Shaders when available  
✅ **Batch Size Limits**: Smaller processing batches to prevent memory issues  
✅ **FAISS Configuration**: Optimized for M2 with simpler index types  
✅ **Ollama Models**: Lightweight 7B parameter models recommended  
✅ **Resource Allocation**:

- **📦 PostgreSQL Container**: 1GB (optimized config)
- **📦 SearxNG Container**: 512MB (lightweight search)
- **📦 Ollama Container**: 5GB (for LLM inference)
- **📦 Vector Service Container**: 3GB (embeddings + FAISS)
- **📦 Retrieval Orchestrator Container**: 512MB (coordination logic)
- **📦 Web Retriever Container**: 1GB (web crawling + SQL queries)
- **📦 API Gateway Container**: 1GB (context + prompt + LLM + response)
- **📦 Frontend Container**: 512MB

#### Monitoring Commands for M2 Mac:

**📦 Component: Host System Monitoring Scripts**

```bash
# monitor_services_m2.sh - Monitoring script for all containers
# Deploy to: MacBook M2 Host System

# Check Docker resource usage
docker stats

# Monitor specific container services
echo "📦 PostgreSQL Container:"
docker logs --tail=10 fizen-rag-postgres-1

echo "📦 SearxNG Container:"
curl -s http://localhost:8080/healthz && echo "✅ Healthy" || echo "❌ Down"

echo "📦 Ollama Container:"
curl -s http://localhost:11434/api/tags | jq '.models[].name'

echo "📦 Vector Service Container:"
curl -s http://localhost:8001/memory | jq '.'

echo "📦 Retrieval Orchestrator Container:"
curl -s http://localhost:8003/health | jq '.status'

echo "📦 Web Retriever Container:"
curl -s http://localhost:8005/health | jq '.status'

echo "📦 API Gateway Container:"
curl -s http://localhost:8000/health | jq '.status'

echo "📦 Frontend Container:"
curl -s http://localhost:3000 > /dev/null && echo "✅ Running" || echo "❌ Down"

# Test SearxNG search functionality
echo "🔍 Testing SearxNG search..."
curl -s "http://localhost:8080/search?q=artificial+intelligence&format=json" | jq '.results[0]'

# View service logs
echo "📋 Recent logs:"
docker-compose logs --tail=5 vector_service
docker-compose logs --tail=5 ollama
docker-compose logs --tail=5 searxng

# Check FAISS index statistics
echo "📊 FAISS Statistics:"
curl -s http://localhost:8001/health | jq '.faiss'

# Monitor SearxNG performance
echo "⚡ SearxNG Performance:"
curl -s http://localhost:8080/stats 2>/dev/null || echo "Stats not available"
```

#### Troubleshooting for M2 Mac:

**📦 Component: All Container Services Debugging**

```bash
# troubleshoot_m2.sh - Troubleshooting script for M2 Mac deployment
# Deploy to: MacBook M2 Host System

echo "🔧 Fizen RAG M2 Troubleshooting Guide"

# If SearxNG Container is not responding:
echo "📦 SearxNG Container Issues:"
if ! curl -s http://localhost:8080/healthz > /dev/null; then
    echo "❌ SearxNG not responding"
    echo "1. Check SearxNG logs:"
    docker-compose logs searxng
    echo "2. Restart SearxNG service:"
    docker-compose restart searxng
    echo "3. Test SearxNG configuration:"
    curl http://localhost:8080/config
fi

# If Ollama Container is running out of memory:
echo "📦 Ollama Container Memory Issues:"
echo "1. Check current models:"
docker exec fizen-rag-ollama-1 ollama list
echo "2. Use smaller model:"
docker exec fizen-rag-ollama-1 ollama pull phi  # Use lightweight model
echo "3. Check model memory usage:"
docker stats fizen-rag-ollama-1

# If Vector Service Container has memory issues:
echo "📦 Vector Service Container Memory Issues:"
MEMORY_MB=$(curl -s http://localhost:8001/memory | jq -r '.estimated_memory_mb // "unknown"')
echo "Current FAISS memory usage: ${MEMORY_MB}MB"
if [ "$MEMORY_MB" != "unknown" ] && [ $(echo "$MEMORY_MB > 2000" | bc -l) -eq 1 ]; then
    echo "⚠️ High memory usage detected"
    echo "Consider reducing batch sizes or dataset size"
fi

# If PostgreSQL Container has connection issues:
echo "📦 PostgreSQL Container Issues:"
if ! docker exec fizen-rag-postgres-1 pg_isready -U fizen_user > /dev/null 2>&1; then
    echo "❌ PostgreSQL not ready"
    echo "1. Check PostgreSQL logs:"
    docker-compose logs postgres
    echo "2. Restart PostgreSQL:"
    docker-compose restart postgres
fi

# If API Gateway Container is failing:
echo "📦 API Gateway Container Issues:"
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ API Gateway not responding"
    echo "1. Check API Gateway logs:"
    docker-compose logs api_gateway
    echo "2. Check dependencies:"
    echo "   - PostgreSQL: $(curl -s http://localhost:5432 > /dev/null && echo "✅" || echo "❌")"
    echo "   - Vector Service: $(curl -s http://localhost:8001/health > /dev/null && echo "✅" || echo "❌")"
    echo "   - Retrieval Orchestrator: $(curl -s http://localhost:8003/health > /dev/null && echo "✅" || echo "❌")"
    echo "   - Ollama: $(curl -s http://localhost:11434/api/tags > /dev/null && echo "✅" || echo "❌")"
fi

# If Retrieval Orchestrator Container is failing:
echo "📦 Retrieval Orchestrator Container Issues:"
if ! curl -s http://localhost:8003/health > /dev/null; then
    echo "❌ Retrieval Orchestrator not responding"
    echo "1. Check Retrieval Orchestrator logs:"
    docker-compose logs retrieval_orchestrator
    echo "2. Check dependencies:"
    echo "   - Vector Service: $(curl -s http://localhost:8001/health > /dev/null && echo "✅" || echo "❌")"
    echo "   - Web Retriever: $(curl -s http://localhost:8005/health > /dev/null && echo "✅" || echo "❌")"
    echo "   - PostgreSQL: $(curl -s http://localhost:5432 > /dev/null && echo "✅" || echo "❌")"
fi

# If Web Retriever Container is failing:
echo "📦 Web Retriever Container Issues:"
if ! curl -s http://localhost:8005/health > /dev/null; then
    echo "❌ Web Retriever not responding"
    echo "1. Check Web Retriever logs:"
    docker-compose logs web_retriever
    echo "2. Check dependencies:"
    echo "   - PostgreSQL: $(curl -s http://localhost:5432 > /dev/null && echo "✅" || echo "❌")"
    echo "   - SearxNG: $(curl -s http://localhost:8080/healthz > /dev/null && echo "✅" || echo "❌")"
    echo "   - Ollama: $(curl -s http://localhost:11434/api/tags > /dev/null && echo "✅" || echo "❌")"
fi

# If system is running out of memory:
echo "💾 System Memory Check:"
TOTAL_MEM_GB=$(echo "$(vm_stat | head -1 | awk '{print $3}') * $(vm_stat | grep 'page size' | awk '{print $8}') / 1024^3" | bc -l)
echo "Total System Memory: ${TOTAL_MEM_GB}GB"
echo "Docker Memory Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}"

# General recovery steps:
echo "🔄 General Recovery Steps:"
echo "1. Clear Docker system:"
echo "   docker system prune -a --volumes"
echo "2. Restart with memory cleanup:"
echo "   docker-compose down && docker-compose up --build -d"
echo "3. Check Docker Desktop resource limits:"
echo "   Ensure 12GB memory allocated to Docker"

# Performance optimization:
echo "⚡ Performance Optimization:"
echo "1. Check if MPS is being used by Vector Service:"
curl -s http://localhost:8001/health | jq '.torch_mps_available'
echo "2. Monitor container resource usage:"
echo "   watch docker stats"
echo "3. Check FAISS index size:"
curl -s http://localhost:8001/health | jq '.faiss'
```

## 9. Performance Optimization for MacBook M2

### 9.1 M2-Specific FAISS Optimization

**📦 Component: Vector Service Container Performance Tuning**

```python
# M2ProductionFAISSStore.py - Production FAISS configuration optimized for M2 Mac
# Deploy to: Vector Service Container (performance enhancement)

class M2ProductionFAISSStore(M2OptimizedFAISSStore):
    """
    Production-ready FAISS store optimized for M2 Mac
    Enhanced version of M2OptimizedFAISSStore for better performance
    Used by: Vector Service Container
    """
    def __init__(self, dimension=384):
        # Use the most memory-efficient configuration for M2
        super().__init__(dimension, index_type="Flat")

    def _init_document_index(self):
        """Initialize production FAISS index for M2 Mac"""
        dataset_size = os.getenv("EXPECTED_DATASET_SIZE", "small")

        if dataset_size == "large":  # >50k documents
            # Use quantized index to save memory
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.document_index = faiss.IndexIVFPQ(
                quantizer, self.dimension,
                nlist=25,   # Reduced clusters for M2
                m=8,        # Subquantizers
                nbits=8     # Bits per subquantizer
            )
        elif dataset_size == "medium":  # 10k-50k documents
            # Use IVF with fewer clusters
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.document_index = faiss.IndexIVFFlat(quantizer, self.dimension, 25)
            self.document_index.nprobe = 3  # Conservative search
        else:  # <10k documents (recommended for M2)
            # Use simple flat index - most accurate and memory predictable
            self.document_index = faiss.IndexFlatIP(self.dimension)

    def optimize_for_m2(self):
        """Apply M2-specific optimizations"""
        # Set OMP threads for optimal M2 performance
        faiss.omp_set_num_threads(2)

        # Configure for M2's memory bandwidth
        if hasattr(self.document_index, 'nprobe'):
            self.document_index.nprobe = min(3, self.document_index.nprobe)
```

### 9.2 M2 Mac Docker Performance Tuning with SearxNG

**📦 Component: All Container Services Performance Configuration**

```yaml
# performance-docker-compose.yml - Enhanced docker-compose.yml for M2 Mac performance with SearxNG
# Deploy to: All Container Services (performance optimized)

version: "3.8"
services:
  vector_service:
    # 📦 Vector Service Container Performance Tuning
    # ... existing config
    environment:
      # M2-specific optimizations
      - OMP_NUM_THREADS=2
      - MKL_NUM_THREADS=2
      - OPENBLAS_NUM_THREADS=2
      - PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 # Prevent MPS memory fragmentation
      - FAISS_EXPECTED_DATASET_SIZE=small # Optimize for small datasets
      - EMBEDDING_BATCH_SIZE=4 # Very conservative batching
      - MAX_CHUNK_SIZE=500 # Smaller chunks
      - TOKENIZER_PARALLELISM=false # Disable tokenizer parallelism
      - SEARXNG_URL=http://searxng:8080 # SearxNG integration
    ulimits:
      memlock:
        soft: -1
        hard: -1

  searxng:
    # 📦 SearxNG Container Performance Tuning
    # ... existing config
    environment:
      - SEARXNG_BASE_URL=http://localhost:8080/
      - SEARXNG_SECRET_KEY=fizen-rag-secret-key
      # Performance optimizations for M2
      - SEARXNG_REDIS_URL= # Disable Redis for simplicity
      - SEARXNG_LIMITER=false # Disable rate limiting
      - SEARXNG_IMAGE_PROXY=false # Disable image proxy to save memory
    # Conservative memory allocation
    mem_limit: 512m
    mem_reservation: 256m

  ollama:
    # 📦 Ollama Container Performance Tuning
    # ... existing config
    environment:
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_NUM_PARALLEL=1 # Single model at a time
      - OLLAMA_MAX_LOADED_MODELS=1 # Prevent multiple models in memory
      - OLLAMA_FLASH_ATTENTION=1 # Enable efficient attention
      - OLLAMA_LOW_VRAM=1 # Conservative memory usage
      - OLLAMA_CPU_THREADS=4 # Use 4 cores for inference
    # Reduced memory allocation to accommodate SearxNG
    mem_limit: 5g # Reduced from 6GB
    mem_reservation: 2g

  postgres:
    # 📦 PostgreSQL Container Performance Tuning
    # ... existing config
    environment:
      # M2-specific PostgreSQL optimizations
      - POSTGRES_SHARED_PRELOAD_LIBRARIES=
      - POSTGRES_MAX_CONNECTIONS=50
      - POSTGRES_SHARED_BUFFERS=128MB
      - POSTGRES_EFFECTIVE_CACHE_SIZE=256MB
    command: |
      postgres 
      -c shared_buffers=128MB 
      -c max_connections=50
      -c effective_cache_size=256MB
      -c maintenance_work_mem=32MB
      -c checkpoint_completion_target=0.9
      -c wal_buffers=8MB
      -c default_statistics_target=50
      -c random_page_cost=1.1
      -c effective_io_concurrency=200
      -c work_mem=4MB
      -c max_worker_processes=4
```

### 9.3 M2 Mac System Optimization

**📦 Component: Host System Optimization Scripts**

```bash
#!/bin/bash
# optimize_m2_for_rag.sh - System optimizations for RAG on M2 Mac
# Deploy to: MacBook M2 Host System

echo "Optimizing MacBook M2 for Fizen RAG deployment..."

# 1. Increase Docker memory limit (if not already done)
echo "Please ensure Docker Desktop is configured with:"
echo "- Memory: 12GB"
echo "- CPU: 6 cores"
echo "- Disk image size: 100GB+"

# 2. Set macOS memory management for better Docker performance
sudo sysctl -w vm.swappiness=1
sudo sysctl -w vm.max_map_count=262144

# 3. Disable unnecessary background processes
echo "Consider disabling these for better performance:"
echo "- Spotlight indexing on project directory"
echo "- Time Machine during heavy processing"
echo "- Browser sync and cloud services"

# 4. Set energy preferences for M2 Mac
sudo pmset -a sleep 0
sudo pmset -a disksleep 0
sudo pmset -a displaysleep 10

# 5. Configure kernel parameters for containers
echo "kern.maxfiles=65536" | sudo tee -a /etc/sysctl.conf
echo "kern.maxfilesperproc=32768" | sudo tee -a /etc/sysctl.conf

# 6. Optimize Docker for M2
echo "Docker optimizations:"
echo "1. Enable VirtioFS for faster file sharing"
echo "2. Disable Docker Scout (saves resources)"
echo "3. Enable Resource Saver mode when not in use"

# 7. Monitor resources
echo "Monitoring commands:"
echo "- Activity Monitor -> Memory tab"
echo "- docker stats"
echo "- curl http://localhost:8001/memory"

echo "M2 Mac optimization complete!"
```

### 9.4 Deployment Architecture Summary (M2 Mac + SearxNG)

**📦 Component: Complete System Architecture Overview**

```
MacBook M2 (16GB RAM) Resource Allocation:
┌─────────────────────────────────────────────────────────────┐
│                     System Resources                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   macOS     │  │   Docker    │  │   Fizen     │        │
│  │   4GB       │  │   Engine    │  │   RAG       │        │
│  │   Reserved  │  │   0.5GB     │  │   11.5GB    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘

Docker Services (ARM64):
┌─────────────────────────────────────────────────────────────┐
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │📦 Frontend      │  │📦 API Gateway   │                  │
│  │  Container      │  │  Container      │                  │
│  │  (Port 3000)    │  │  (Port 8000)    │                  │
│  │  512MB RAM      │  │  1GB RAM        │                  │
│  │  0.5 CPU        │  │  1 CPU          │                  │
│  └─────────────────┘  └─────────────────┘                  │
│           │                     │                          │
│           └─────────────────────┼──────────────────────────┤
│                                 ▼                          │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │📦 Vector Service│  │📦 PostgreSQL    │                  │
│  │  Container      │  │  Container      │                  │
│  │  (Port 8001)    │  │  (Port 5432)    │                  │
│  │  3GB RAM        │  │  1GB RAM        │                  │
│  │  2 CPU + MPS    │  │  1 CPU          │                  │
│  │                 │  │                 │                  │
│  │ • BGE-Small     │  │ • Metadata      │                  │
│  │ • FAISS Index   │  │ • Chat History  │                  │
│  │ • ARM64 Opt     │  │ • Job Queue     │                  │
│  └─────────────────┘  └─────────────────┘                  │
│           │                     │                          │
│           ▼                     ▼                          │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │📦 Ollama        │  │📦 SearxNG       │                  │
│  │  Container      │  │  Container      │                  │
│  │  (Port 11434)   │  │  (Port 8080)    │                  │
│  │  5GB RAM        │  │  512MB RAM      │                  │
│  │  4 CPU          │  │  1 CPU          │                  │
│  │                 │  │                 │                  │
│  │ • Llama2 7B     │  │ • Meta Search   │                  │
│  │ • Mistral 7B    │  │ • Privacy Focus │                  │
│  │ • Local Models  │  │ • No API Keys   │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘

Search Flow with SearxNG:
┌─────────────────────────────────────────────────────────────┐
│  User Query → 📦 API Gateway Container → Web Retriever      │
│                     ↓                                       │
│  📦 SearxNG Container (Port 8080)                           │
│                     ↓                                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ DuckDuckGo  │ │    Bing     │ │   Google    │           │
│  │   Search    │ │   Search    │ │   Search    │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│                     ↓                                       │
│  Aggregated Results → Content Extraction → RAG Pipeline    │
└─────────────────────────────────────────────────────────────┘

File System:
┌─────────────────────────────────────────────────────────────┐
│  Local Storage:                                             │
│  • /data/faiss/     - FAISS indices (📦 Vector Service)    │
│  • /data/uploads/   - Uploaded documents (📦 Vector Service)│
│  • postgres_data/   - PostgreSQL data (📦 PostgreSQL)      │
│  • ollama_data/     - Ollama models (📦 Ollama)            │
│  • searxng_data/    - SearxNG config & cache (📦 SearxNG)  │
└─────────────────────────────────────────────────────────────┘
```

### Benefits of SearxNG Integration:

**📦 Component: SearxNG Container Advantages**

✅ **Privacy-First**: No tracking, no API keys required  
✅ **Multi-Engine**: Aggregates results from multiple search engines  
✅ **Self-Hosted**: Complete control over search functionality  
✅ **Cost-Free**: No API usage costs or rate limits  
✅ **Customizable**: Configure which search engines to use  
✅ **ARM64 Compatible**: Runs natively on M2 Mac  
✅ **Lightweight**: Only 512MB RAM footprint  
✅ **Open Source**: Transparent and auditable search functionality

## 7. Complete Container Service Summary & Deployment Architecture

**📦 Component: All Container Services Overview**

This comprehensive Fizen RAG specification provides a complete 8-container microservice architecture optimized for MacBook M2 with 16GB RAM. Each container has well-defined responsibilities and clear integration endpoints.

### 7.1 Container Service Deployment Map

**1. 📦 PostgreSQL Container** (`postgres:15-alpine`)

- **Memory Allocation**: 1GB
- **CPU Allocation**: 1 CPU core
- **Port**: 5432
- **Code Components**: Database schema, SQL queries, connection configuration
- **Primary Purpose**: Metadata storage, chat history, job queue management
- **Data Stored**: Document metadata, chunk references, chat sessions, crawl jobs, processing queues
- **Access Pattern**: Direct database connections from Vector Service, Web Retriever, API Gateway

**2. 📦 SearxNG Container** (`searxng/searxng:latest`)

- **Memory Allocation**: 512MB
- **CPU Allocation**: 1 CPU core
- **Port**: 8080
- **Code Components**: Search configuration, engine management, result aggregation
- **Primary Purpose**: Privacy-focused metasearch engine without external API dependencies
- **Data Sources**: DuckDuckGo, Bing, Google, Wikipedia, ArXiv, GitHub
- **Access Pattern**: HTTP requests from Web Retriever for real-time search

**3. 📦 Ollama Container** (`ollama/ollama:latest`)

- **Memory Allocation**: 5GB (largest allocation for LLM models)
- **CPU Allocation**: 4 CPU cores
- **Port**: 11434
- **Code Components**: Model management, inference engine, ARM64 optimizations
- **Primary Purpose**: Local LLM inference (Llama2 7B, Mistral 7B, CodeLlama 7B)
- **Models Supported**: Multiple 7B parameter models optimized for M2 Mac
- **Access Pattern**: HTTP API calls from API Gateway and Web Retriever

**4. 📦 Vector Service Container** (Custom build with FAISS)

- **Memory Allocation**: 3GB
- **CPU Allocation**: 2 CPU cores + MPS acceleration
- **Port**: 8001
- **Code Components**: FAISS vector store, BGE-Small embeddings, memory management
- **Primary Purpose**: Document ingestion, vector search, embedding generation, chat context
- **Technologies**: FAISS (CPU optimized), BGE-Small-EN-v1.5, PyTorch with MPS
- **Access Pattern**: HTTP API calls from API Gateway and Retrieval Orchestrator

**5. 📦 Retrieval Orchestrator Container** (Custom build with FastAPI)

- **Memory Allocation**: 512MB
- **CPU Allocation**: 1 CPU core
- **Port**: 8003
- **Code Components**: Multi-source retrieval coordination, result unification, deduplication algorithms
- **Primary Purpose**: Coordinate searches across vector, SQL, and web sources with intelligent ranking
- **Integration Role**: Central hub for all retrieval operations, implements sophisticated result merging
- **Access Pattern**: HTTP API calls from API Gateway for complex queries

**6. 📦 Web Retriever Container** (Custom build with FastAPI)

- **Memory Allocation**: 1GB
- **CPU Allocation**: 1 CPU core
- **Port**: 8005
- **Code Components**: Web crawling engine, real-time search, SQL retrieval, SearxNG integration
- **Primary Purpose**: External data retrieval, structured query processing, web content management
- **Dual Functionality**: Automated crawling for knowledge base + real-time search for current info
- **Access Pattern**: HTTP API calls from Retrieval Orchestrator for data retrieval

**7. 📦 API Gateway Container** (Custom build with FastAPI)

- **Memory Allocation**: 1GB
- **CPU Allocation**: 1 CPU core
- **Port**: 8000
- **Code Components**: Context manager, prompt builder, LLM orchestrator, response formatter, authentication
- **Primary Purpose**: Request routing, context management, response generation, user session handling
- **Central Role**: Main entry point for all user interactions, coordinates entire query pipeline
- **Access Pattern**: HTTP requests from Frontend, orchestrates calls to all other services

**8. 📦 Frontend Container** (React build)

- **Memory Allocation**: 512MB
- **CPU Allocation**: 0.5 CPU core
- **Port**: 3000
- **Code Components**: React UI components, admin dashboard, chat interface, file upload
- **Primary Purpose**: User interface for both admin management and public chat interactions
- **Responsive Design**: Optimized for desktop and mobile, real-time chat with WebSocket support
- **Access Pattern**: Serves static files, makes HTTP requests to API Gateway

### 7.2 Resource Allocation Summary

**Total Memory Usage: 12.5GB (78% of 16GB M2 Mac)**

```
┌─────────────────────────────────────────────────────────────┐
│                MacBook M2 (16GB) Memory Distribution        │
├─────────────────────────────────────────────────────────────┤
│ macOS System Reserve           │ 3GB    │ ████████         │
│ Docker Engine                  │ 0.5GB  │ ██               │
│ 📦 Ollama Container           │ 5GB    │ ████████████████ │
│ 📦 Vector Service Container   │ 3GB    │ ███████████      │
│ 📦 PostgreSQL Container       │ 1GB    │ ████             │
│ 📦 Web Retriever Container    │ 1GB    │ ████             │
│ 📦 API Gateway Container      │ 1GB    │ ████             │
│ 📦 SearxNG Container          │ 512MB  │ ██               │
│ 📦 Retrieval Orchestrator     │ 512MB  │ ██               │
│ 📦 Frontend Container         │ 512MB  │ ██               │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 Service Communication Architecture

**Request Flow Pattern**:

```
📱 User Request
    ↓
📦 Frontend Container (Port 3000)
    ↓ HTTP/WebSocket
📦 API Gateway Container (Port 8000)
    ↓ Context Enhancement
📦 Vector Service Container (Port 8001) - Chat History
    ↓ Multi-Source Retrieval
📦 Retrieval Orchestrator Container (Port 8003)
    ↓ ↓ ↓ Parallel Calls
📦 Vector Service    📦 Web Retriever    📦 PostgreSQL
   (Documents)         (Web + SQL)        (Direct Query)
    ↓ ↓ ↓ Results Unification
📦 Retrieval Orchestrator Container
    ↓ Unified Results
📦 API Gateway Container
    ↓ Prompt Building → LLM Generation → Response Formatting
📦 Ollama Container (Port 11434)
    ↓ Final Response
📱 User Interface
```

### 7.4 Container Dependencies

**Startup Order**:

1. **📦 PostgreSQL Container** - Base data layer (no dependencies)
2. **📦 SearxNG Container** - Search engine (no dependencies)
3. **📦 Ollama Container** - LLM engine (no dependencies)
4. **📦 Vector Service Container** - Requires PostgreSQL + optional SearxNG
5. **📦 Web Retriever Container** - Requires PostgreSQL + SearxNG + Ollama
6. **📦 Retrieval Orchestrator Container** - Requires Vector Service + Web Retriever
7. **📦 API Gateway Container** - Requires all above services
8. **📦 Frontend Container** - Requires API Gateway

**Health Check Dependencies**:

- All containers implement `/health` endpoints for monitoring
- Docker Compose uses health checks to ensure proper startup order
- Services wait for dependencies to be healthy before starting

### 7.5 Key Benefits of This Container Architecture

✅ **Clear Separation of Concerns**: Each service has well-defined responsibilities  
✅ **Independent Scalability**: Individual containers can be scaled based on load  
✅ **Easy Development & Debugging**: Services can be developed and tested independently  
✅ **Production Ready**: Proper health checks, logging, and error handling  
✅ **Resource Optimized**: Memory allocation tuned specifically for M2 Mac constraints  
✅ **Privacy-Focused**: No external API dependencies, SearxNG provides private search  
✅ **Cost-Effective**: Zero ongoing API costs for search or LLM functionality  
✅ **ARM64 Optimized**: All components tuned for Apple Silicon performance  
✅ **Microservice Architecture**: True service separation with dedicated orchestration  
✅ **Fault Tolerant**: Individual service failures don't crash entire system

### 7.6 Deployment Readiness

This architecture provides:

- **Complete Implementation Blueprint**: Every service maps to actual Docker containers
- **Consistent API Contracts**: All inter-service calls use documented endpoints
- **Resource Constraints**: Memory and CPU allocations fit within M2 Mac limits
- **Development Workflow**: Clear separation allows parallel development of services
- **Production Deployment**: Can be easily adapted for cloud deployment with container orchestration
- **Monitoring & Observability**: Health checks and metrics endpoints for all services
- **Security Boundaries**: Proper network isolation between containers
- **Data Persistence**: Proper volume mounting for stateful services

Vector Service: 8003:8003
Retrieval Orchestrator: 8002:8002
Web Retriever: 8006:8006
API Service: 8000:8000
LLM Orchestrator: 8008:8008
Context Manager: 8001:8001
Response Formatter: 8009:8009

# Small models that work with 3-4GB memory:

# 1. Phi-3 Mini (3.8B parameters, ~2.3GB)
docker exec -it ollama ollama pull phi3:mini

# 2. Gemma 2B (2B parameters, ~1.4GB) 
docker exec -it ollama ollama pull gemma:2b

# 3. Qwen2 0.5B (500M parameters, ~0.4GB)
docker exec -it ollama ollama pull qwen2:0.5b

# 4. TinyLlama (1.1B parameters, ~0.6GB)
docker exec -it ollama ollama pull tinyllama

# 5. CodeGemma 2B for coding tasks (~1.4GB)
docker exec -it ollama ollama pull codegemma:2b

# Test a small model:
docker exec -it ollama ollama run phi3:mini "Hello, how are you?"
