# Fizen RAG Deep Technical Specification - Upgraded (FAISS + Ollama)

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
5. Store in PostgreSQL (metadata) and FAISS index (vectors).

**For Query Retrieval Mode:**

1. Compute query embedding with BGE-small.
2. Perform vector search using FAISS index across document chunks.
3. Apply confidence threshold (e.g., score > 0.75).
4. Attach metadata from PostgreSQL.
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
    "relevance_score": 0.92
  }
]
```

### 4.6 Prompt Builder

**Purpose**:
Build an optimal LLM prompt that blends query + top evidence + instructions.

**Internal Logic:**

1. Template includes user query, context summary, top evidence chunks with sources
2. Chunks are concatenated in order of relevance
3. Adds formatting instructions (e.g., JSON answer, citations)
4. Applies truncation to stay under token limit

**Integration Endpoints (Calls to Other Services)**:

- None (internal processing only)

**Exposed Endpoints**:

- `POST /prompt/build` - Build optimized prompt from query and evidence
- `POST /prompt/template` - Apply custom prompt templates
- `GET /prompt/health` - Service health check

**Input**:

```json
{
  "query": "How many items were returned?",
  "evidence": [
    { "type": "sql", "data": [{ "month": "Jan", "returns": 130 }] },
    { "type": "vector", "text": "Return period: 30 days" }
  ]
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

## Integration Endpoint Summary Table

| Component                      | Exposes Endpoints                                                                                              | Calls External Services                                                                             |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Context Manager**            | `/context/enhance`, `/context/session/{id}`                                                                    | Vector Service (`/chat/context`)                                                                    |
| **Retrieval Orchestrator**     | `/orchestrate/retrieve`, `/orchestrate/sources`                                                                | Vector Service (`/search/vector`), SQL Retriever (`/sql/query`), Web Retriever (`/search/realtime`) |
| **Vector Storage & Retrieval** | `/search/vector`, `/chat/context`, `/ingest/document`, `/ingest/web`, `/embed`, `/health`, `/memory`, `/stats` | PostgreSQL (direct), BGE-Small (local)                                                              |
| **SQL Retriever**              | `/sql/query`, `/sql/schema`, `/sql/validate`, `/sql/health`                                                    | Ollama (`/api/generate`), PostgreSQL (direct)                                                       |
| **Web Retriever**              | `/search/realtime`, `/crawl/start`, `/crawl/status/{id}`, `/crawl/jobs`, `/crawl/stop/{id}`, `/search/health`  | SearxNG (`/search`), Vector Service (`/ingest/web`), Ollama (`/api/generate`)                       |
| **Prompt Builder**             | `/prompt/build`, `/prompt/template`, `/prompt/health`                                                          | None                                                                                                |
| **LLM Orchestrator**           | `/llm/generate`, `/llm/chat`, `/llm/models`, `/llm/model/switch`, `/llm/health`                                | Ollama (`/api/generate`, `/api/tags`, `/api/pull`)                                                  |
| **Response Formatter**         | `/format/response`, `/format/markdown`, `/format/json`, `/format/health`                                       | None                                                                                                |

## Service-to-Service Communication Flow

```
Query Flow:
API Gateway → Context Manager → Retrieval Orchestrator
                                        ↓
                    ┌─────────────────────┼─────────────────────┐
                    ↓                     ↓                     ↓
           Vector Service        SQL Retriever        Web Retriever
                    ↓                     ↓                     ↓
              (FAISS + BGE)         (PostgreSQL +         (SearxNG +
                                     Ollama)              Vector Service)
                    ↓                     ↓                     ↓
                    └─────────────────────┼─────────────────────┘
                                        ↓
                              Prompt Builder → LLM Orchestrator → Response Formatter
                                                     ↓
                                               Ollama Service
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
       - Original relevance scores
       - Content recency (web content weighted higher)
       - Source authority (internal docs > web content)
       - User context relevance
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
```

### 7.2 FAISS Configuration (M2 Mac Optimized)

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

### 7.3 Sample Queries

**📦 Component: API Gateway & Vector Service Containers**

```sql
-- sample_queries.sql - Database queries used by various services
-- Used by: API Gateway and Vector Service containers

-- Get document chunk metadata by chunk_id (after FAISS search)
-- Used by: Vector Service Container
SELECT
    chunk_text,
    doc_id,
    page_number,
    content_type,
    source_url
FROM document_chunks
WHERE chunk_id = ANY($1::text[]);

-- Get chat message metadata by message IDs (after FAISS search)
-- Used by: Vector Service Container
SELECT
    message_text,
    created_at,
    message_type,
    session_id
FROM chat_messages
WHERE id = ANY($1::integer[]);

-- Get crawl job statistics
-- Used by: API Gateway Container (Admin endpoints)
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
-- Used by: API Gateway Container (Analytics endpoints)
SELECT
    content_type,
    COUNT(*) as chunk_count,
    COUNT(DISTINCT COALESCE(doc_id, domain)) as source_count
FROM document_chunks
GROUP BY content_type;

-- Get session activity summary
-- Used by: API Gateway Container (User analytics)
SELECT
    session_type,
    COUNT(*) as session_count,
    COUNT(DISTINCT session_id) as unique_sessions,
    AVG(EXTRACT(EPOCH FROM (last_activity - created_at))/60) as avg_session_duration_minutes
FROM chat_sessions
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY session_type;
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
      - SEARXNG_URL=http://searxng:8080
      # ARM64 optimizations
      - OMP_NUM_THREADS=2
      - MKL_NUM_THREADS=2
      - FAISS_LARGE_DATASET=false
      - EMBEDDING_BATCH_SIZE=8 # Smaller batches for limited memory
    volumes:
      - faiss_data:/app/data/faiss
      - uploaded_files:/app/data/files
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

  api_gateway:
    # 📦 API Gateway Container (Main Backend)
    build:
      context: ./api_gateway
      dockerfile: Dockerfile.m2
    platform: linux/arm64
    ports:
      - "8000:80"
    environment:
      - DATABASE_URL=postgresql://fizen_user:fizen_password@postgres:5432/fizen_rag
      - VECTOR_SERVICE_URL=http://vector_service:80
      - OLLAMA_URL=http://ollama:11434
      - SEARXNG_URL=http://searxng:8080
      # Performance tuning
      - WORKERS=2
      - MAX_REQUESTS=100
    depends_on:
      postgres:
        condition: service_started
      vector_service:
        condition: service_healthy
      ollama:
        condition: service_healthy
      searxng:
        condition: service_healthy
    networks:
      - fizen_rag_network
    # Lightweight API gateway
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

@app.post("/ingest")
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

@app.post("/retrieve")
async def retrieve_chunks(req: Request):
    """Chunk retrieval endpoint for Vector Service Container"""
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

```python
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

```bash
#!/bin/bash
# M2 Mac optimized Ollama setup

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
- **📦 API Gateway Container**: 1GB
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
    echo "   - Ollama: $(curl -s http://localhost:11434/api/tags > /dev/null && echo "✅" || echo "❌")"
    echo "   - SearxNG: $(curl -s http://localhost:8080/healthz > /dev/null && echo "✅" || echo "❌")"
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

### 8.7 Deployment Architecture Summary (M2 Mac + SearxNG)

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

### 9.4 Memory Management and Monitoring

**📦 Component: Vector Service Container Memory Management**

```python
# m2_memory_monitor.py - Enhanced memory monitoring for M2 Mac
# Deploy to: Vector Service Container (memory management)

import psutil
import torch
import gc

class M2MemoryMonitor:
    """
    Memory monitoring and management for M2 Mac
    Used by: Vector Service Container
    """
    def __init__(self):
        self.memory_threshold = 0.85  # 85% memory usage threshold

    def get_system_memory(self):
        """Get system memory statistics"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percentage": memory.percent
        }

    def get_torch_memory(self):
        """Get PyTorch MPS memory usage on M2"""
        if torch.backends.mps.is_available():
            try:
                allocated = torch.mps.current_allocated_memory()
                return {
                    "allocated_mb": round(allocated / (1024**2), 2),
                    "device": "mps"
                }
            except:
                return {"allocated_mb": 0, "device": "mps_unavailable"}
        return {"allocated_mb": 0, "device": "cpu"}

    def cleanup_memory(self):
        """Force memory cleanup on M2"""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def should_cleanup(self):
        """Check if memory cleanup is needed"""
        memory = self.get_system_memory()
        return memory["percentage"] > (self.memory_threshold * 100)

    def auto_cleanup_if_needed(self):
        """Automatically cleanup if memory usage is high"""
        if self.should_cleanup():
            print("High memory usage detected, cleaning up...")
            self.cleanup_memory()
            return True
        return False

# Integration in vector service
monitor = M2MemoryMonitor()

@app.middleware("http")
async def memory_management_middleware(request: Request, call_next):
    """Automatic memory management for M2 - Used by Vector Service Container"""
    response = await call_next(request)

    # Cleanup memory after heavy operations
    if request.url.path in ["/ingest", "/retrieve", "/embed"]:
        monitor.auto_cleanup_if_needed()

    return response
```

### 9.5 Performance Tuning Guidelines for M2 Mac

**📦 Component: All Container Services Configuration Guidelines**

#### Recommended Limits for Different Use Cases:

**Development & Testing:**
**📦 Component: All Containers (Development Mode)**

```yaml
# Small dataset (<1000 documents)
vector_service:
  mem_limit: 2g
  environment:
    - FAISS_EXPECTED_DATASET_SIZE=small
    - EMBEDDING_BATCH_SIZE=4
    - MAX_CONCURRENT_REQUESTS=2

ollama:
  mem_limit: 4g
  environment:
    - OLLAMA_MAX_LOADED_MODELS=1

searxng:
  mem_limit: 256m
  environment:
    - SEARXNG_LIMITER=true

postgres:
  mem_limit: 512m
```

**Production (Small Business):**
**📦 Component: All Containers (Production Mode)**

```yaml
# Medium dataset (1000-10000 documents)
vector_service:
  mem_limit: 4g
  environment:
    - FAISS_EXPECTED_DATASET_SIZE=medium
    - EMBEDDING_BATCH_SIZE=8
    - MAX_CONCURRENT_REQUESTS=5

ollama:
  mem_limit: 6g
  environment:
    - OLLAMA_NUM_PARALLEL=1

searxng:
  mem_limit: 512m
  environment:
    - SEARXNG_LIMITER=false

postgres:
  mem_limit: 1g
```

**High Performance (at memory limit):**
**📦 Component: All Containers (High Performance Mode)**

```yaml
# Large dataset (>10000 documents) - use with caution on 16GB
vector_service:
  mem_limit: 6g
  environment:
    - FAISS_EXPECTED_DATASET_SIZE=large
    - EMBEDDING_BATCH_SIZE=16
    - MAX_CONCURRENT_REQUESTS=3

ollama:
  mem_limit: 8g
  environment:
    - OLLAMA_NUM_PARALLEL=1
    - OLLAMA_CPU_THREADS=6

searxng:
  mem_limit: 512m

postgres:
  mem_limit: 1g
```

#### Performance Monitoring Dashboard for M2 + SearxNG:

**📦 Component: Host System Monitoring (MacBook M2)**

```bash
#!/bin/bash
# monitor_m2_performance.sh - Enhanced with SearxNG monitoring
# Deploy to: MacBook M2 Host System (monitors all containers)

watch -n 5 '
echo "=== M2 Mac RAG + SearxNG Performance Monitor ==="
echo "Time: $(date)"
echo ""
echo "📊 System Memory:"
vm_stat | head -5
echo ""
echo "📦 Docker Containers Status:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
echo ""
echo "🔍 RAG Services Health:"
echo "📦 Vector Service Memory: $(curl -s http://localhost:8001/memory | jq -r ".estimated_memory_mb")MB"
echo "📦 SearxNG Status: $(curl -s http://localhost:8080/healthz > /dev/null && echo "✅ Healthy" || echo "❌ Down")"
echo "📦 Ollama Status: $(curl -s http://localhost:11434/api/tags > /dev/null && echo "✅ Ready" || echo "❌ Down")"
echo "📦 PostgreSQL Status: $(docker exec fizen-rag-postgres-1 pg_isready -U fizen_user > /dev/null 2>&1 && echo "✅ Ready" || echo "❌ Down")"
echo "📦 API Gateway Status: $(curl -s http://localhost:8000/health > /dev/null && echo "✅ Healthy" || echo "❌ Down")"
echo "📦 Frontend Status: $(curl -s http://localhost:3000 > /dev/null && echo "✅ Running" || echo "❌ Down")"
echo ""
echo "⚡ Search Performance:"
echo "📦 SearxNG Response Time: $(curl -s -w "%{time_total}" http://localhost:8080/search?q=test&format=json -o /dev/null)s"
echo ""
echo "🌡️ Temperature (if available):"
sudo powermetrics -n 1 -i 1 --samplers smc 2>/dev/null | grep -i temp | head -3 || echo "Temperature monitoring not available"
'
```

### 9.6 SearxNG Performance Optimization

**📦 Component: SearxNG Container Performance Enhancement**

```python
# searxng_performance_monitor.py - SearxNG performance monitoring for M2 Mac
# Deploy to: API Gateway Container (monitors SearxNG container)

import asyncio
import aiohttp
import time
from typing import Dict, Any

class SearxNGPerformanceMonitor:
    """
    SearxNG performance monitoring for M2 Mac
    Used by: API Gateway Container to monitor SearxNG Container
    """
    def __init__(self, searxng_url: str = "http://searxng:8080"):
        self.searxng_url = searxng_url

    async def health_check(self) -> Dict[str, Any]:
        """Check SearxNG health and performance"""
        try:
            start_time = time.time()

            async with aiohttp.ClientSession() as session:
                # Health check
                async with session.get(f"{self.searxng_url}/healthz") as response:
                    health_status = response.status == 200

                # Performance test
                test_query = "test performance"
                search_params = {'q': test_query, 'format': 'json'}

                async with session.get(f"{self.searxng_url}/search", params=search_params) as response:
                    search_time = time.time() - start_time
                    search_results = await response.json() if response.status == 200 else {}

                return {
                    "healthy": health_status,
                    "response_time_ms": round(search_time * 1000, 2),
                    "search_results_count": len(search_results.get('results', [])),
                    "status": "optimal" if search_time < 2.0 else "slow" if search_time < 5.0 else "degraded"
                }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "status": "error"
            }

    async def get_engine_status(self) -> Dict[str, Any]:
        """Get status of enabled search engines"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.searxng_url}/config") as response:
                    if response.status == 200:
                        config = await response.json()
                        engines = config.get('engines', {})

                        enabled_engines = [name for name, engine in engines.items()
                                         if not engine.get('disabled', False)]

                        return {
                            "total_engines": len(engines),
                            "enabled_engines": len(enabled_engines),
                            "engines_list": enabled_engines[:10]  # Top 10
                        }
        except Exception:
            pass

        return {"total_engines": 0, "enabled_engines": 0, "engines_list": []}

# Integration with existing memory monitoring
class M2MemoryMonitorWithSearxNG(M2MemoryMonitor):
    """
    Enhanced memory monitoring including SearxNG
    Used by: Vector Service Container (enhanced monitoring)
    """
    def __init__(self):
        super().__init__()
        self.searxng_monitor = SearxNGPerformanceMonitor()

    async def get_full_system_status(self):
        """Get comprehensive system status including SearxNG"""
        base_status = {
            "system_memory": self.get_system_memory(),
            "torch_memory": self.get_torch_memory()
        }

        # Add SearxNG status
        searxng_status = await self.searxng_monitor.health_check()
        engine_status = await self.searxng_monitor.get_engine_status()

        return {
            **base_status,
            "searxng": {
                **searxng_status,
                **engine_status
            }
        }

# Container Health Check Aggregator
class ContainerHealthAggregator:
    """
    Aggregates health status from all containers
    Used by: API Gateway Container (system health monitoring)
    """
    def __init__(self):
        self.services = {
            "postgres": "http://postgres:5432",
            "searxng": "http://searxng:8080/healthz",
            "ollama": "http://ollama:11434/api/tags",
            "vector_service": "http://vector_service:80/health"
        }

    async def check_all_services(self) -> Dict[str, Any]:
        """Check health of all container services"""
        status_results = {}

        async with aiohttp.ClientSession() as session:
            for service_name, url in self.services.items():
                try:
                    async with session.get(url, timeout=5) as response:
                        status_results[service_name] = {
                            "status": "healthy" if response.status == 200 else "unhealthy",
                            "response_code": response.status
                        }
                except Exception as e:
                    status_results[service_name] = {
                        "status": "unreachable",
                        "error": str(e)
                    }

        # Calculate overall system health
        healthy_services = sum(1 for s in status_results.values() if s["status"] == "healthy")
        total_services = len(status_results)
        overall_health = "healthy" if healthy_services == total_services else "degraded"

        return {
            "overall_health": overall_health,
            "healthy_services": f"{healthy_services}/{total_services}",
            "services": status_results
        }
```

### 9.7 Complete Container Service Summary

**📦 Component: All Container Services Overview**

This comprehensive update ensures that every code example in the Fizen RAG specification is clearly marked with its corresponding Docker container component. Here's the complete container service breakdown:

#### Container Service Deployment Map:

1. **📦 PostgreSQL Container** (`postgres:15-alpine`)

   - **Code Components**: Database schema, SQL queries, connection configuration
   - **Memory**: 1GB
   - **Purpose**: Metadata storage, chat history, job queue

2. **📦 SearxNG Container** (`searxng/searxng:latest`)

   - **Code Components**: Search configuration, setup scripts, performance monitoring
   - **Memory**: 512MB
   - **Purpose**: Privacy-focused metasearch engine

3. **📦 Ollama Container** (`ollama/ollama:latest`)

   - **Code Components**: Model setup scripts, client implementation, health checks
   - **Memory**: 5GB
   - **Purpose**: Local LLM inference (Llama2, Mistral, etc.)

4. **📦 Vector Service Container** (Custom build with FAISS)

   - **Code Components**: FAISS store, BGE-Small embeddings, memory management
   - **Memory**: 3GB
   - **Purpose**: Document ingestion, vector search, embedding generation

5. **📦 API Gateway Container** (Custom build with FastAPI)

   - **Code Components**: Web retriever, Ollama client, health aggregator, main API
   - **Memory**: 1GB
   - **Purpose**: Request routing, business logic, external integrations

6. **📦 Frontend Container** (React build)
   - **Code Components**: React UI, admin dashboard, user interface
   - **Memory**: 512MB
   - **Purpose**: User interface for both admin and public users

#### Key Benefits of This Container Architecture:

✅ **Clear Separation**: Each service has well-defined responsibilities  
✅ **Easy Maintenance**: Code is clearly mapped to specific containers  
✅ **Scalable Design**: Individual containers can be scaled independently  
✅ **Resource Optimization**: Memory allocation tuned for M2 Mac constraints  
✅ **Development Friendly**: Easy to debug and develop individual components  
✅ **Production Ready**: Proper health checks and monitoring for all services

This completes the comprehensive update of the Fizen RAG specification with clear Docker container component markings for all code examples, optimized specifically for MacBook M2 deployment with 16GB RAM.
