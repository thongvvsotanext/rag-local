# Updated Component Specifications with Clear Integration Endpoints

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
    {"type": "sql", "data": [{"month": "Jan", "returns": 130}]},
    {"type": "vector", "text": "Return period: 30 days"}
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
  "citations": [{ "ref": "[1]", "doc": "sql", "value": "return stats Jan 2024" }]
}
```

## Integration Endpoint Summary Table

| Component | Exposes Endpoints | Calls External Services |
|-----------|------------------|-------------------------|
| **Context Manager** | `/context/enhance`, `/context/session/{id}` | Vector Service (`/chat/context`) |
| **Retrieval Orchestrator** | `/orchestrate/retrieve`, `/orchestrate/sources` | Vector Service (`/search/vector`), SQL Retriever (`/sql/query`), Web Retriever (`/search/realtime`) |
| **Vector Storage & Retrieval** | `/search/vector`, `/chat/context`, `/ingest/document`, `/ingest/web`, `/embed`, `/health`, `/memory`, `/stats` | PostgreSQL (direct), BGE-Small (local) |
| **SQL Retriever** | `/sql/query`, `/sql/schema`, `/sql/validate`, `/sql/health` | Ollama (`/api/generate`), PostgreSQL (direct) |
| **Web Retriever** | `/search/realtime`, `/crawl/start`, `/crawl/status/{id}`, `/crawl/jobs`, `/crawl/stop/{id}`, `/search/health` | SearxNG (`/search`), Vector Service (`/ingest/web`), Ollama (`/api/generate`) |
| **Prompt Builder** | `/prompt/build`, `/prompt/template`, `/prompt/health` | None |
| **LLM Orchestrator** | `/llm/generate`, `/llm/chat`, `/llm/models`, `/llm/model/switch`, `/llm/health` | Ollama (`/api/generate`, `/api/tags`, `/api/pull`) |
| **Response Formatter** | `/format/response`, `/format/markdown`, `/format/json`, `/format/health` | None |

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

This comprehensive endpoint mapping ensures clear service boundaries and makes it easy to understand how components interact with each other through well-defined APIs.