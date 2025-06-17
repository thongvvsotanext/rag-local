from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import logging
import sys
from datetime import datetime
import json
from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.engine import URL
import sqlparse
from contextlib import asynccontextmanager
import asyncio
import aiohttp
import re
import time
import traceback
import uuid

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
file_handler = logging.FileHandler('sql_retriever.log')
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

# Database Query Control Configuration
ENABLE_DB_QUERY = os.getenv("ENABLE_DB_QUERY", "true").lower() == "true"
DB_QUERY_DISABLED_MESSAGE = os.getenv("DB_QUERY_DISABLED_MESSAGE", "Database querying is currently disabled")

# Configuration
# Build database URL from individual environment variables
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_NAME = os.getenv("POSTGRES_DB")

if not all([DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME]):
    raise ValueError("Missing required database configuration. Please set all POSTGRES_* environment variables.")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# LLM Orchestrator Configuration
LLM_ORCHESTRATOR_URL = os.getenv("LLM_ORCHESTRATOR_URL", "http://llm-orchestrator:8008")
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))

# Log configuration on startup
logger.info("SQL_RETRIEVER_STARTUP - SQL Retriever Service starting", extra={
    'config': {
        'database_url': f"postgresql://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        'llm_orchestrator_url': LLM_ORCHESTRATOR_URL,
        'llm_request_timeout': LLM_REQUEST_TIMEOUT
    }
})

# Database setup
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Input/Output Models
class SQLQueryRequest(BaseModel):
    query: str
    max_rows: Optional[int] = 100
    timeout: Optional[int] = 30

class SQLQueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    sql_generated: str
    execution_time: float
    row_count: int
    schema_info: Dict[str, Any]

class SchemaInfo(BaseModel):
    tables: Dict[str, Dict[str, Any]]
    relationships: List[Dict[str, Any]]

# LLM Integration Models
class LLMRequest(BaseModel):
    prompt: str
    parameters: Dict[str, Any]

class LLMResponse(BaseModel):
    response: str
    metadata: Dict[str, Any]

# Helper Functions
def get_schema_info(request_id: str = "N/A") -> Dict[str, Any]:
    """Get database schema information."""
    start_time = time.time()
    
    try:
        log_with_context(
            logger, 'info',
            f"SCHEMA_INFO_START - Getting database schema information",
            request_id, "get_schema_info"
        )
        
        inspector = inspect(engine)
        schema_info = {
            "tables": {},
            "relationships": []
        }
        
        table_names = inspector.get_table_names()
        
        log_with_context(
            logger, 'info',
            f"SCHEMA_INFO_STEP - Found tables in database",
            request_id, "get_schema_info",
            table_count=len(table_names),
            table_names=table_names
        )
        
        # Get table information
        for table_name in table_names:
            try:
                columns = []
                for column in inspector.get_columns(table_name):
                    columns.append({
                        "name": column["name"],
                        "type": str(column["type"]),
                        "nullable": column["nullable"],
                        "default": str(column["default"]) if column["default"] else None
                    })
                
                # Get primary keys
                pk = inspector.get_pk_constraint(table_name)
                primary_keys = pk["constrained_columns"] if pk else []
                
                # Get foreign keys
                foreign_keys = []
                for fk in inspector.get_foreign_keys(table_name):
                    foreign_keys.append({
                        "constrained_columns": fk["constrained_columns"],
                        "referred_table": fk["referred_table"],
                        "referred_columns": fk["referred_columns"]
                    })
                    schema_info["relationships"].append({
                        "from_table": table_name,
                        "to_table": fk["referred_table"],
                        "from_columns": fk["constrained_columns"],
                        "to_columns": fk["referred_columns"]
                    })
                
                schema_info["tables"][table_name] = {
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "foreign_keys": foreign_keys
                }
                
                log_with_context(
                    logger, 'debug',
                    f"SCHEMA_INFO_TABLE - Processed table schema",
                    request_id, "get_schema_info",
                    table_name=table_name,
                    column_count=len(columns),
                    primary_key_count=len(primary_keys),
                    foreign_key_count=len(foreign_keys)
                )
                
            except Exception as e:
                log_with_context(
                    logger, 'warning',
                    f"SCHEMA_INFO_TABLE_ERROR - Error processing table schema",
                    request_id, "get_schema_info",
                    table_name=table_name,
                    error=str(e)
                )
                continue
        
        processing_time = time.time() - start_time
        
        log_with_context(
            logger, 'info',
            f"SCHEMA_INFO_SUCCESS - Schema information retrieved successfully",
            request_id, "get_schema_info",
            total_tables=len(schema_info["tables"]),
            total_relationships=len(schema_info["relationships"]),
            processing_time_seconds=round(processing_time, 3)
        )
        
        return schema_info
        
    except Exception as e:
        processing_time = time.time() - start_time
        log_with_context(
            logger, 'error',
            f"SCHEMA_INFO_ERROR - Failed to get schema information",
            request_id, "get_schema_info",
            error=str(e),
            traceback=traceback.format_exc(),
            processing_time_seconds=round(processing_time, 3)
        )
        raise

def extract_table_names_from_sql(sql: str) -> List[str]:
    """Extract table names from SQL query."""
    try:
        # Parse SQL to extract table names
        parsed = sqlparse.parse(sql)[0]
        table_names = set()
        
        # Simple regex to find table names (basic approach)
        # This regex looks for patterns like "FROM table_name" or "JOIN table_name"
        table_patterns = [
            r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'\bINTO\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        ]
        
        sql_upper = sql.upper()
        for pattern in table_patterns:
            matches = re.findall(pattern, sql_upper, re.IGNORECASE)
            table_names.update(matches)
        
        return list(table_names)
    except Exception as e:
        logger.warning(f"Error extracting table names from SQL: {str(e)}")
        return []

def generate_sql_prompt(natural_query: str, schema_info: Dict[str, Any]) -> str:
    """Generate prompt for SQL generation."""
    schema_str = json.dumps(schema_info, indent=2)
    
    prompt = f"""You are a SQL expert. Convert the following natural language query into a safe, read-only SQL query.
Use the provided database schema to ensure accuracy.

Database Schema:
{schema_str}

Natural Language Query:
{natural_query}

Requirements:
1. Generate only SELECT queries (no INSERT, UPDATE, DELETE)
2. Include proper JOIN conditions based on foreign keys
3. Use appropriate WHERE clauses for filtering
4. Add LIMIT clause to prevent large result sets (default LIMIT 100)
5. Include comments explaining the query logic
6. Only use tables that exist in the provided schema
7. If no relevant tables exist in the schema, return "NO_TABLES_FOUND"
8. Return only the SQL query without any explanations or formatting

SQL Query:"""
    
    return prompt

async def call_llm_orchestrator(prompt: str, request_id: str = "N/A") -> str:
    """Call LLM orchestrator service to generate SQL."""
    start_time = time.time()
    
    try:
        log_with_context(
            logger, 'info',
            f"LLM_ORCHESTRATOR_CALL_START - Calling LLM orchestrator for SQL generation",
            request_id, "call_llm_orchestrator",
            prompt_length=len(prompt),
            llm_orchestrator_url=f"{LLM_ORCHESTRATOR_URL}/generate"
        )
        
        # Prepare request for LLM orchestrator
        llm_request = {
            "prompt": prompt,
            "parameters": {
                "temperature": 0.1,  # Low temperature for more deterministic SQL
                "max_tokens": 1000
            }
        }
        
        # Call LLM orchestrator with retry logic
        for attempt in range(MAX_RETRIES):
            try:
                llm_call_start = time.time()
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{LLM_ORCHESTRATOR_URL}/generate",
                        json=llm_request,
                        timeout=aiohttp.ClientTimeout(total=LLM_REQUEST_TIMEOUT)
                    ) as response:
                        llm_call_time = time.time() - llm_call_start
                        
                        if response.status != 200:
                            error_text = await response.text()
                            log_with_context(
                                logger, 'error',
                                f"LLM_ORCHESTRATOR_ERROR - LLM orchestrator returned error status",
                                request_id, "call_llm_orchestrator",
                                status_code=response.status,
                                error_response=error_text,
                                attempt=attempt + 1,
                                llm_call_time_seconds=round(llm_call_time, 3)
                            )
                            
                            if attempt == MAX_RETRIES - 1:
                                raise HTTPException(
                                    status_code=response.status, 
                                    detail=f"LLM orchestrator error: {error_text}"
                                )
                            
                            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                            continue
                        
                        llm_response = await response.json()
                        
                        processing_time = time.time() - start_time
                        
                        log_with_context(
                            logger, 'info',
                            f"LLM_ORCHESTRATOR_CALL_SUCCESS - LLM orchestrator call completed",
                            request_id, "call_llm_orchestrator",
                            llm_call_time_seconds=round(llm_call_time, 3),
                            total_processing_time_seconds=round(processing_time, 3),
                            attempt=attempt + 1,
                            response_length=len(llm_response.get("response", "")),
                            llm_metadata=llm_response.get("metadata", {})
                        )
                        
                        return llm_response.get("response", "")
                        
            except asyncio.TimeoutError:
                log_with_context(
                    logger, 'warning',
                    f"LLM_ORCHESTRATOR_TIMEOUT - LLM orchestrator call timed out",
                    request_id, "call_llm_orchestrator",
                    attempt=attempt + 1,
                    timeout_seconds=LLM_REQUEST_TIMEOUT
                )
                
                if attempt == MAX_RETRIES - 1:
                    raise HTTPException(
                        status_code=408, 
                        detail=f"LLM orchestrator timeout after {LLM_REQUEST_TIMEOUT} seconds"
                    )
                
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                continue
                
            except Exception as e:
                log_with_context(
                    logger, 'error',
                    f"LLM_ORCHESTRATOR_CALL_ERROR - Error calling LLM orchestrator",
                    request_id, "call_llm_orchestrator",
                    error=str(e),
                    traceback=traceback.format_exc(),
                    attempt=attempt + 1
                )
                
                if attempt == MAX_RETRIES - 1:
                    raise HTTPException(
                        status_code=503, 
                        detail=f"Failed to call LLM orchestrator after {MAX_RETRIES} attempts: {str(e)}"
                    )
                
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                continue
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        log_with_context(
            logger, 'error',
            f"LLM_ORCHESTRATOR_CALL_FATAL_ERROR - Fatal error calling LLM orchestrator",
            request_id, "call_llm_orchestrator",
            error=str(e),
            traceback=traceback.format_exc(),
            processing_time_seconds=round(processing_time, 3)
        )
        raise HTTPException(status_code=500, detail=f"Error calling LLM orchestrator: {str(e)}")

async def generate_sql(natural_query: str, schema_info: Dict[str, Any], request_id: str = "N/A") -> str:
    """Generate SQL from natural language using LLM orchestrator."""
    start_time = time.time()
    
    try:
        log_with_context(
            logger, 'info',
            f"SQL_GENERATION_START - Starting SQL generation from natural language",
            request_id, "generate_sql",
            natural_query=natural_query,
            available_tables=list(schema_info.get("tables", {}).keys())
        )
        
        prompt = generate_sql_prompt(natural_query, schema_info)
        
        log_with_context(
            logger, 'debug',
            f"SQL_GENERATION_STEP - Generated prompt for LLM orchestrator",
            request_id, "generate_sql",
            prompt_length=len(prompt)
        )
        
        # Call LLM orchestrator
        sql_response = await call_llm_orchestrator(prompt, request_id)
        
        # Process the response
        sql = sql_response.strip()
        
        # Check if no tables were found
        if "NO_TABLES_FOUND" in sql:
            log_with_context(
                logger, 'warning',
                f"SQL_GENERATION_NO_TABLES - No relevant tables found for query",
                request_id, "generate_sql",
                natural_query=natural_query,
                available_tables=list(schema_info.get("tables", {}).keys())
            )
            raise HTTPException(status_code=404, detail="No relevant tables found in database for this query")
        
        # Extract SQL from the response (in case model includes explanations)
        sql_match = re.search(r"```sql\n(.*?)\n```", sql, re.DOTALL)
        if sql_match:
            sql = sql_match.group(1)
        
        # Clean up the SQL
        sql = sql.strip()
        if not sql:
            raise ValueError("Empty SQL generated")
        
        processing_time = time.time() - start_time
        
        log_with_context(
            logger, 'info',
            f"SQL_GENERATION_SUCCESS - SQL generation completed",
            request_id, "generate_sql",
            generated_sql=sql,
            processing_time_seconds=round(processing_time, 3)
        )
        
        return sql
    
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        log_with_context(
            logger, 'error',
            f"SQL_GENERATION_ERROR - SQL generation failed",
            request_id, "generate_sql",
            error=str(e),
            traceback=traceback.format_exc(),
            natural_query=natural_query,
            processing_time_seconds=round(processing_time, 3)
        )
        raise HTTPException(status_code=500, detail=f"Error generating SQL: {str(e)}")

def validate_sql(sql: str, schema_info: Dict[str, Any], request_id: str = "N/A") -> bool:
    """Validate SQL query for safety and table existence."""
    start_time = time.time()
    
    try:
        log_with_context(
            logger, 'info',
            f"SQL_VALIDATION_START - Starting SQL validation",
            request_id, "validate_sql",
            sql=sql
        )
        
        # Parse SQL
        parsed = sqlparse.parse(sql)[0]
        
        # Check if it's a SELECT query
        if not parsed.get_type().lower() == "select":
            log_with_context(
                logger, 'warning',
                f"SQL_VALIDATION_ERROR - Not a SELECT query",
                request_id, "validate_sql",
                query_type=parsed.get_type().lower(),
                sql=sql
            )
            return False
        
        # Check for dangerous operations
        dangerous_ops = ["insert", "update", "delete", "drop", "alter", "create", "truncate"]
        sql_lower = sql.lower()
        for op in dangerous_ops:
            if op in sql_lower:
                log_with_context(
                    logger, 'warning',
                    f"SQL_VALIDATION_ERROR - Dangerous operation found",
                    request_id, "validate_sql",
                    dangerous_operation=op,
                    sql=sql
                )
                return False
        
        # Check for proper LIMIT clause
        if "limit" not in sql_lower:
            log_with_context(
                logger, 'warning',
                f"SQL_VALIDATION_ERROR - Missing LIMIT clause",
                request_id, "validate_sql",
                sql=sql
            )
            return False
        
        # Extract and validate table names
        referenced_tables = extract_table_names_from_sql(sql)
        available_tables = set(schema_info.get("tables", {}).keys())
        missing_tables = [table.lower() for table in referenced_tables if table.lower() not in [t.lower() for t in available_tables]]
        
        if missing_tables:
            log_with_context(
                logger, 'warning',
                f"SQL_VALIDATION_ERROR - Referenced tables not found in schema",
                request_id, "validate_sql",
                missing_tables=missing_tables,
                referenced_tables=referenced_tables,
                available_tables=list(available_tables),
                sql=sql
            )
            return False
        
        processing_time = time.time() - start_time
        
        log_with_context(
            logger, 'info',
            f"SQL_VALIDATION_SUCCESS - SQL validation passed",
            request_id, "validate_sql",
            referenced_tables=referenced_tables,
            processing_time_seconds=round(processing_time, 3)
        )
        
        return True
    
    except Exception as e:
        processing_time = time.time() - start_time
        log_with_context(
            logger, 'error',
            f"SQL_VALIDATION_ERROR - SQL validation failed",
            request_id, "validate_sql",
            error=str(e),
            traceback=traceback.format_exc(),
            sql=sql,
            processing_time_seconds=round(processing_time, 3)
        )
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI app."""
    # Startup
    logger.info("SQL_RETRIEVER_STARTUP - Starting up SQL Retriever service...")
    try:
        # Test database connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("SQL_RETRIEVER_STARTUP - Database connection successful")
        
        # Test LLM orchestrator connection
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{LLM_ORCHESTRATOR_URL}/health") as response:
                if response.status != 200:
                    raise Exception(f"LLM orchestrator not healthy: {response.status}")
        logger.info("SQL_RETRIEVER_STARTUP - LLM orchestrator connection successful")
        
        # Load schema metadata
        metadata.reflect(bind=engine)
        logger.info("SQL_RETRIEVER_STARTUP - Database schema loaded successfully", extra={
            'table_count': len(metadata.tables)
        })
        
    except Exception as e:
        logger.error(f"SQL_RETRIEVER_STARTUP - Startup failed: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("SQL_RETRIEVER_SHUTDOWN - Shutting down SQL Retriever service...")
    try:
        engine.dispose()
        logger.info("SQL_RETRIEVER_SHUTDOWN - Database connection disposed")
    except Exception as e:
        logger.error(f"SQL_RETRIEVER_SHUTDOWN - Shutdown error: {str(e)}")

app = FastAPI(title="SQL Retriever Service", lifespan=lifespan)

@app.post("/query", response_model=SQLQueryResponse)
async def execute_query(request: SQLQueryRequest):
    """Execute a natural language query and return results."""
    request_id = str(uuid.uuid4())
    endpoint_logger = get_contextual_logger(request_id, "/query")
    start_time = time.time()
    
    try:
        # Log input parameters
        log_with_context(
            logger, 'info',
            f"SQL_QUERY_START - Starting SQL query execution",
            request_id, "/query",
            input_params={
                'natural_query': request.query,
                'max_rows': request.max_rows,
                'timeout': request.timeout,
                'query_length': len(request.query)
            },
            enable_db_query=ENABLE_DB_QUERY
        )
        # Check if database querying is enabled
        if not ENABLE_DB_QUERY:
            total_time = time.time() - start_time
            
            log_with_context(
                logger, 'info',
                f"SQL_QUERY_DISABLED - Database querying is disabled, returning empty results",
                request_id, "/query",
                natural_query=request.query,
                disabled_message=DB_QUERY_DISABLED_MESSAGE,
                total_time_seconds=round(total_time, 3)
            )
            
            # Return empty results with metadata indicating disabled state
            return SQLQueryResponse(
                results=[],  # ✅ Empty array as requested
                sql_generated="-- Database querying is disabled",
                execution_time=0.0,
                row_count=0,
                schema_info={
                    "tables": {},
                    "relationships": [],
                    "disabled_reason": DB_QUERY_DISABLED_MESSAGE
                }
            )
        
        # Get schema information
        endpoint_logger.info("SQL_QUERY_STEP - Getting schema information")
        schema_start_time = time.time()
        schema_info = get_schema_info(request_id)
        schema_time = time.time() - schema_start_time
        
        available_tables = list(schema_info.get("tables", {}).keys())
        if not available_tables:
            log_with_context(
                logger, 'warning',
                f"SQL_QUERY_NO_TABLES - No tables found in database",
                request_id, "/query",
                database_name=DB_NAME
            )
            raise HTTPException(status_code=404, detail="No tables found in database")
        
        endpoint_logger.info(f"SQL_QUERY_STEP - Schema information retrieved", extra={
            'schema_time_seconds': round(schema_time, 3),
            'available_tables': available_tables,
            'table_count': len(available_tables)
        })
        
        # Generate SQL from natural language
        endpoint_logger.info("SQL_QUERY_STEP - Generating SQL from natural language")
        sql_gen_start_time = time.time()
        sql = await generate_sql(request.query, schema_info, request_id)
        sql_gen_time = time.time() - sql_gen_start_time
        
        # Validate SQL
        endpoint_logger.info("SQL_QUERY_STEP - Validating generated SQL")
        validation_start_time = time.time()
        if not validate_sql(sql, schema_info, request_id):
            log_with_context(
                logger, 'error',
                f"SQL_QUERY_VALIDATION_ERROR - Generated SQL query is not safe or references non-existent tables",
                request_id, "/query",
                generated_sql=sql,
                available_tables=available_tables
            )
            raise HTTPException(status_code=400, detail="Generated SQL query is not safe or references non-existent tables")
        validation_time = time.time() - validation_start_time
        
        endpoint_logger.info(f"SQL_QUERY_STEP - SQL validation passed", extra={
            'validation_time_seconds': round(validation_time, 3)
        })
        
        # Execute query with timeout
        endpoint_logger.info("SQL_QUERY_STEP - Executing SQL query")
        execution_start_time = time.time()
        
        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text(sql),
                    execution_options={"timeout": request.timeout}
                )
                
                # Fetch results
                columns = result.keys()
                rows = result.fetchmany(request.max_rows)
                
                # Convert to list of dicts
                results = [dict(zip(columns, row)) for row in rows]
            
            execution_time = time.time() - execution_start_time
            
            endpoint_logger.info(f"SQL_QUERY_STEP - Query execution completed", extra={
                'execution_time_seconds': round(execution_time, 3),
                'rows_returned': len(results),
                'columns_returned': len(columns) if columns else 0
            })
            
        except Exception as e:
            execution_time = time.time() - execution_start_time
            log_with_context(
                logger, 'error',
                f"SQL_QUERY_EXECUTION_ERROR - Query execution failed",
                request_id, "/query",
                error=str(e),
                traceback=traceback.format_exc(),
                generated_sql=sql,
                execution_time_seconds=round(execution_time, 3),
                
            )
            raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")
        
        total_time = time.time() - start_time
        
        # Prepare response
        response = SQLQueryResponse(
            results=results,
            sql_generated=sql,
            execution_time=execution_time,
            row_count=len(results),
            schema_info=schema_info
        )
        
        # Log successful completion
        log_with_context(
            logger, 'info',
            f"SQL_QUERY_SUCCESS - SQL query completed successfully",
            request_id, "/query",
            natural_query=request.query,
            generated_sql=sql,
            rows_returned=len(results),
            timing={
                'total_time_seconds': round(total_time, 3),
                'schema_time_seconds': round(schema_time, 3),
                'sql_generation_time_seconds': round(sql_gen_time, 3),
                'validation_time_seconds': round(validation_time, 3),
                'execution_time_seconds': round(execution_time, 3)
            },
            response_summary={
                'row_count': response.row_count,
                'execution_time': response.execution_time,
                'schema_tables_used': extract_table_names_from_sql(sql)
            }
        )
        
        return response
    
    except HTTPException as he:
        total_time = time.time() - start_time
        log_with_context(
            logger, 'error',
            f"SQL_QUERY_HTTP_ERROR - HTTP exception occurred",
            request_id, "/query",
            error_detail=str(he.detail) if hasattr(he, 'detail') else str(he),
            status_code=getattr(he, 'status_code', 'unknown'),
            natural_query=request.query,
            total_time_seconds=round(total_time, 3),
            enable_db_query=ENABLE_DB_QUERY
        )
        raise he
    except Exception as e:
        total_time = time.time() - start_time
        log_with_context(
            logger, 'error',
            f"SQL_QUERY_FATAL_ERROR - Unexpected error occurred",
            request_id, "/query",
            error=str(e),
            traceback=traceback.format_exc(),
            natural_query=request.query,
            total_time_seconds=round(total_time, 3),
            enable_db_query=ENABLE_DB_QUERY
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schema", response_model=SchemaInfo)
async def get_schema():
    """Get database schema information."""
    request_id = str(uuid.uuid4())
    endpoint_logger = get_contextual_logger(request_id, "/schema")
    start_time = time.time()
    
    try:
        log_with_context(
            logger, 'info',
            f"SCHEMA_REQUEST_START - Getting schema information",
            request_id, "/schema"
        )
        
        schema_info = get_schema_info(request_id)
        
        total_time = time.time() - start_time
        
        log_with_context(
            logger, 'info',
            f"SCHEMA_REQUEST_SUCCESS - Schema information retrieved",
            request_id, "/schema",
            table_count=len(schema_info.get("tables", {})),
            relationship_count=len(schema_info.get("relationships", [])),
            processing_time_seconds=round(total_time, 3)
        )
        
        return schema_info
        
    except Exception as e:
        total_time = time.time() - start_time
        log_with_context(
            logger, 'error',
            f"SCHEMA_REQUEST_ERROR - Error getting schema",
            request_id, "/schema",
            error=str(e),
            traceback=traceback.format_exc(),
            processing_time_seconds=round(total_time, 3)
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        log_with_context(
            logger, 'info',
            f"HEALTH_CHECK_START - Starting health check",
            request_id, "/health"
        )
        
        # Check database
        db_start_time = time.time()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_time = time.time() - db_start_time
        
        # Check LLM orchestrator
        llm_start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{LLM_ORCHESTRATOR_URL}/health") as response:
                if response.status != 200:
                    raise Exception(f"LLM orchestrator not healthy: {response.status}")
        llm_time = time.time() - llm_start_time
        
        # Get schema info for health check
        schema_start_time = time.time()
        schema_info = get_schema_info(request_id)
        schema_time = time.time() - schema_start_time
        
        total_time = time.time() - start_time
        
        health_data = {
            "status": "healthy",
            "database": "connected",
            "llm_orchestrator": "connected",
            "schema": {
                "tables_available": len(schema_info.get("tables", {})),
                "relationships": len(schema_info.get("relationships", []))
            },
            "timing": {
                "database_check_seconds": round(db_time, 3),
                "llm_orchestrator_check_seconds": round(llm_time, 3),
                "schema_check_seconds": round(schema_time, 3),
                "total_time_seconds": round(total_time, 3)
            }
        }
        
        log_with_context(
            logger, 'info',
            f"HEALTH_CHECK_SUCCESS - Health check completed",
            request_id, "/health",
            health_status=health_data,
            total_time_seconds=round(total_time, 3)
        )
        
        return health_data
        
    except Exception as e:
        total_time = time.time() - start_time
        log_with_context(
            logger, 'error',
            f"HEALTH_CHECK_ERROR - Health check failed",
            request_id, "/health",
            error=str(e),
            traceback=traceback.format_exc(),
            total_time_seconds=round(total_time, 3)
        )
        return {"status": "unhealthy", "error": str(e)}