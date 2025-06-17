from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import logging
import sys
from datetime import datetime
import boto3
import json
from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.engine import URL
import sqlparse
from contextlib import asynccontextmanager
import asyncio
import aiohttp
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('sql_retriever.log')
    ]
)
logger = logging.getLogger(__name__)

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

BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
BEDROCK_MODEL = os.getenv("BEDROCK_MODEL", "anthropic.claude-v2")

# Initialize Bedrock client
bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

# Database setup
engine = create_engine(DATABASE_URL)
metadata = MetaData()
metadata.reflect(bind=engine)

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

# Helper Functions
def get_schema_info() -> Dict[str, Any]:
    """Get database schema information."""
    inspector = inspect(engine)
    schema_info = {
        "tables": {},
        "relationships": []
    }
    
    # Get table information
    for table_name in inspector.get_table_names():
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
    
    return schema_info

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
4. Add LIMIT clause to prevent large result sets
5. Include comments explaining the query logic

SQL Query:"""
    
    return prompt

async def generate_sql(natural_query: str, schema_info: Dict[str, Any]) -> str:
    """Generate SQL from natural language using Bedrock."""
    try:
        prompt = generate_sql_prompt(natural_query, schema_info)
        
        response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL,
            body=json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": 1000,
                "temperature": 0.1,
                "stop_sequences": ["\n\n"]
            }),
            contentType="application/json"
        )
        
        result = json.loads(response["body"].read())
        sql = result["completion"].strip()
        
        # Extract SQL from the response (in case model includes explanations)
        sql_match = re.search(r"```sql\n(.*?)\n```", sql, re.DOTALL)
        if sql_match:
            sql = sql_match.group(1)
        
        # Validate SQL
        parsed = sqlparse.parse(sql)[0]
        if not parsed.get_type().lower() == "select":
            raise ValueError("Only SELECT queries are allowed")
        
        return sql
    
    except Exception as e:
        logger.error(f"Error generating SQL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating SQL: {str(e)}")

def validate_sql(sql: str) -> bool:
    """Validate SQL query for safety."""
    try:
        # Parse SQL
        parsed = sqlparse.parse(sql)[0]
        
        # Check if it's a SELECT query
        if not parsed.get_type().lower() == "select":
            return False
        
        # Check for dangerous operations
        dangerous_ops = ["insert", "update", "delete", "drop", "alter", "create", "truncate"]
        sql_lower = sql.lower()
        if any(op in sql_lower for op in dangerous_ops):
            return False
        
        # Check for proper LIMIT clause
        if "limit" not in sql_lower:
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error validating SQL: {str(e)}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI app."""
    # Startup
    logger.info("Starting up SQL Retriever service...")
    try:
        # Test database connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection successful")
        
        # Test Bedrock connection
        bedrock.list_foundation_models()
        logger.info("Bedrock connection successful")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down SQL Retriever service...")
    try:
        engine.dispose()
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

app = FastAPI(title="SQL Retriever Service", lifespan=lifespan)

@app.post("/query", response_model=SQLQueryResponse)
async def execute_query(request: SQLQueryRequest):
    """Execute a natural language query and return results."""
    try:
        start_time = datetime.utcnow()
        
        # Get schema information
        schema_info = get_schema_info()
        
        # Generate SQL from natural language
        sql = await generate_sql(request.query, schema_info)
        
        # Validate SQL
        if not validate_sql(sql):
            raise HTTPException(status_code=400, detail="Generated SQL query is not safe")
        
        # Execute query with timeout
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
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return SQLQueryResponse(
            results=results,
            sql_generated=sql,
            execution_time=execution_time,
            row_count=len(results),
            schema_info=schema_info
        )
    
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schema", response_model=SchemaInfo)
async def get_schema():
    """Get database schema information."""
    try:
        return get_schema_info()
    except Exception as e:
        logger.error(f"Error getting schema: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Check Bedrock
        bedrock.list_foundation_models()
        
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)} 