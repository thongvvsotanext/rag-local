import os
import psycopg2
from sqlalchemy import inspect
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import logging
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from database.database import Base, engine, SessionLocal
from models import User, Document, CrawlJob, WebPage, DocumentChunk, ChatSession, ChatMessage, ProcessingJob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_database_exists():
    """Check if the database exists, create it if it doesn't."""
    try:
        # Get database configuration from environment variables
        db_user = os.getenv("POSTGRES_USER", "fizen_user")
        db_password = os.getenv("POSTGRES_PASSWORD", "fizen_password")
        db_name = os.getenv("POSTGRES_DB", "fizen_rag")
        db_host = os.getenv("POSTGRES_HOST", "db")
        db_port = os.getenv("POSTGRES_PORT", "5432")

        # Connect to default postgres database
        conn = psycopg2.connect(
            dbname="postgres",
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cursor.fetchone()
        
        if not exists:
            logger.info(f"Creating database {db_name}")
            cursor.execute(f"CREATE DATABASE {db_name}")
            logger.info(f"Database {db_name} created successfully")
        else:
            logger.info(f"Database {db_name} already exists")

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error checking/creating database: {str(e)}")
        return False

def create_schema():
    """Create the public schema if it doesn't exist."""
    try:
        # Connect to the database
        conn = psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB", "fizen_rag"),
            user=os.getenv("POSTGRES_USER", "fizen_user"),
            password=os.getenv("POSTGRES_PASSWORD", "fizen_password"),
            host=os.getenv("POSTGRES_HOST", "db"),
            port=os.getenv("POSTGRES_PORT", "5432")
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Create schema if it doesn't exist
        cursor.execute("CREATE SCHEMA IF NOT EXISTS public")
        logger.info("Schema 'public' created or already exists")

        # Set search path to public
        cursor.execute("SET search_path TO public")
        logger.info("Search path set to public schema")

        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error creating schema: {str(e)}")
        return False

def check_tables_exist():
    """Check if all required tables exist."""
    try:
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        required_tables = [
            "users",
            "documents",
            "crawl_jobs",
            "web_pages",
            "document_chunks",
            "chat_sessions",
            "chat_messages",
            "processing_jobs"
        ]
        
        missing_tables = [table for table in required_tables if table not in existing_tables]
        if missing_tables:
            logger.warning(f"Missing tables: {', '.join(missing_tables)}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error checking tables: {str(e)}")
        return False

def create_tables():
    """Create tables in the correct order to handle foreign key dependencies."""
    try:
        # Drop all tables first to ensure clean state
        Base.metadata.drop_all(bind=engine)
        logger.info("Dropped all existing tables")

        # Create tables in the correct order
        tables = [
            User.__table__,
            Document.__table__,
            CrawlJob.__table__,
            WebPage.__table__,
            DocumentChunk.__table__,
            ChatSession.__table__,
            ChatMessage.__table__,
            ProcessingJob.__table__
        ]

        for table in tables:
            table.create(engine)
            logger.info(f"Created table: {table.name}")

        return True
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
        return False

def init_db():
    """Initialize the database and create tables if they don't exist."""
    try:
        # First check if database exists
        if not check_database_exists():
            logger.error("Failed to verify database existence")
            return False

        # Create schema
        if not create_schema():
            logger.error("Failed to create schema")
            return False

        # Then check if tables exist
        if not check_tables_exist():
            logger.info("Creating database tables...")
            if not create_tables():
                return False
            logger.info("Database tables created successfully")
        else:
            logger.info("All required tables exist")

        return True
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Initialize database
    if init_db():
        logger.info("Database initialization completed successfully")
    else:
        logger.error("Database initialization failed") 