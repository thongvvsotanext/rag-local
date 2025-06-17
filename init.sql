-- Create the database if it doesn't exist
SELECT 'CREATE DATABASE fizen_rag' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'fizen_rag')\gexec

-- Connect to the database
\c fizen_rag

-- Create the required extensions if they don't exist
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
