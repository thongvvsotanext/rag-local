#!/bin/bash

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
while ! nc -z $POSTGRES_HOST $POSTGRES_PORT; do
    sleep 0.1
done
echo "PostgreSQL is ready!"

# Initialize database
echo "Initializing database..."
python -m database.init_db

# Start the API
echo "Starting API..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload 