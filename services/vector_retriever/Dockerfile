# Use Python 3.11 slim as base image for ARM64
FROM python:3.11-slim

# Set environment variables for M2 optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TRANSFORMERS_CACHE=/app/model_cache \
    TORCH_HOME=/app/model_cache \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    VECLIB_MAXIMUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    MKL_DYNAMIC=FALSE \
    OMP_DYNAMIC=FALSE \
    OPENBLAS_DYNAMIC=FALSE

# Set working directory
WORKDIR /app

# Install system dependencies optimized for M2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies - all packages are now in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create model cache directory and data directories
RUN mkdir -p /app/model_cache && \
    mkdir -p /app/data/faiss && \
    mkdir -p /app/data/vectors && \
    mkdir -p /app/data/cache

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Run the application with optimized settings for M2
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "2", "--loop", "uvloop", "--http", "httptools", "--limit-concurrency", "100", "--backlog", "2048"]