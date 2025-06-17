# Local Deployment Guide

This guide provides step-by-step instructions for deploying the Multi-Source RAG system on a local machine, with specific optimizations for M2 Mac.

## Prerequisites

1. **System Requirements**:
   - MacBook with M2 chip
   - Minimum 16GB RAM
   - At least 50GB free disk space
   - macOS 12.0 or later

2. **Required Software**:
   - Docker Desktop for Mac (latest version)
   - Git
   - Python 3.11 or later
   - Ollama (for local LLM support)

## Installation Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Multisource-RAG-Local
```

### 2. Install Ollama
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
```

### 3. Pull Required Models
```bash
# Make the setup script executable
chmod +x setup_ollama.sh

# Run the setup script
./setup_ollama.sh
```

### 4. Configure Environment
```bash
# Create and configure .env file
cp .env.example .env

# Edit .env with your settings
nano .env
```

Required environment variables:
```env
# Database
POSTGRES_USER=raguser
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=ragdb
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Vector Service
VECTOR_SERVICE_PORT=8003
FAISS_INDEX_PATH=/app/data/faiss
MODEL_CACHE_DIR=/app/model_cache

# LLM Service
LLM_SERVICE_PORT=8004
OLLAMA_HOST=host.docker.internal
OLLAMA_PORT=11434

# API Service
API_PORT=8000
```

### 5. Build and Start Services
```bash
# Build all services
docker-compose build

# Start services in detached mode
docker-compose up -d
```

### 6. Verify Deployment
```bash
# Check service status
docker-compose ps

# Check service logs
docker-compose logs -f
```

## Service Endpoints

Once deployed, the following endpoints will be available:

1. **API Service**: `http://localhost:8000`
   - Health check: `GET /health`
   - Chat endpoint: `POST /chat`
   - Document upload: `POST /documents/upload`

2. **Vector Service**: `http://localhost:8003`
   - Health check: `GET /health`
   - Search: `POST /search`
   - Chat context: `POST /chat-context`

3. **LLM Service**: `http://localhost:8004`
   - Health check: `GET /health`
   - Generate: `POST /generate`

## Performance Optimization

### Memory Management
- Monitor memory usage: `docker stats`
- Adjust worker counts in docker-compose.yml if needed
- Use `docker-compose down` to free resources when not in use

### Storage Management
- FAISS indices are stored in `./data/faiss`
- Document chunks are stored in PostgreSQL
- Model cache is in `./model_cache`

### Troubleshooting

1. **Service Not Starting**:
   ```bash
   # Check logs
   docker-compose logs <service-name>
   
   # Restart service
   docker-compose restart <service-name>
   ```

2. **Memory Issues**:
   ```bash
   # Reduce worker count in docker-compose.yml
   # Restart services
   docker-compose down
   docker-compose up -d
   ```

3. **Database Connection Issues**:
   ```bash
   # Check database logs
   docker-compose logs postgres
   
   # Verify connection
   docker-compose exec postgres psql -U raguser -d ragdb
   ```

## Development Workflow

1. **Making Changes**:
   ```bash
   # Stop services
   docker-compose down
   
   # Rebuild specific service
   docker-compose build <service-name>
   
   # Start services
   docker-compose up -d
   ```

2. **Viewing Logs**:
   ```bash
   # All services
   docker-compose logs -f
   
   # Specific service
   docker-compose logs -f <service-name>
   ```

3. **Database Management**:
   ```bash
   # Connect to database
   docker-compose exec postgres psql -U raguser -d ragdb
   
   # Run migrations
   docker-compose exec api alembic upgrade head
   ```

## Security Considerations

1. **Environment Variables**:
   - Never commit `.env` file
   - Use strong passwords
   - Rotate credentials regularly

2. **Network Security**:
   - Services are only exposed locally
   - Use HTTPS in production
   - Implement proper authentication

3. **Data Security**:
   - Regular backups of PostgreSQL data
   - Secure storage of FAISS indices
   - Proper access controls

## Maintenance

1. **Regular Updates**:
   ```bash
   # Pull latest changes
   git pull
   
   # Rebuild services
   docker-compose build
   
   # Restart services
   docker-compose down
   docker-compose up -d
   ```

2. **Backup**:
   ```bash
   # Backup database
   docker-compose exec postgres pg_dump -U raguser ragdb > backup.sql
   
   # Backup FAISS indices
   tar -czf faiss_backup.tar.gz ./data/faiss
   ```

3. **Cleanup**:
   ```bash
   # Remove unused containers
   docker container prune
   
   # Remove unused images
   docker image prune
   
   # Remove unused volumes
   docker volume prune
   ```

## Support

For issues and support:
1. Check the troubleshooting section
2. Review service logs
3. Check GitHub issues
4. Contact development team

## License

[Your License Information] 