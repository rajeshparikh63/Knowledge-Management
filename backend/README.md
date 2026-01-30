# SoldierIQ Backend

Tactical Intelligence Knowledge Management System - FastAPI Backend

## Features

- ✅ FastAPI application with async support
- ✅ Health check endpoints
- ✅ CORS middleware
- ✅ Custom logging middleware
- ✅ Security headers middleware
- ✅ Environment-based configuration
- ✅ UV for fast dependency management

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

1. **Install uv (if not already installed):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Sync dependencies:**
```bash
cd backend
uv sync
```

3. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Run the application:**
```bash
uv run python -m app.server
```

Or using uvicorn directly:
```bash
uv run uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /healthz` - Health check (alternative)

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── server.py          # Main FastAPI application
│   ├── settings.py        # Configuration settings
│   ├── middleware.py      # Custom middleware
│   └── logger.py          # Logging configuration
├── routers/
│   ├── __init__.py
│   └── health.py          # Health check router
├── models/
│   ├── __init__.py
│   └── health.py          # Pydantic models
├── utils/                 # Utility functions
├── services/              # Business logic services
├── .env.example           # Environment variables template
├── .gitignore
├── pyproject.toml         # UV dependencies
└── README.md
```

## Development

The application includes:
- **Logging Middleware**: Logs all requests with processing time
- **Security Headers**: Adds security headers to responses
- **CORS**: Configured for cross-origin requests
- **Async/Await**: Full async support for better performance

## UV Commands

```bash
# Install dependencies
uv sync

# Add a new dependency
uv add package-name

# Remove a dependency
uv remove package-name

# Update dependencies
uv lock --upgrade

# Run Python scripts
uv run python script.py

# Run tests
uv run pytest
```

## Next Steps

- Implement knowledge base management endpoints
- Add document upload and processing
- Integrate vector database (Pinecone)
- Implement RAG query system
- Add authentication middleware
- Implement TTS/STT endpoints
