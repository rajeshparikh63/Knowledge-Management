# Backend Dockerfile for SoldierIQ FastAPI Application

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libxcb1 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency installation
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies using uv
RUN uv sync --no-cache

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run both FastAPI and Celery worker in same container
# & runs them simultaneously
# Removed --pool=solo: Pinecone SDK requires ThreadPool (incompatible with solo mode)
# --concurrency=1 + rate_limit=5/m prevents resource exhaustion
CMD ["sh", "-c", "uv run uvicorn app.server:app --host 0.0.0.0 --port 8000 & uv run celery -A app.worker.celery_app worker --loglevel=info --pool=threads --concurrency=2"]