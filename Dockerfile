FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY pyproject.toml /app/

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools hatchling && \
    pip install --no-cache-dir .

# Copy application source and deploy artifacts
COPY src /app/src
COPY data /app/data
COPY deploy /app/deploy

# Expose API and Prometheus port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default command: run FastAPI via uvicorn
CMD ["uvicorn", "aiops.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
