# ==============================================================================
# Dockerfile for ScamShield-VN AI Pipeline Engine
# Multi-stage lightweight Python runtime with ONNX support
# ==============================================================================

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY pipeline /app/pipeline
COPY config /app/config
COPY models /app/models
COPY api /app/api
COPY scripts /app/scripts
COPY db/migrations /app/db/migrations
COPY demo_scamshield_mvp.py /app/

# Expose microservice port
EXPOSE 8000

# Healthcheck probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run Uvicorn server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
