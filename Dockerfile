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

# Install CPU-only PyTorch first to avoid pulling multi-GB CUDA wheels on Railway
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.5,<3"

# Install remaining python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY pipeline /app/pipeline
COPY config /app/config
COPY models /app/models
# The scan router loads the release tokenizer and prefers these INT8 weights.
COPY visobert_5runs_release/tokenizer.json visobert_5runs_release/tokenizer_config.json visobert_5runs_release/config.json visobert_5runs_release/visobert_tier1_int8.onnx /app/visobert_5runs_release/
# Fail the build if assets are missing, LFS pointers, or incompatible with inference.
RUN python -c "from pipeline.local_classifier import LocalViSoBERTClassifier; model = LocalViSoBERTClassifier(); assert model.has_onnx and model.tokenizer is not None, 'ViSoBERT ONNX model/tokenizer failed to load'; result = model.predict('Kiem tra ViSoBERT'); assert result['tier'] == 'Tier 1 (Local ViSoBERT ONNX INT8/CPU)', result"
COPY api /app/api
COPY scripts /app/scripts
COPY db/migrations /app/db/migrations
COPY demo_scamshield_mvp.py /app/

# Expose microservice port
EXPOSE 8000

# Healthcheck probe (respects Railway dynamic PORT)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD sh -c 'curl -f http://localhost:${PORT:-8000}/health || exit 1'

# Run Uvicorn server (binds to Railway $PORT or defaults to 8000)
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
