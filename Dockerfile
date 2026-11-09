# syntax=docker/dockerfile:1
# Agentic RAG Assistant: API + web UI in one image (CPU-only PyTorch).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app
RUN useradd --create-home --uid 10001 appuser

# Dependencies first for layer caching. The lock pins torch==<ver>+cpu from the PyTorch CPU index.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

# Bake the embedding model into the image so containers start without network access.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG PRELOAD_MODEL=true
RUN if [ "$PRELOAD_MODEL" = "true" ]; then \
      python -c "import sys; from sentence_transformers import SentenceTransformer; SentenceTransformer(sys.argv[1])" "$EMBEDDING_MODEL"; \
    fi

COPY backend/ backend/
COPY frontend/ frontend/
RUN mkdir -p /app/data/uploads && chown -R appuser:appuser /app

USER appuser
WORKDIR /app/backend
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Migrations run once before the server starts (AUTO_MIGRATE is disabled in compose).
CMD ["sh", "-c", "python -m app.db.migrate && exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips=${FORWARDED_ALLOW_IPS:-127.0.0.1}"]
