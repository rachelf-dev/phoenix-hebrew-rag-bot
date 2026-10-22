# Phoenix RAG FastAPI image: builds Chroma/BM25 on first boot if missing (see check_deploy_ready.py).
# No local torch/e5 — fits Render free tier RAM; index source is chunks.jsonl or data/clean on disk.

FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /app

# lxml + chromadb native deps; curl for optional health/debug
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY config/ config/
COPY phoenix_rag/ phoenix_rag/
COPY app/ app/
COPY scripts/check_deploy_ready.py scripts/check_deploy_ready.py
RUN mkdir -p data/chroma logs
# Index bootstrap source (tracked in git); must exist at /app/data/chunks.jsonl in the container.
COPY data/chunks.jsonl /app/data/chunks.jsonl
RUN test -s /app/data/chunks.jsonl

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=7200s --retries=5 \
    CMD curl -fsS http://127.0.0.1:${PORT:-8080}/health || exit 1

ENTRYPOINT ["python", "scripts/check_deploy_ready.py"]
CMD ["python", "-m", "phoenix_rag.cli", "run", "--host", "0.0.0.0"]
