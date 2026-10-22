# Phoenix RAG FastAPI image: Gemini embeddings at query time, Chroma + BM25 baked from build context.
# No local torch/e5 — fits Render free tier RAM when data/ is pre-indexed.

FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
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
# Index is gitignored; build context must include data/chroma + data/bm25.pkl (or mount a volume).
COPY data/ data/
RUN mkdir -p logs

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT:-8080}/health || exit 1

ENTRYPOINT ["python", "scripts/check_deploy_ready.py"]
CMD ["python", "-m", "phoenix_rag.cli", "run", "--host", "0.0.0.0"]
