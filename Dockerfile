# Phoenix RAG FastAPI image: serves from committed index_artifacts/ (no document embed on boot).

FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PHOENIX_RUNTIME=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
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
COPY index_artifacts/ index_artifacts/
RUN mkdir -p logs

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -fsS http://127.0.0.1:${PORT:-8080}/health || exit 1

ENTRYPOINT ["python", "scripts/check_deploy_ready.py"]
CMD ["python", "-m", "phoenix_rag.cli", "run", "--host", "0.0.0.0"]
