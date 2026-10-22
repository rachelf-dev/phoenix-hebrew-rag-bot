# Phoenix RAG FastAPI image for Railway: CPU PyTorch, offline e5 weights, Chroma + BM25 from build context.
# Plan for 2–4 GB RAM (embedding model ~500 MB loaded + Chroma + request headroom).

FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_XET=1

WORKDIR /app

# lxml + chromadb native deps; curl for optional health/debug
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch keeps image smaller and matches sentence-transformers on Railway.
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt && \
    pip install "huggingface_hub>=0.26.0"

# Bake multilingual-e5-base into the image (not in git). Index data/ must be present at build or mounted at runtime.
RUN python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "intfloat/multilingual-e5-base",
    local_dir="/app/models/multilingual-e5-base",
    local_dir_use_symlinks=False,
)
PY

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY config/ config/
COPY phoenix_rag/ phoenix_rag/
COPY app/ app/
COPY scripts/check_deploy_ready.py scripts/check_deploy_ready.py
# Index is gitignored; `railway up` from a machine with data/ sends it in the build context.
# Or mount a Railway volume at /app/data with chroma/ + bm25.pkl.
COPY data/ data/
RUN mkdir -p logs

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT:-8080}/health || exit 1

ENTRYPOINT ["python", "scripts/check_deploy_ready.py"]
CMD ["python", "-m", "phoenix_rag.cli", "run", "--host", "0.0.0.0"]
