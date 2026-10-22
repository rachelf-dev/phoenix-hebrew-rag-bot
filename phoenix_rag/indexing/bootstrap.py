# Startup helper: detect a missing Chroma/BM25 index and build it from chunks.jsonl or data/clean/.
"""Used by check_deploy_ready before the web server starts on Render and similar hosts."""

from __future__ import annotations

import logging

import chromadb
from chromadb.errors import NotFoundError

from ..config import Config
from .chunk import chunk_page, load_clean_pages, read_chunks, write_chunks
from .embed import Embedder
from .store import VectorIndex, read_chroma_embedding_model

log = logging.getLogger(__name__)


def vector_index_ready(cfg: Config) -> bool:
    """True when Chroma + BM25 exist, the embedding model matches config, and the collection is non-empty."""
    if not cfg.chroma_dir.is_dir():
        return False
    if not (cfg.chroma_dir / "chroma.sqlite3").is_file():
        return False
    if not cfg.bm25_path.is_file():
        return False
    expected = cfg.embedding.get("model") or "gemini-embedding-001"
    collection = cfg.vector_store.get("collection", "fnx_pages")
    stored = read_chroma_embedding_model(cfg.chroma_dir, collection_name=collection)
    if stored and stored != expected:
        return False
    try:
        client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
        col = client.get_collection(name=collection)
        return col.count() > 0
    except NotFoundError:
        return False
    except Exception:
        return False


def _load_or_build_chunks(cfg: Config, *, rechunk: bool) -> list:
    if rechunk or not cfg.chunks_path.is_file():
        if not cfg.clean_dir.is_dir() or not any(cfg.clean_dir.glob("*.json")):
            raise RuntimeError(
                f"vector index missing and no {cfg.chunks_path} or JSON under {cfg.clean_dir} — "
                "run crawl/clean/chunk locally, copy chunks.jsonl to the server, or mount a volume"
            )
        log.info("building chunks from %s", cfg.clean_dir)
        chunks = []
        for page in load_clean_pages(cfg.clean_dir, skip_short=True):
            chunks.extend(chunk_page(page, cfg))
        write_chunks(chunks, cfg.chunks_path)
        log.info("wrote %d chunks to %s", len(chunks), cfg.chunks_path)
        return chunks
    chunks = list(read_chunks(cfg.chunks_path))
    log.info("loaded %d chunks from %s", len(chunks), cfg.chunks_path)
    return chunks


def build_vector_index(
    cfg: Config,
    *,
    rechunk: bool = False,
    limit: int = 0,
    skip_probe: bool = True,
) -> int:
    """Embed chunks and write Chroma + BM25 (same path as `cli index`, without argparse)."""
    from ..cli import _embed_with_cache, _run_probe, _setup_logging

    log_path = _setup_logging(cfg, "index")
    idx_log = logging.getLogger("index")
    cfg.ensure_dirs()
    chunks = _load_or_build_chunks(cfg, rechunk=rechunk)
    if limit:
        chunks = chunks[:limit]
        idx_log.info("limit: using first %d chunks", len(chunks))
    if not chunks:
        raise RuntimeError("no chunks to index")

    embedder = Embedder.from_config(cfg)
    vectors = _embed_with_cache(cfg, embedder, chunks, idx_log)
    index = VectorIndex(cfg, embedder, recreate_on_mismatch=True)
    index.replace_all(chunks, vectors)
    idx_log.info(
        "indexed %d chunks model=%s dim=%d chroma=%d log=%s",
        len(chunks), embedder.model_name, embedder.dim, index.count(), log_path,
    )
    if not skip_probe:
        _run_probe(cfg, embedder, index, k=int(cfg.retrieval.get("top_k", 12)), hybrid=True)
    return index.count()


def ensure_vector_index(cfg: Config) -> None:
    if vector_index_ready(cfg):
        log.info("vector index already present at %s", cfg.chroma_dir)
        return
    log.warning("vector index missing or stale; building before serving traffic")
    build_vector_index(cfg, skip_probe=True)
