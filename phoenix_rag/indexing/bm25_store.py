"""BM25 sidecar for hybrid retrieval (tokenised corpus on disk, BM25Okapi rebuilt on load).

If bm25.pkl cannot be unpickled, startup rebuilds the index from chunk text with no API calls.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from .chunk import Chunk
from .hebrew_lex import tokenize

log = logging.getLogger(__name__)


def write_bm25(chunks: list[Chunk], path: Path, *, model: str = "") -> None:
    corpus = [tokenize(c.embed_text) for c in chunks]
    payload = {
        "ids": [c.chunk_id for c in chunks],
        "urls": [c.url for c in chunks],
        "tokens": corpus,
        "model": model,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(payload, protocol=4))
    log.info("wrote BM25 corpus %s (%d docs)", path, len(chunks))


def load_bm25(path: Path) -> tuple[BM25Okapi, list[str], list[str]]:
    payload = pickle.loads(path.read_bytes())
    return BM25Okapi(payload["tokens"]), payload["ids"], payload["urls"]


def _row_field(row: object, name: str) -> str:
    if isinstance(row, dict):
        return str(row.get(name) or "")
    return str(getattr(row, name, "") or "")


def rebuild_bm25_from_chunks(chunks: list) -> tuple[BM25Okapi, list[str], list[str]]:
    """Tokenise chunk text already on disk and build BM25Okapi. Does not call an embedding API."""
    if not chunks:
        raise ValueError("cannot rebuild BM25: index_chunks.json has no rows")
    tokens = [tokenize(_row_field(row, "embed_text") or _row_field(row, "original_text")) for row in chunks]
    ids = [_row_field(row, "chunk_id") for row in chunks]
    urls = [_row_field(row, "url") for row in chunks]
    log.info("rebuilt BM25 from %d chunks (no API calls)", len(chunks))
    return BM25Okapi(tokens), ids, urls


def load_or_rebuild_bm25(path: Path, chunks: list) -> tuple[BM25Okapi, list[str], list[str]]:
    """Load bm25.pkl. On pickle or version mismatch, rebuild from chunk metadata."""
    if path.is_file():
        try:
            bm25, ids, urls = load_bm25(path)
            if len(ids) != len(chunks):
                raise ValueError(f"bm25 docs {len(ids)} != chunks {len(chunks)}")
            log.info("loaded BM25 %s (%d docs)", path, len(ids))
            return bm25, ids, urls
        except Exception as exc:
            log.warning(
                "bm25.pkl failed to load (%s: %s); rebuilding from index_chunks.json",
                type(exc).__name__,
                exc,
            )
    else:
        log.warning("bm25.pkl missing at %s; rebuilding from index_chunks.json", path)
    return rebuild_bm25_from_chunks(chunks)
