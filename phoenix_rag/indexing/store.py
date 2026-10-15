"""Chroma vector store + BM25 sidecar. Collection metadata records the embedding model."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Iterable

import chromadb
from rank_bm25 import BM25Okapi

from ..config import Config
from .chunk import Chunk
from .hebrew_lex import tokenize

log = logging.getLogger(__name__)

_META_KEYS = (
    "url",
    "display_url",
    "page_title",
    "heading_path",
    "fetch_date",
    "content_hash",
    "section_type",
    "language",
    "block_types",
    "form_links",
    "token_count",
)


def _meta(c: Chunk) -> dict:
    return {
        "url": c.url,
        "display_url": c.display_url,
        "page_title": c.page_title,
        "heading_path": c.heading_path,
        "fetch_date": c.fetch_date or "",
        "content_hash": c.content_hash or "",
        "section_type": c.section_type,
        "language": c.language,
        "block_types": c.block_types,
        "form_links": (c.form_links or "")[:1000],
        "token_count": int(c.token_count),
    }


class VectorIndex:
    def __init__(self, cfg: Config, model_name: str):
        cfg.ensure_dirs()
        self.cfg = cfg
        self.model_name = model_name
        self.client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
        name = cfg.vector_store.get("collection", "fnx_pages")
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine", "embedding_model": model_name},
        )
        self.bm25_path = cfg.bm25_path

    def count(self) -> int:
        return self.collection.count()

    def replace_all(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Full rebuild: drop existing ids in this collection, then add."""
        existing = self.collection.get(include=[])
        if existing["ids"]:
            # delete in batches; chroma chokes on huge delete lists
            ids = existing["ids"]
            for i in range(0, len(ids), 500):
                self.collection.delete(ids=ids[i : i + 500])
            log.info("cleared %d old vectors", len(ids))
        self._add(chunks, embeddings)
        self._write_bm25(chunks)

    def _add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        batch = 200
        for i in range(0, len(chunks), batch):
            sl = chunks[i : i + batch]
            self.collection.add(
                ids=[c.chunk_id for c in sl],
                embeddings=embeddings[i : i + batch],
                documents=[c.original_text for c in sl],
                metadatas=[_meta(c) for c in sl],
            )
        log.info("upserted %d vectors", len(chunks))

    def _write_bm25(self, chunks: list[Chunk]) -> None:
        corpus = [tokenize(c.embed_text) for c in chunks]
        payload = {
            "ids": [c.chunk_id for c in chunks],
            "urls": [c.url for c in chunks],
            "tokens": corpus,
            "model": self.model_name,
        }
        # Keep the tokenised corpus; BM25Okapi is rebuilt on load (pickle of the class is brittle).
        self.bm25_path.write_bytes(pickle.dumps(payload, protocol=4))
        log.info("wrote BM25 corpus %s (%d docs)", self.bm25_path, len(chunks))

    def query(self, embedding: list[float], k: int = 12, where: dict | None = None) -> dict:
        kwargs: dict = {"query_embeddings": [embedding], "n_results": k, "include": ["documents", "metadatas", "distances"]}
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)


def load_bm25(path: Path) -> tuple[BM25Okapi, list[str], list[str]]:
    payload = pickle.loads(path.read_bytes())
    return BM25Okapi(payload["tokens"]), payload["ids"], payload["urls"]
