"""Chroma vector store + BM25 sidecar. Collection metadata records the embedding model; Gemini EF avoids default ONNX."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from rank_bm25 import BM25Okapi

from ..config import Config
from .chunk import Chunk
from .embed import Embedder, GeminiEmbeddingFunction
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


def _collection_meta(model_name: str) -> dict[str, str]:
    return {"hnsw:space": "cosine", "embedding_model": model_name}


def _stored_embedding_model(collection) -> str | None:
    meta = collection.metadata or {}
    raw = meta.get("embedding_model")
    return str(raw) if raw else None


def open_collection(
    client: chromadb.PersistentClient,
    name: str,
    model_name: str,
    ef: GeminiEmbeddingFunction,
    *,
    recreate_on_mismatch: bool,
) -> chromadb.Collection:
    """Open or create the collection; on index runs, drop and recreate when the stored model differs."""
    try:
        try:
            col = client.get_collection(name=name, embedding_function=ef)
        except ValueError as exc:
            # Legacy collections were created with Chroma's default EF; queries use query_embeddings only.
            if "Embedding function conflict" not in str(exc):
                raise
            log.info("opening collection %s without persisted embedding function (legacy index)", name)
            col = client.get_collection(name=name)
        stored = _stored_embedding_model(col)
        if stored and stored != model_name:
            if recreate_on_mismatch:
                log.info("embedding model changed (%s -> %s); recreating collection %s", stored, model_name, name)
                client.delete_collection(name)
                return client.create_collection(name=name, metadata=_collection_meta(model_name), embedding_function=ef)
            raise RuntimeError(
                f"Chroma collection {name!r} was built with embedding_model={stored!r}, "
                f"but config expects {model_name!r}. Run: python -m phoenix_rag.cli index"
            )
        return col
    except NotFoundError:
        return client.create_collection(
            name=name, metadata=_collection_meta(model_name), embedding_function=ef,
        )


class VectorIndex:
    def __init__(self, cfg: Config, embedder: Embedder, *, recreate_on_mismatch: bool = False):
        cfg.ensure_dirs()
        self.cfg = cfg
        self.embedder = embedder
        self.model_name = embedder.model_name
        self.client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
        name = cfg.vector_store.get("collection", "fnx_pages")
        ef = GeminiEmbeddingFunction(embedder)
        self.collection = open_collection(
            self.client, name, self.model_name, ef, recreate_on_mismatch=recreate_on_mismatch,
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


def read_chroma_embedding_model(chroma_dir: Path, collection_name: str = "fnx_pages") -> str | None:
    """Return embedding_model metadata without loading a default Chroma embedder."""
    if not chroma_dir.is_dir():
        return None
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        col = client.get_collection(name=collection_name)
    except NotFoundError:
        return None
    return _stored_embedding_model(col)
