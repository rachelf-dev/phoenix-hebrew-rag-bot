"""Load the committed NumPy index from index_artifacts/ (no document embedding at runtime)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..indexing.bm25_store import load_or_rebuild_bm25
from ..indexing.embed import Embedder
from ..indexing.hebrew_lex import tokenize

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    url: str
    display_url: str
    page_title: str
    heading_path: str
    section_type: str
    form_links: str
    original_text: str
    embed_text: str


class NumpyRetriever:
    """Load index.npy + index_chunks.json + bm25.pkl once from index_artifacts/."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.index_npy_path = cfg.index_npy_path
        self.index_chunks_path = cfg.index_chunks_path
        self.bm25_path = cfg.bm25_path
        self._matrix: np.ndarray | None = None
        self._chunks: list[StoredChunk] | None = None
        self._id_to_row: dict[str, int] | None = None
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._bm25_urls: list[str] = []

    def load(self) -> None:
        if self._matrix is not None:
            return
        if not self.index_npy_path.is_file():
            raise FileNotFoundError(
                f"missing vector index: {self.index_npy_path} "
                "(copy index.npy into index_artifacts/; the server never embeds documents)"
            )
        if not self.index_chunks_path.is_file():
            raise FileNotFoundError(
                f"missing chunk metadata: {self.index_chunks_path} "
                "(copy index_chunks.json into index_artifacts/)"
            )

        matrix = np.asarray(np.load(self.index_npy_path), dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"index.npy must be 2-D, got shape {matrix.shape}")
        raw = json.loads(self.index_chunks_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("index_chunks.json must be a JSON array")
        chunks: list[StoredChunk] = []
        for row in raw:
            chunks.append(
                StoredChunk(
                    chunk_id=str(row["chunk_id"]),
                    url=str(row.get("url") or ""),
                    display_url=str(row.get("display_url") or ""),
                    page_title=str(row.get("page_title") or ""),
                    heading_path=str(row.get("heading_path") or ""),
                    section_type=str(row.get("section_type") or ""),
                    form_links=str(row.get("form_links") or ""),
                    original_text=str(row.get("original_text") or ""),
                    embed_text=str(row.get("embed_text") or ""),
                )
            )
        if len(chunks) != matrix.shape[0]:
            raise ValueError(
                f"index row count {matrix.shape[0]} != chunks {len(chunks)} "
                f"({self.index_npy_path.name} / {self.index_chunks_path.name})"
            )
        self._matrix = matrix
        self._chunks = chunks
        self._id_to_row = {c.chunk_id: i for i, c in enumerate(chunks)}
        self._bm25, self._bm25_ids, self._bm25_urls = load_or_rebuild_bm25(self.bm25_path, chunks)
        log.info(
            "numpy index loaded shape=%s dim=%d path=%s bm25_docs=%d",
            matrix.shape,
            matrix.shape[1],
            self.index_npy_path,
            len(self._bm25_ids),
        )

    @property
    def has_bm25(self) -> bool:
        self.load()
        return self._bm25 is not None and bool(self._bm25_ids)

    def count(self) -> int:
        self.load()
        assert self._chunks is not None
        return len(self._chunks)

    def chunk_by_id(self, chunk_id: str) -> StoredChunk | None:
        self.load()
        assert self._id_to_row is not None and self._chunks is not None
        i = self._id_to_row.get(chunk_id)
        return self._chunks[i] if i is not None else None

    def semantic_query(self, embedding: list[float], k: int) -> list[tuple[str, float]]:
        """Return (chunk_id, cosine score) for top-k rows."""
        self.load()
        assert self._matrix is not None
        k = max(1, min(int(k), self.count()))
        q = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm
        scores = self._matrix @ q
        return self._top_pairs(self._chunks, scores, k)  # type: ignore[arg-type]

    def lexical_query(self, question: str, k: int) -> list[tuple[str, float]]:
        """BM25 ranking from the in-memory corpus (rebuilt at startup if pickle failed)."""
        self.load()
        if self._bm25 is None or not self._bm25_ids:
            return []
        k = max(1, min(int(k), len(self._bm25_ids)))
        scores = self._bm25.get_scores(tokenize(question))
        return self._top_id_pairs(self._bm25_ids, scores, k)

    @staticmethod
    def _top_pairs(chunks: list[StoredChunk], scores: np.ndarray, k: int) -> list[tuple[str, float]]:
        top_idx = NumpyRetriever._top_indices(scores, k)
        return [(chunks[int(i)].chunk_id, float(scores[int(i)])) for i in top_idx]

    @staticmethod
    def _top_id_pairs(ids: list[str], scores: np.ndarray, k: int) -> list[tuple[str, float]]:
        top_idx = NumpyRetriever._top_indices(np.asarray(scores, dtype=np.float32), k)
        return [(ids[int(i)], float(scores[int(i)])) for i in top_idx]

    @staticmethod
    def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
        if k >= len(scores):
            return np.argsort(-scores)
        part = np.argpartition(-scores, k - 1)[:k]
        return part[np.argsort(-scores[part])]

    @classmethod
    def open(cls, cfg: Config, embedder: Embedder | None = None) -> "NumpyRetriever":
        del embedder  # query embedder is constructed separately; never used to embed documents here
        ret = cls(cfg)
        ret.load()
        return ret
