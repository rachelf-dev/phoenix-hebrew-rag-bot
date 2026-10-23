# Optional helper: check that index_artifacts files exist. Deploy no longer downloads or embeds.

from __future__ import annotations

import logging

from ..config import Config
from .fetch_index import ensure_index_files

log = logging.getLogger(__name__)


def vector_index_ready(cfg: Config) -> bool:
    """True when index.npy and index_chunks.json exist and row counts match."""
    if not cfg.index_npy_path.is_file() or not cfg.index_chunks_path.is_file():
        return False
    try:
        import numpy as np

        matrix = np.load(cfg.index_npy_path, mmap_mode="r")
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return False
        import json

        raw = json.loads(cfg.index_chunks_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or len(raw) != matrix.shape[0]:
            return False
        if cfg.retrieval.get("hybrid", True) and not cfg.bm25_path.is_file():
            return False
        return True
    except Exception:
        return False


def ensure_vector_index(cfg: Config) -> None:
    """Download pre-built index artifacts if missing (no Gemini document embeddings)."""
    if vector_index_ready(cfg):
        log.info(
            "vector index OK (%s, %s)",
            cfg.index_npy_path.name,
            cfg.index_chunks_path.name,
        )
        return
    cfg.ensure_dirs()
    ensure_index_files(
        cfg.index_npy_path,
        cfg.index_chunks_path,
        bm25_path=cfg.bm25_path if cfg.retrieval.get("hybrid", True) else None,
    )
    if not vector_index_ready(cfg):
        raise RuntimeError(
            f"index files present but invalid: {cfg.index_npy_path}, {cfg.index_chunks_path}"
        )
    log.info("vector index ready after download/load check")
