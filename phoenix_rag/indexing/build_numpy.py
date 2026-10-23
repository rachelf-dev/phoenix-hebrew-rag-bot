# Shared offline index build: chunks.jsonl -> index.npy + index_chunks.json + bm25.pkl.
"""Used by scripts/build_index_local.py and `python -m phoenix_rag.cli index`."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict

import numpy as np

from ..config import Config
from .bm25_store import write_bm25
from .chunk import read_chunks
from .embed import Embedder


def build_numpy_index(
    cfg: Config,
    *,
    limit: int = 0,
    pause_seconds: float | None = None,
    log: logging.Logger | None = None,
) -> tuple[int, tuple[int, int]]:
    from ..cli import _embed_with_cache

    log = log or logging.getLogger("build_numpy")
    if not cfg.chunks_path.is_file():
        raise FileNotFoundError(f"missing {cfg.chunks_path}")

    chunks = list(read_chunks(cfg.chunks_path))
    if limit:
        chunks = chunks[:limit]
    if not chunks:
        raise RuntimeError("no chunks to embed")

    embedder = Embedder.from_config(cfg)
    if pause_seconds is not None:
        embedder.pause_seconds = float(pause_seconds)
    elif embedder.pause_seconds <= 0:
        embedder.pause_seconds = 1.0

    log.info("embedding %d chunks model=%s dim=%d", len(chunks), embedder.model_name, embedder.dim)
    t0 = time.monotonic()
    vectors = _embed_with_cache(cfg, embedder, chunks, log)
    log.info("embeddings done in %.1f min", (time.monotonic() - t0) / 60)

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    cfg.ensure_dirs()
    np.save(cfg.index_npy_path, matrix)
    payload = [asdict(c) for c in chunks]
    cfg.index_chunks_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    write_bm25(chunks, cfg.bm25_path, model=embedder.model_name)

    log.info(
        "wrote %s shape=%s bm25=%s",
        cfg.index_npy_path,
        matrix.shape,
        cfg.bm25_path,
    )
    return len(chunks), (int(matrix.shape[0]), int(matrix.shape[1]))
