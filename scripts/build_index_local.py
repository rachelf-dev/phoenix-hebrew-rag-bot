# Offline rebuild of index_artifacts/index.new.npy from index_chunks.json + Gemini cache.
# Never overwrites index.npy. Only embeds missing RETRIEVAL_DOCUMENT vectors; stops on daily quota.
"""Usage: python scripts/build_index_local.py [--report-only] [--max-new N] [--pause SECONDS]"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_ROWS = 11590
MODEL = "gemini-embedding-001"
DIM = 768
TASK = "RETRIEVAL_DOCUMENT"
LOG = logging.getLogger("build_index")


def _cache_key(text: str) -> str:
    """Same formula as EmbeddingCache: {model}:{dim}:{sha1(embed_text)}. Task is not stored."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{MODEL}:{DIM}:{digest}"


def _load_chunks(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != EXPECTED_ROWS:
        raise SystemExit(
            f"refuse: {path} must have exactly {EXPECTED_ROWS} rows, got "
            f"{len(raw) if isinstance(raw, list) else type(raw).__name__}"
        )
    return raw


def _inspect_cache(cache_path: Path, texts: list[str]) -> tuple[int, int, dict[str, int]]:
    """Return (cached_matching, missing, prefix_counts). No API calls."""
    prefixes: dict[str, int] = {}
    have: set[str] = set()
    if cache_path.is_file():
        conn = sqlite3.connect(str(cache_path))
        for (key,) in conn.execute("SELECT key FROM emb"):
            parts = key.split(":")
            pref = f"{parts[0]}:{parts[1]}" if len(parts) >= 3 else key[:48]
            prefixes[pref] = prefixes.get(pref, 0) + 1
            have.add(key)
        conn.close()
    wanted = [_cache_key(t) for t in texts]
    cached = sum(1 for k in wanted if k in have)
    return cached, len(wanted) - cached, prefixes


def _report(cache_path: Path, texts: list[str]) -> tuple[int, int]:
    cached, missing, prefixes = _inspect_cache(cache_path, texts)
    print(f"chunks {len(texts)} from index_chunks.json")
    print(f"cache file: {cache_path}  exists={cache_path.is_file()}")
    print("cache key: {model}:{dim}:{sha1(utf-8 embed_text)}")
    print(f"  model={MODEL}  dim={DIM}  task expected={TASK} (task is NOT in the key)")
    print(f"key prefixes in sqlite: {prefixes or '{}'}")
    e5 = prefixes.get("models/multilingual-e5-base:768", 0)
    if e5:
        print(f"NOTE: {e5} e5 vectors share the file; they use a different prefix and are never mixed in.")
    print(f"cached {cached} / missing {missing}")
    return cached, missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild index_artifacts/index.new.npy (Gemini RETRIEVAL_DOCUMENT). Does not touch index.npy.",
    )
    parser.add_argument("--report-only", action="store_true", help="cache coverage only, no API calls")
    parser.add_argument("--max-new", type=int, default=0, help="embed at most N missing chunks this run (0 = all missing)")
    parser.add_argument("--pause", type=float, default=None, help="seconds after each embed batch")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    os.environ.pop("PHOENIX_RUNTIME", None)

    from phoenix_rag.config import load_config
    from phoenix_rag.indexing.embed import Embedder
    from phoenix_rag.indexing.embed_cache import EmbeddingCache

    cfg = load_config(args.config)
    chunks_path = cfg.index_chunks_path
    if not chunks_path.is_file():
        LOG.error("missing %s", chunks_path)
        return 1

    rows = _load_chunks(chunks_path)
    texts = [str(row.get("embed_text") or "") for row in rows]
    if any(not t for t in texts):
        LOG.error("refuse: empty embed_text in index_chunks.json")
        return 1

    cache_rel = cfg.embedding.get("cache_path") or "data/embeddings_cache.sqlite"
    cache_path = cfg.root / cache_rel
    cached_n, missing_n = _report(cache_path, texts)
    if args.report_only:
        return 0

    if cfg.embedding.get("model") != MODEL or int(cfg.embedding.get("output_dimensionality") or 0) != DIM:
        LOG.error("refuse: config embedding must be %s dim %d", MODEL, DIM)
        return 1
    if cfg.embedding.get("task_document", TASK) != TASK:
        LOG.error("refuse: task_document must be %s", TASK)
        return 1

    embedder = Embedder.from_config(cfg)
    if embedder.model_name != MODEL or int(embedder.dim) != DIM:
        LOG.error("refuse: embedder is %s dim %s", embedder.model_name, embedder.dim)
        return 1
    if args.pause is not None:
        embedder.pause_seconds = float(args.pause)
    elif embedder.pause_seconds <= 0:
        embedder.pause_seconds = 1.0

    cache = EmbeddingCache(cache_path, MODEL, DIM)
    have = cache.get_many(texts)
    todo = [i for i, text in enumerate(texts) if cache.key(text) not in have]
    if args.max_new and args.max_new > 0:
        todo = todo[: args.max_new]

    embedded_now = 0
    group = 50
    started = time.monotonic()
    try:
        for g in range(0, len(todo), group):
            idx = todo[g : g + group]
            batch_texts = [texts[i] for i in idx]
            try:
                vecs = embedder.encode_passages(batch_texts)
            except Exception as exc:
                msg = str(exc)
                cache.close()
                print(f"cached {cached_n} / missing {missing_n} / embedded now {embedded_now}")
                if "daily embedding quota" in msg or "daily quota" in msg.lower():
                    LOG.warning("daily quota — cache kept, index.new.npy not written")
                    print(msg[:400])
                    return 2
                raise
            cache.put_many(batch_texts, vecs)
            embedded_now += len(vecs)
            elapsed = time.monotonic() - started
            LOG.info(
                "embedded %d/%d this run  (%.0fs)",
                embedded_now,
                len(todo),
                elapsed,
            )
    finally:
        cache.close()

    cache = EmbeddingCache(cache_path, MODEL, DIM)
    have = cache.get_many(texts)
    present = [cache.key(t) in have for t in texts]
    cached_after = sum(present)
    still_missing = len(texts) - cached_after
    print(f"cached {cached_after} / missing {still_missing} / embedded now {embedded_now}")

    out_path = cfg.index_npy_path.parent / "index.new.npy"
    if still_missing:
        cache.close()
        print(f"refuse to finalize: need {EXPECTED_ROWS} vectors, have {cached_after}. Re-run later (cache kept).")
        print(f"did not write {out_path}")
        return 2

    import numpy as np

    matrix = np.asarray([have[cache.key(t)] for t in texts], dtype=np.float32)
    cache.close()
    if matrix.shape != (EXPECTED_ROWS, DIM):
        LOG.error("refuse: matrix shape %s != (%d, %d)", matrix.shape, EXPECTED_ROWS, DIM)
        return 1
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    np.save(out_path, matrix)
    print(f"wrote {out_path} shape={tuple(matrix.shape)}  (index.npy untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
