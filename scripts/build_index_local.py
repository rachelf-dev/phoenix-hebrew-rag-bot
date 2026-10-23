#!/usr/bin/env python3
# One-time local build: embed data/chunks.jsonl via Gemini, write index.npy + index_chunks.json + bm25.pkl.
"""Run from repo root: python scripts/build_index_local.py [--limit N] [--pause SECONDS]"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NumPy RAG index from chunks.jsonl (offline Gemini embed).")
    parser.add_argument("--limit", type=int, default=0, help="embed only first N chunks (smoke test)")
    parser.add_argument(
        "--pause",
        type=float,
        default=None,
        help="seconds to sleep after each embed batch (overrides config embedding.pause_seconds)",
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("build_index")

    from phoenix_rag.config import load_config
    from phoenix_rag.indexing.build_numpy import build_numpy_index

    cfg = load_config(args.config)
    if not cfg.chunks_path.is_file():
        log.error("missing %s — run: python -m phoenix_rag.cli chunk", cfg.chunks_path)
        return 1

    n, shape = build_numpy_index(cfg, limit=args.limit, pause_seconds=args.pause, log=log)
    print("")
    print(f"Done: {n} chunks, matrix shape {shape}")
    print("Wrote prebuilt files for the server (index_artifacts/). Do not embed on Render.")
    print(f"  {cfg.index_npy_path}")
    print(f"  {cfg.index_chunks_path}")
    print(f"  {cfg.bm25_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
