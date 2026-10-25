# Rebuild index_artifacts/bm25.pkl from index_chunks.json (no embedding API calls).
"""Usage: python scripts/rebuild_bm25.py [--config PATH]"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rank_bm25 import BM25Okapi

from phoenix_rag.config import load_config
from phoenix_rag.indexing.hebrew_lex import tokenize


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild bm25.pkl from index_chunks.json")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    chunks_path = cfg.index_chunks_path
    bm25_path = cfg.bm25_path
    import json

    raw = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit("index_chunks.json must be a JSON array")

    tokens = [tokenize(row.get("embed_text") or row.get("original_text") or "") for row in raw]
    ids = [str(row["chunk_id"]) for row in raw]
    urls = [str(row.get("url") or "") for row in raw]
    BM25Okapi(tokens)  # validate corpus
    payload = {"ids": ids, "urls": urls, "tokens": tokens, "model": "hebrew_lex_v2"}
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    bm25_path.write_bytes(pickle.dumps(payload, protocol=4))
    print(f"wrote {bm25_path} docs={len(ids)} (order matches index_chunks.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
