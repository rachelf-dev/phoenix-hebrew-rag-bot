# Pre-flight for Docker/Render: verify GEMINI_API_KEY and on-disk Chroma/BM25 before uvicorn starts.
"""Usage: check_deploy_ready.py [command ...] — exits 1 if deploy prerequisites are missing."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _fail(msg: str) -> None:
    print(f"deploy check failed: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if not (os.getenv("GEMINI_API_KEY") or "").strip():
        _fail("GEMINI_API_KEY is not set (Render/Railway → Environment)")

    chroma = ROOT / "data" / "chroma"
    if not chroma.is_dir():
        _fail(f"missing Chroma directory {chroma} — bake data/ into the image or mount a volume at /app/data")
    sqlite = chroma / "chroma.sqlite3"
    if not sqlite.is_file():
        _fail(f"missing {sqlite} — run `python -m phoenix_rag.cli index` locally and redeploy with data/")

    bm25 = ROOT / "data" / "bm25.pkl"
    if not bm25.is_file():
        _fail(f"missing {bm25} — run index locally and include data/bm25.pkl in the build context")

    from phoenix_rag.config import load_config
    from phoenix_rag.indexing.store import read_chroma_embedding_model

    cfg = load_config()
    expected = cfg.embedding.get("model") or "gemini-embedding-001"
    collection = cfg.vector_store.get("collection", "fnx_pages")
    stored = read_chroma_embedding_model(cfg.chroma_dir, collection_name=collection)
    if stored and stored != expected:
        _fail(
            f"Chroma embedding_model={stored!r} but config expects {expected!r} — "
            "re-run `python -m phoenix_rag.cli index` and redeploy data/"
        )

    if len(sys.argv) <= 1:
        print("deploy checks OK")
        return

    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
