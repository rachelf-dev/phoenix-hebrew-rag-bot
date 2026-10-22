# Pre-flight for Docker/Railway: verify GEMINI_API_KEY and on-disk Chroma/BM25 before uvicorn starts.
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
        _fail("GEMINI_API_KEY is not set (Railway → Variables)")

    chroma = ROOT / "data" / "chroma"
    if not chroma.is_dir():
        _fail(f"missing Chroma directory {chroma} — bake data/ into the image or mount a Railway volume at /app/data")
    sqlite = chroma / "chroma.sqlite3"
    if not sqlite.is_file():
        _fail(f"missing {sqlite} — run `python -m phoenix_rag.cli index` locally and redeploy with data/")

    bm25 = ROOT / "data" / "bm25.pkl"
    if not bm25.is_file():
        _fail(f"missing {bm25} — run index locally and include data/bm25.pkl in the build context")

    model_dir = ROOT / "models" / "multilingual-e5-base"
    weights = model_dir / "model.safetensors"
    if not weights.is_file():
        _fail(f"missing {weights} — rebuild the Docker image (weights are downloaded in Dockerfile)")

    if len(sys.argv) <= 1:
        print("deploy checks OK")
        return

    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
