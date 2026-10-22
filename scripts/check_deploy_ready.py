# Pre-flight for Docker/Render: ensure GEMINI_API_KEY, build Chroma/BM25 if missing, then exec the server.
"""Usage: check_deploy_ready.py [command ...] — exits 1 if deploy prerequisites are missing."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _fail(msg: str) -> None:
    print(f"deploy check failed: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from phoenix_rag.gemini_keys import gemini_keys

    if not gemini_keys():
        _fail("GEMINI_API_KEY is not set (Render/Railway → Environment)")

    from phoenix_rag.config import load_config
    from phoenix_rag.indexing.bootstrap import ensure_vector_index, vector_index_ready

    cfg = load_config()
    cfg.ensure_dirs()

    if vector_index_ready(cfg):
        print("vector index OK")
    else:
        print("vector index missing — building from chunks/clean (this may take a long time on first boot)", flush=True)
        try:
            ensure_vector_index(cfg)
        except Exception as exc:
            _fail(str(exc))
        if not vector_index_ready(cfg):
            _fail("index build finished but vector store is still empty")
        print("vector index build complete")

    if len(sys.argv) <= 1:
        print("deploy checks OK")
        return

    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
