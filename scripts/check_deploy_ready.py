# Pre-flight for Docker/Render: ensure GEMINI_API_KEY, build Chroma/BM25 if missing, then exec the server.
"""Usage: check_deploy_ready.py [command ...] — exits 1 if deploy prerequisites are missing."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _ensure_import_path() -> None:
    """Running as `python scripts/check_deploy_ready.py` puts scripts/ on sys.path, not the repo root."""
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)
    prev = os.environ.get("PYTHONPATH", "")
    if root not in prev.split(os.pathsep):
        os.environ["PYTHONPATH"] = root if not prev else f"{root}{os.pathsep}{prev}"


_ensure_import_path()


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
        has_chunks = cfg.chunks_path.is_file() and cfg.chunks_path.stat().st_size > 0
        has_clean = cfg.clean_dir.is_dir() and any(cfg.clean_dir.glob("*.json"))
        if not has_chunks and not has_clean:
            _fail(
                f"missing index source: {cfg.chunks_path} (and no JSON under {cfg.clean_dir}). "
                "Docker image must COPY data/chunks.jsonl into /app/data/."
            )
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

    os.execvpe(sys.argv[1], sys.argv[1:], os.environ)


if __name__ == "__main__":
    main()
