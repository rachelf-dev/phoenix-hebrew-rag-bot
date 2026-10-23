# Pre-flight for Docker/Render: require GEMINI_API_KEY and the three index_artifacts files, then exec the server.
"""Never embeds documents. Fails fast if the prebuilt index is missing from the image."""

from __future__ import annotations

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
    from phoenix_rag.config import load_config
    from phoenix_rag.gemini_keys import gemini_keys

    cfg = load_config()
    if not gemini_keys():
        _fail("GEMINI_API_KEY is not set (Render → Environment, or a local .env)")
    required = (
        ("index.npy", cfg.index_npy_path),
        ("index_chunks.json", cfg.index_chunks_path),
        ("bm25.pkl", cfg.bm25_path),
    )
    missing = [f"{name} ({path})" for name, path in required if not path.is_file()]
    if missing:
        _fail(
            "prebuilt index missing: "
            + "; ".join(missing)
            + ". Copy index.npy, index_chunks.json and bm25.pkl into index_artifacts/ "
            "before building the image. The server never embeds documents."
        )

    print(
        f"deploy checks OK (gemini key present, index files in {cfg.index_npy_path.parent})",
        flush=True,
    )
    os.environ.setdefault("PHOENIX_RUNTIME", "1")

    if len(sys.argv) <= 1:
        return

    os.execvpe(sys.argv[1], sys.argv[1:], os.environ)


if __name__ == "__main__":
    main()
