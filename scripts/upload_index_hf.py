#!/usr/bin/env python3
# Upload index.npy, index_chunks.json, and bm25.pkl to a Hugging Face dataset repo.
"""Requires: pip install huggingface_hub  and  HF_TOKEN or huggingface-cli login."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload RAG index files to Hugging Face Hub (dataset).")
    parser.add_argument("repo_id", help="owner/dataset-name, e.g. myuser/phoenix-rag-index")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"))
    parser.add_argument("--private", action="store_true", help="create/use a private dataset")
    args = parser.parse_args()

    if not args.token:
        print("Set HF_TOKEN or pass --token (Hugging Face → Settings → Access Tokens, Write)", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("pip install huggingface_hub", file=sys.stderr)
        return 1

    from phoenix_rag.config import load_config

    cfg = load_config()
    files = [
        ("index.npy", cfg.index_npy_path),
        ("index_chunks.json", cfg.index_chunks_path),
        ("bm25.pkl", cfg.bm25_path),
    ]
    for name, path in files:
        if not path.is_file():
            print(f"missing {path} — run scripts/build_index_local.py first", file=sys.stderr)
            return 1

    api = HfApi(token=args.token)
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True, private=args.private)
    for name, path in files:
        print(f"uploading {path.name} -> {args.repo_id}/{name}")
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=args.repo_id, repo_type="dataset")
    print("")
    print("Set on Render:")
    print(f"  INDEX_HF_REPO={args.repo_id}")
    print("  (public dataset: no token needed for download)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
