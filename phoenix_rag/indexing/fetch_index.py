# Download pre-built index.npy and index_chunks.json when missing (Render cold start).
"""URLs from INDEX_NPY_URL / INDEX_CHUNKS_URL, or Hugging Face INDEX_HF_REPO + optional INDEX_HF_REVISION."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


def _hf_resolve_url(repo: str, filename: str, revision: str) -> str:
    repo = repo.strip().strip("/")
    if "/" not in repo:
        raise ValueError("INDEX_HF_REPO must be owner/dataset-name")
    rev = (revision or "main").strip()
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{filename}"


def _download(url: str, dest: Path, *, timeout: float = 600.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s -> %s", url, dest)
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                fh.write(chunk)
    tmp.replace(dest)
    log.info("saved %s (%.1f MB)", dest.name, dest.stat().st_size / (1024 * 1024))


def ensure_index_files(
    index_npy: Path,
    index_chunks: Path,
    *,
    bm25_path: Path | None = None,
    index_npy_url: str | None = None,
    index_chunks_url: str | None = None,
    bm25_url: str | None = None,
    hf_repo: str | None = None,
    hf_revision: str | None = None,
) -> None:
    """Ensure index files exist locally; download from configured URLs if not."""
    need_bm25 = bm25_path is not None
    if index_npy.is_file() and index_chunks.is_file() and (not need_bm25 or bm25_path.is_file()):
        return

    npy_url = (index_npy_url or os.getenv("INDEX_NPY_URL") or "").strip()
    chunks_url = (index_chunks_url or os.getenv("INDEX_CHUNKS_URL") or "").strip()
    bm25_u = (bm25_url or os.getenv("INDEX_BM25_URL") or "").strip()
    repo = (hf_repo or os.getenv("INDEX_HF_REPO") or "").strip()
    revision = (hf_revision or os.getenv("INDEX_HF_REVISION") or "main").strip()

    if repo:
        if not npy_url:
            npy_url = _hf_resolve_url(repo, "index.npy", revision)
        if not chunks_url:
            chunks_url = _hf_resolve_url(repo, "index_chunks.json", revision)
        if need_bm25 and not bm25_u:
            bm25_u = _hf_resolve_url(repo, "bm25.pkl", revision)

    if not index_npy.is_file():
        if not npy_url:
            raise FileNotFoundError(
                f"missing {index_npy}; set INDEX_NPY_URL or INDEX_HF_REPO (and upload index.npy)"
            )
        _download(npy_url, index_npy)

    if not index_chunks.is_file():
        if not chunks_url:
            raise FileNotFoundError(
                f"missing {index_chunks}; set INDEX_CHUNKS_URL or INDEX_HF_REPO (and upload index_chunks.json)"
            )
        _download(chunks_url, index_chunks)

    if need_bm25 and bm25_path is not None and not bm25_path.is_file():
        if not bm25_u:
            raise FileNotFoundError(
                f"missing {bm25_path}; set INDEX_BM25_URL or INDEX_HF_REPO (and upload bm25.pkl)"
            )
        _download(bm25_u, bm25_path)
