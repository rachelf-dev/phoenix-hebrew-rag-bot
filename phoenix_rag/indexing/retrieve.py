"""Hybrid retrieval: semantic (Chroma) + BM25, fused with Reciprocal Rank Fusion.

Uses a deeper candidate pool than top_k and keeps at most two chunks per URL so
self-service action pages around rank 20 still reach the reranker. Pool size
follows retrieval.rerank.max_candidates from config.yaml."""

from __future__ import annotations

from dataclasses import dataclass

from .embed import Embedder
from .hebrew_lex import tokenize
from .store import VectorIndex, load_bm25


@dataclass
class Hit:
    chunk_id: str
    url: str
    page_title: str
    heading_path: str
    text: str
    score: float
    section_type: str
    form_links: str
    display_url: str = ""
    rerank_score: float = 0.0


_BM25_CACHE: dict[str, tuple[float, tuple]] = {}


def _bm25_cached(path):
    """Rebuilding BM25Okapi over ~14k docs takes seconds; keep one copy per file mtime."""
    mtime = path.stat().st_mtime
    hit = _BM25_CACHE.get(str(path))
    if hit is None or hit[0] != mtime:
        hit = (mtime, load_bm25(path))
        _BM25_CACHE[str(path)] = hit
    return hit[1]


def rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for i, cid in enumerate(ranks):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + i + 1)
    return scores


def search(index: VectorIndex, embedder: Embedder, question: str, *, top_k: int = 12,
           hybrid: bool = True) -> list[Hit]:
    qvec = embedder.encode_queries([question])[0]
    pool = max(top_k * 2, 40)
    sem = index.query(qvec, k=pool)
    sem_ids = sem["ids"][0] if sem["ids"] else []
    by_id: dict[str, tuple[str, dict]] = {}
    for i, cid in enumerate(sem_ids):
        by_id[cid] = (sem["documents"][0][i], sem["metadatas"][0][i])

    if hybrid and index.bm25_path.exists():
        bm25, bm_ids, _urls = _bm25_cached(index.bm25_path)
        scores = bm25.get_scores(tokenize(question))
        order = sorted(range(len(bm_ids)), key=lambda i: scores[i], reverse=True)[:pool]
        bm_rank = [bm_ids[i] for i in order]
        fused = rrf_fuse([sem_ids, bm_rank])
    else:
        fused = {cid: 1.0 / (i + 1) for i, cid in enumerate(sem_ids)}

    # Fetch any BM25-only ids that aren't in the semantic payload.
    missing = [cid for cid in fused if cid not in by_id]
    if missing:
        got = index.collection.get(ids=missing[:80], include=["documents", "metadatas"])
        for i, cid in enumerate(got["ids"]):
            by_id[cid] = (got["documents"][i], got["metadatas"][i])

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    hits: list[Hit] = []
    per_url: dict[str, int] = {}
    for cid, score in ranked:
        if cid not in by_id:
            continue
        doc, meta = by_id[cid]
        url = meta.get("display_url", "") or meta.get("url", "")
        key = (url or "").rstrip("/")
        if key and per_url.get(key, 0) >= 2:
            continue
        if key:
            per_url[key] = per_url.get(key, 0) + 1
        hits.append(
            Hit(
                chunk_id=cid,
                url=meta.get("url", ""),
                page_title=meta.get("page_title", ""),
                heading_path=meta.get("heading_path", ""),
                text=doc,
                score=score,
                section_type=meta.get("section_type", ""),
                form_links=meta.get("form_links", ""),
                display_url=url,
            )
        )
        if len(hits) >= top_k:
            break
    return hits
