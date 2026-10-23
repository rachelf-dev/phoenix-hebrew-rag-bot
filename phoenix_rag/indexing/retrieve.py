"""Hybrid retrieval: semantic (NumPy) + BM25, fused with Reciprocal Rank Fusion.

Uses a deeper candidate pool than top_k and keeps at most two chunks per URL so
self-service action pages around rank 20 still reach the reranker. Pool size
follows retrieval.rerank.max_candidates from config.yaml (40 by default).

If the query embedding hits 429 on every Gemini key, search falls back to BM25 only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .embed import Embedder, QueryEmbeddingRateLimited
from ..retrieval.numpy_retriever import NumpyRetriever

log = logging.getLogger(__name__)


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


def rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for i, cid in enumerate(ranks):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + i + 1)
    return scores


def _hit_from_stored(stored, score: float) -> Hit:
    return Hit(
        chunk_id=stored.chunk_id,
        url=stored.url,
        page_title=stored.page_title,
        heading_path=stored.heading_path,
        text=stored.original_text,
        score=score,
        section_type=stored.section_type,
        form_links=stored.form_links,
        display_url=stored.display_url or stored.url,
    )


def _hits_from_ranked(
    index: NumpyRetriever,
    ranked: list[tuple[str, float]],
    *,
    top_k: int,
    threshold: float,
    max_per_url: int,
) -> list[Hit]:
    hits: list[Hit] = []
    per_url: dict[str, int] = {}
    for cid, score in ranked:
        stored = index.chunk_by_id(cid)
        if stored is None:
            continue
        url = stored.display_url or stored.url
        key = (url or "").rstrip("/")
        if threshold > 0 and score < threshold:
            continue
        if key and per_url.get(key, 0) >= max_per_url:
            continue
        if key:
            per_url[key] = per_url.get(key, 0) + 1
        hits.append(_hit_from_stored(stored, score))
        if len(hits) >= top_k:
            break
    return hits


def search(
    index: NumpyRetriever,
    embedder: Embedder,
    question: str,
    *,
    top_k: int = 12,
    hybrid: bool = True,
    threshold: float = 0.0,
    max_per_url: int = 2,
) -> list[Hit]:
    """Hybrid search; `threshold` is an RRF floor (0.0 until Day-4 calibration)."""
    pool = max(top_k * 2, 40)
    max_per_url = max(1, int(max_per_url))

    try:
        qvec = embedder.encode_query(question)
    except QueryEmbeddingRateLimited:
        log.warning("query embedding 429 on all keys; falling back to BM25-only retrieval")
        ranked = index.lexical_query(question, pool)
        return _hits_from_ranked(
            index, ranked, top_k=top_k, threshold=threshold, max_per_url=max_per_url,
        )

    sem_pairs = index.semantic_query(qvec, k=pool)
    sem_ids = [cid for cid, _ in sem_pairs]
    sem_scores = {cid: sc for cid, sc in sem_pairs}

    if hybrid and index.has_bm25:
        bm_rank = [cid for cid, _ in index.lexical_query(question, pool)]
        fused = rrf_fuse([sem_ids, bm_rank])
    else:
        fused = {cid: sem_scores.get(cid, 1.0 / (i + 1)) for i, cid in enumerate(sem_ids)}

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return _hits_from_ranked(
        index, ranked, top_k=top_k, threshold=threshold, max_per_url=max_per_url,
    )
