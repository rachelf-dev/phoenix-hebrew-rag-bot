# Expand LLM context with in-order chunks from the same short claim page (decision trees).
"""When the top hit is a small multi-chunk page, prefer full page order over mixed URLs."""

from __future__ import annotations

from ..indexing.chunk import ntokens
from ..indexing.retrieve import Hit, _hit_from_stored
from ..retrieval.numpy_retriever import NumpyRetriever

_DEFAULT_TOKEN_BUDGET = 2500
_SHORT_PAGE_MAX_CHUNKS = 14


def _page_url(hit: Hit) -> str:
    return (hit.display_url or hit.url or "").rstrip("/")


def expand_same_page_hits(
    hits: list[Hit],
    index: NumpyRetriever,
    *,
    token_budget: int = _DEFAULT_TOKEN_BUDGET,
    short_page_max_chunks: int = _SHORT_PAGE_MAX_CHUNKS,
) -> list[Hit]:
    """Replace the LLM chunk list with same-page chunks in index order when the page is small."""
    if not hits:
        return hits
    top = hits[0]
    url = _page_url(top)
    if not url:
        return hits

    page_rows = index.sorted_chunks_for_url(url)
    if len(page_rows) <= 1:
        return hits

    total_tokens = sum(ntokens(c.original_text) for c in page_rows)
    if len(page_rows) > short_page_max_chunks and total_tokens > token_budget:
        return hits

    by_id = {h.chunk_id: h for h in hits}
    base_score = float(top.score or 0.0)
    expanded: list[Hit] = []
    used = 0
    for row in page_rows:
        tok = ntokens(row.original_text)
        if used + tok > token_budget and expanded:
            break
        used += tok
        if row.chunk_id in by_id:
            expanded.append(by_id[row.chunk_id])
        else:
            expanded.append(_hit_from_stored(row, base_score * 0.95))

    return expanded if len(expanded) > 1 else hits
