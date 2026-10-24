# Retrieval-only eval: dense vs BM25 vs hybrid RRF, no LLM and no Cohere.
# Caches query embeddings in SQLite so a second run costs zero Gemini calls.
"""Usage: python scripts/eval_retrieval.py [--limit N]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phoenix_rag.answering.context import retrieval_query
from phoenix_rag.answering.eval_set import load_questions
from phoenix_rag.config import load_config
from phoenix_rag.indexing.embed import Embedder
from phoenix_rag.indexing.embed_cache import EmbeddingCache
from phoenix_rag.indexing.retrieve import rrf_fuse
from phoenix_rag.retrieval.numpy_retriever import NumpyRetriever

FOCUS = ("f03", "f04", "f06", "d10", "d11")
CUTS = (5, 10, 20)
DEEP_K = 200


def _search_text(q) -> str:
    """Heuristic retrieval query only — no Gemini rewrite."""
    return retrieval_query(q.question, q.history or None)


def _url_of(index: NumpyRetriever, chunk_id: str) -> str:
    stored = index.chunk_by_id(chunk_id)
    if stored is None:
        return ""
    return stored.display_url or stored.url or ""


def first_url_rank(pairs: list[tuple[str, float]], index: NumpyRetriever, needles: list[str]) -> int | None:
    """1-based rank of the first chunk whose URL contains any expected needle."""
    if not needles:
        return None
    for i, (cid, _score) in enumerate(pairs, 1):
        url = _url_of(index, cid)
        if any(n in url for n in needles):
            return i
    return None


def first_url_rank_deduped(
    pairs: list[tuple[str, float]],
    index: NumpyRetriever,
    needles: list[str],
    *,
    max_per_url: int = 2,
) -> int | None:
    """1-based URL rank after the same max_per_url=2 cap the server uses."""
    if not needles:
        return None
    seen: dict[str, int] = {}
    url_rank = 0
    for cid, _score in pairs:
        stored = index.chunk_by_id(cid)
        if stored is None:
            continue
        url = stored.display_url or stored.url or ""
        key = url.rstrip("/")
        if key and seen.get(key, 0) >= max_per_url:
            continue
        if key:
            seen[key] = seen.get(key, 0) + 1
        url_rank += 1
        if any(n in url for n in needles):
            return url_rank
    return None


def _marks(rank: int | None) -> str:
    parts = []
    for cut in CUTS:
        parts.append(f"@{cut}={'Y' if rank is not None and rank <= cut else 'n'}")
    return " ".join(parts)


def _rank_s(rank: int | None) -> str:
    return str(rank) if rank is not None else "miss"


def cached_query_vec(embedder: Embedder, cache: EmbeddingCache, text: str) -> list[float]:
    have = cache.get_many([text])
    key = cache.key(text)
    if key in have:
        return have[key]
    vec = embedder.encode_query(text)
    cache.put_many([text], [vec])
    return vec


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval-only eval (dense / BM25 / hybrid). No LLM, no Cohere.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-hidden", action="store_true")
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="SQLite cache path (default: data/query_embed_cache.sqlite)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    qpath = args.questions or (cfg.root / "eval" / "questions.jsonl")
    if not qpath.is_absolute():
        qpath = cfg.root / qpath
    questions = load_questions(qpath, include_hidden=not args.no_hidden)
    if args.limit:
        questions = questions[: args.limit]

    scored = [q for q in questions if q.expected_url_contains]
    print(f"questions={len(questions)} with_expected_url={len(scored)}")
    print(f"index={cfg.index_npy_path}  pool_prod=40  deep_k={DEEP_K}")
    print()

    index = NumpyRetriever.open(cfg)
    embedder = Embedder.from_config(cfg)
    cache_path = args.cache or (cfg.data_dir / "query_embed_cache.sqlite")
    cache = EmbeddingCache(cache_path, embedder.model_name, embedder.dim)
    print(f"query-embed cache {cache_path}  already={cache.count()}  dim={embedder.dim}  task={embedder.task_query}")

    pool = max(int(cfg.retrieval.get("top_k") or 12) * 2, 40)
    max_per_url = int(cfg.retrieval.get("max_per_url") or 2)

    tallies = {
        "dense": {c: 0 for c in CUTS},
        "bm25": {c: 0 for c in CUTS},
        "hybrid": {c: 0 for c in CUTS},
    }
    focus_rows: list[str] = []
    n = 0

    for q in scored:
        text = _search_text(q)
        qvec = cached_query_vec(embedder, cache, text)
        dense = index.semantic_query(qvec, DEEP_K)
        bm25 = index.lexical_query(text, DEEP_K)
        fused = rrf_fuse(
            [[cid for cid, _ in dense[:pool]], [cid for cid, _ in bm25[:pool]]],
        )
        hybrid = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        d_rank = first_url_rank(dense, index, q.expected_url_contains)
        b_rank = first_url_rank(bm25, index, q.expected_url_contains)
        h_rank = first_url_rank(hybrid, index, q.expected_url_contains)
        h_url = first_url_rank_deduped(hybrid, index, q.expected_url_contains, max_per_url=max_per_url)

        n += 1
        for cut in CUTS:
            if d_rank is not None and d_rank <= cut:
                tallies["dense"][cut] += 1
            if b_rank is not None and b_rank <= cut:
                tallies["bm25"][cut] += 1
            if h_rank is not None and h_rank <= cut:
                tallies["hybrid"][cut] += 1

        line = (
            f"{q.id:4s} dense={_rank_s(d_rank):>4} {_marks(d_rank)} | "
            f"bm25={_rank_s(b_rank):>4} {_marks(b_rank)} | "
            f"hybrid={_rank_s(h_rank):>4} {_marks(h_rank)} "
            f"(url-dedupe@{max_per_url}={_rank_s(h_url)})"
        )
        hid = " [hidden]" if q.hidden else ""
        print(f"{line}{hid}  {q.question[:70]}")
        if q.id in FOCUS:
            top_h = []
            for cid, score in hybrid[:8]:
                top_h.append(f"    {score:.4f}  {_url_of(index, cid)}")
            focus_rows.append(
                f"\n=== {q.id}  expect={q.expected_url_contains}\n"
                f"search_q: {text}\n"
                f"{line}\n"
                f"hybrid top-8:\n" + "\n".join(top_h)
            )

    cache.close()
    print()
    print(f"hit-rate over {n} questions with expected URLs (chunk rank, no Cohere):")
    for name in ("dense", "bm25", "hybrid"):
        bits = "  ".join(f"@{c}={tallies[name][c]}/{n}={tallies[name][c] / n:.0%}" for c in CUTS)
        print(f"  {name:7s}  {bits}")
    print()
    print("Focus failures from the full eval (ranks of expected URL):")
    print("".join(focus_rows) if focus_rows else "  (none of f03/f04/f06/d10/d11 in this run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
