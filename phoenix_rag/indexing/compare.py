"""Embedding model bake-off on a Hebrew subset (spec 6.4: compare at least two candidates on retrieval hit rate).
Takes every chunk of the probe's expected pages plus random distractor chunks, embeds the subset with each
candidate model (through the on-disk cache), and reports hit rate and MRR for the probe questions, semantic-only."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import yaml

from ..config import CONFIG_DIR, Config
from .chunk import Chunk, read_chunks
from .embed import Embedder
from .embed_cache import EmbeddingCache

log = logging.getLogger(__name__)


@dataclass
class ModelResult:
    model: str
    dim: int
    hits: int
    mrr: float
    n: int
    ranks: list[int | None]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def build_subset(cfg: Config, expects: list[str], distractors: int, seed: int = 7) -> list[Chunk]:
    chunks = list(read_chunks(cfg.chunks_path))
    wanted = [c for c in chunks if any(e.rstrip("/") in c.url for e in expects)]
    wanted_ids = {c.chunk_id for c in wanted}
    rest = [c for c in chunks if c.chunk_id not in wanted_ids]
    random.Random(seed).shuffle(rest)
    subset = wanted + rest[:distractors]
    log.info("subset: %d chunks from %d expected pages + %d distractors", len(subset), len(expects), min(distractors, len(rest)))
    return subset


def evaluate_model(cfg: Config, model: str, dim: int, subset: list[Chunk], questions: list[dict], k: int) -> ModelResult:
    """`model` is a Gemini model name, or `st:<path-or-hub-id>` for a local sentence-transformers model."""
    emb = cfg.embedding
    if model.startswith("st:"):
        embedder = Embedder(model[3:], provider="sentence-transformers", batch_size=int(emb.get("batch_size", 32)))
    else:
        embedder = Embedder(
            model, provider="google", batch_size=int(emb.get("batch_size", 32)), dim=dim,
            tokens_per_minute=int(emb.get("tokens_per_minute") or 0), max_batch_tokens=int(emb.get("max_batch_tokens") or 0),
        )
    cache_rel = emb.get("cache_path") or "data/embeddings_cache.sqlite"
    cache = EmbeddingCache(cfg.root / cache_rel, embedder.model_name, embedder.dim)
    texts = [c.embed_text for c in subset]
    have = cache.get_many(texts)
    todo = [i for i, t in enumerate(texts) if cache.key(t) not in have]
    log.info("%s: %d cached, %d to embed", model, len(texts) - len(todo), len(todo))
    for g in range(0, len(todo), 200):
        idx = todo[g : g + 200]
        vecs = embedder.encode_passages([texts[i] for i in idx], token_counts=[int(subset[i].token_count) for i in idx])
        log.info("%s: embedded %d/%d", model, min(g + 200, len(todo)), len(todo))
        cache.put_many([texts[i] for i in idx], vecs)
        for i, v in zip(idx, vecs):
            have[cache.key(texts[i])] = v
    vectors = [have[cache.key(t)] for t in texts]
    cache.close()

    qvecs = embedder.encode_queries([q["q"] for q in questions])
    hits, rr, ranks = 0, 0.0, []
    for q, qv in zip(questions, qvecs):
        order = sorted(range(len(subset)), key=lambda i: _dot(qv, vectors[i]), reverse=True)
        # Rank of the first chunk from the expected page (page-level hit, like the probe).
        rank = next((r for r, i in enumerate(order[:k], 1) if q["expect"].rstrip("/") in subset[i].url), None)
        ranks.append(rank)
        if rank:
            hits += 1
            rr += 1.0 / rank
    return ModelResult(model=model, dim=embedder.dim, hits=hits, mrr=rr / len(questions), n=len(questions), ranks=ranks)


def run_comparison(cfg: Config, models: list[str], *, distractors: int = 400, k: int = 5) -> list[ModelResult]:
    spec = yaml.safe_load((CONFIG_DIR / "probe.yaml").read_text(encoding="utf-8"))
    questions = spec["questions"]
    subset = build_subset(cfg, [q["expect"] for q in questions], distractors)
    dim = int(cfg.embedding.get("output_dimensionality") or 0)
    results = []
    for m in models:
        res = evaluate_model(cfg, m, dim, subset, questions, k)
        results.append(res)
        print(f"{m:32s} dim={res.dim:<5d} hit@{k}={res.hits}/{res.n}  MRR={res.mrr:.3f}  ranks={res.ranks}")
    return results
