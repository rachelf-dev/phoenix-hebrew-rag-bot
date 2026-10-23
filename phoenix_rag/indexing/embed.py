"""Gemini embedding wrapper for query-time (and offline index build). Rotates GEMINI_API_KEY* on 429,
TPM throttle, retryDelay, and daily-quota detection.

Runtime query embedding is one embed_content call per key. Document embedding stays on the offline index build.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import deque
from typing import Sequence

from ..config import Config
from ..gemini_keys import gemini_client, gemini_keys

log = logging.getLogger(__name__)

_RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "timeout", "Timeout")


class QueryEmbeddingRateLimited(RuntimeError):
    """Every configured Gemini key returned 429/RESOURCE_EXHAUSTED for this query."""


class Embedder:
    def __init__(self, model_name: str, *, provider: str, batch_size: int = 32, dim: int = 0,
                 pause_seconds: float = 0.0, task_document: str = "RETRIEVAL_DOCUMENT",
                 task_query: str = "RETRIEVAL_QUERY", tokens_per_minute: int = 0, max_batch_tokens: int = 0):
        if provider != "google":
            raise ValueError(f"unknown embedding provider: {provider} (only google is supported)")
        self.model_name = model_name
        self.provider = provider
        self.batch_size = batch_size
        self.pause_seconds = pause_seconds
        self.task_document = task_document
        self.task_query = task_query
        self.dim = dim
        self.tokens_per_minute = tokens_per_minute
        self.max_batch_tokens = max_batch_tokens
        self._window: deque[tuple[float, int]] = deque()  # (sent_at, tokens) for the sliding 60s budget
        from google.genai import types

        self._keys = gemini_keys()
        if not self._keys:
            raise RuntimeError("GEMINI_API_KEY is missing; copy .env.example to .env")
        self._key_i = 0
        self._vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")
        self._types = types
        self._client = gemini_client(self._keys[self._key_i])
        if not self.dim:
            self.dim = int(os.getenv("EMBEDDING_DIM") or 768)
        self._base_url = _client_base_url(self._client)
        log.info(
            "Gemini embedder %s dim=%d batch=%d tpm=%d batch_tokens=%d keys=%d mode=%s endpoint=%s",
            model_name, self.dim, batch_size, tokens_per_minute, max_batch_tokens, len(self._keys),
            "vertex-express" if self._vertex else "gemini-api", self._base_url,
        )

    def _bind_key(self, ki: int) -> None:
        self._key_i = ki % len(self._keys)
        self._client = gemini_client(self._keys[self._key_i])
        self._base_url = _client_base_url(self._client)

    def _advance_key(self) -> None:
        """Round-robin: point the next request at the following API key."""
        self._bind_key(self._key_i + 1)

    @classmethod
    def from_config(cls, cfg: Config) -> "Embedder":
        emb = cfg.embedding
        return cls(
            emb["model"],
            provider=emb.get("provider", "google"),
            batch_size=int(emb.get("batch_size", 32)),
            dim=int(emb.get("output_dimensionality") or 0),
            pause_seconds=float(emb.get("pause_seconds") or 0),
            task_document=emb.get("task_document", "RETRIEVAL_DOCUMENT"),
            task_query=emb.get("task_query", "RETRIEVAL_QUERY"),
            tokens_per_minute=int(emb.get("tokens_per_minute") or 0),
            max_batch_tokens=int(emb.get("max_batch_tokens") or 0),
        )

    def encode_passages(self, texts: Sequence[str], token_counts: Sequence[int] | None = None) -> list[list[float]]:
        # The web server sets this so a bug cannot re-embed the corpus on boot or per question.
        if os.getenv("PHOENIX_RUNTIME", "").lower() in ("1", "true", "yes"):
            raise RuntimeError(
                "refusing to embed documents at runtime; load index_artifacts/ and embed the query only"
            )
        return self._encode_gemini(texts, self.task_document, token_counts)

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Offline/batch query embedding (probe compare). The live server uses encode_query."""
        return self._encode_gemini(texts, self.task_query, None)

    def encode_query(self, text: str) -> list[float]:
        """Embed one user question. One API call per key, then stop. No document texts, no backoff loop."""
        cfg = self._types.EmbedContentConfig(
            task_type=self.task_query,
            output_dimensionality=self.dim if self.dim else None,
        )
        nkeys = len(self._keys)
        start = self._key_i % nkeys
        key_order = list(range(start, nkeys)) + list(range(0, start))
        last: Exception | None = None
        saw_429 = False
        for ki in key_order:
            self._bind_key(ki)
            try:
                resp = self._client.models.embed_content(
                    model=self.model_name, contents=[text], config=cfg,
                )
                embeddings = list(resp.embeddings or [])
                if not embeddings:
                    raise RuntimeError("Gemini returned no query embedding")
                if nkeys > 1:
                    self._advance_key()
                return _l2(list(embeddings[0].values))
            except Exception as exc:
                last = exc
                msg = str(exc)
                if "blockByNetFree" in msg or "418" in msg:
                    raise RuntimeError(
                        f"NetFree blocked the embedding request: {self.model_name} -> {self._base_url} "
                        f"(HTTP 418 Blocked by NetFree). Original error: {msg[:300]}"
                    ) from exc
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    saw_429 = True
                    log.warning("query embed 429 on key %d/%d", ki + 1, nkeys)
                    continue
                raise
        if saw_429:
            raise QueryEmbeddingRateLimited(
                f"query embedding 429 on all {nkeys} key(s)"
            ) from last
        raise RuntimeError(f"query embedding failed: {last}") from last

    def _encode_gemini(self, texts: Sequence[str], task: str,
                       token_counts: Sequence[int] | None) -> list[list[float]]:
        counts = list(token_counts) if token_counts is not None else [_estimate_tokens(t) for t in texts]
        out: list[list[float]] = []
        total = len(texts)
        done = 0
        for start, end in self._batches(counts):
            batch = list(texts[start:end])
            batch_tokens = sum(counts[start:end])
            self._throttle(batch_tokens)
            vecs = self._gemini_batch(batch, task)
            self._window.append((time.monotonic(), batch_tokens))
            out.extend(vecs)
            done += len(batch)
            if done == total or (done // max(self.batch_size, 1)) % 10 == 0:
                log.info("embedded %d/%d", done, total)
            if self.pause_seconds:
                time.sleep(self.pause_seconds)
        return out

    def _batches(self, counts: list[int]) -> list[tuple[int, int]]:
        """Split indices into batches bounded by batch_size and (if set) max_batch_tokens."""
        spans: list[tuple[int, int]] = []
        start, tokens = 0, 0
        for i, n in enumerate(counts):
            too_many = (i - start) >= self.batch_size
            too_big = self.max_batch_tokens and tokens + n > self.max_batch_tokens and i > start
            if too_many or too_big:
                spans.append((start, i))
                start, tokens = i, 0
            tokens += n
        if start < len(counts):
            spans.append((start, len(counts)))
        return spans

    def _throttle(self, next_tokens: int) -> None:
        """Sleep until the sliding 60s window has room for next_tokens (no-op if tokens_per_minute == 0)."""
        if not self.tokens_per_minute:
            return
        while True:
            now = time.monotonic()
            while self._window and now - self._window[0][0] >= 60:
                self._window.popleft()
            used = sum(t for _, t in self._window)
            if used + next_tokens <= self.tokens_per_minute or not self._window:
                return
            wait = 60 - (now - self._window[0][0]) + 0.2
            log.debug("tpm throttle: used=%d next=%d sleeping %.1fs", used, next_tokens, wait)
            time.sleep(max(wait, 0.2))

    def _gemini_batch(self, batch: list[str], task: str, retries: int = 8) -> list[list[float]]:
        cfg = self._types.EmbedContentConfig(
            task_type=task,
            output_dimensionality=self.dim if self.dim else None,
        )
        nkeys = len(self._keys)
        start = self._key_i % nkeys
        key_order = list(range(start, nkeys)) + list(range(0, start))
        last: Exception | None = None

        for ki in key_order:
            self._bind_key(ki)
            delay = 2.0
            for attempt in range(1, retries + 1):
                try:
                    resp = self._client.models.embed_content(model=self.model_name, contents=batch, config=cfg)
                    if nkeys > 1:
                        self._advance_key()
                    return [_l2(list(e.values)) for e in resp.embeddings]
                except Exception as exc:
                    last = exc
                    msg = str(exc)
                    if "blockByNetFree" in msg or "418" in msg:
                        raise RuntimeError(
                            f"NetFree blocked the embedding request: {self.model_name} -> {self._base_url} "
                            f"(HTTP 418 Blocked by NetFree). Original error: {msg[:300]}"
                        ) from exc
                    if not any(s in msg for s in _RETRYABLE):
                        raise
                    server_delay = _retry_delay(msg)
                    daily = server_delay > MAX_RETRY_WAIT_SECONDS
                    if daily and nkeys > 1 and ki != key_order[-1]:
                        log.warning(
                            "embed daily quota on key %d/%d for %s; rotating key",
                            ki + 1, nkeys, self.model_name,
                        )
                        break
                    if daily:
                        raise RuntimeError(
                            f"Gemini daily embedding quota exhausted for {self.model_name} "
                            f"(tried {nkeys} key(s)); server asks to retry in "
                            f"{server_delay / 60:.0f} min. Re-run `index` later (cached progress is kept) or enable "
                            f"billing. Detail: {_quota_summary(msg)}"
                        ) from exc
                    if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and nkeys > 1:
                        log.warning(
                            "embed 429 on key %d/%d attempt %d/%d; rotating key (%s)",
                            ki + 1, nkeys, attempt, retries, msg[:100],
                        )
                        self._window.append((time.monotonic(), self.tokens_per_minute or 0))
                        break
                    sleep_for = max(delay, server_delay)
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                        self._window.append((time.monotonic(), self.tokens_per_minute or 0))
                    log.warning(
                        "embed retry key %d/%d %d/%d after %s (sleep %.1fs%s)",
                        ki + 1, nkeys, attempt, retries, msg[:120], sleep_for,
                        f", server asked {server_delay:.0f}s" if server_delay else "",
                    )
                    time.sleep(sleep_for)
                    delay = min(delay * 2, 60)
                    continue
        raise RuntimeError(f"Gemini embeddings failed after retries: {last}") from last


MAX_RETRY_WAIT_SECONDS = 15 * 60  # longer server-requested waits are treated as "quota gone for today"


def _quota_summary(msg: str) -> str:
    """Compact 'metric=..., limit=...' from a google-genai 429 message, for the operator."""
    metric = re.search(r"Quota exceeded for metric: ([^,]+), limit: (\d+)", msg)
    return f"{metric.group(1)} limit={metric.group(2)}" if metric else msg[:160]


def _retry_delay(msg: str) -> float:
    """Pull the server-suggested wait out of a google-genai error string (retryDelay '12s' / 'retry in 12.3s')."""
    m = re.search(r"retryDelay'?\"?:\s*'?\"?(\d+(?:\.\d+)?)s", msg) or re.search(r"retry in (\d+(?:\.\d+)?)s", msg)
    return float(m.group(1)) if m else 0.0


def _estimate_tokens(text: str) -> int:
    """cl100k count via the chunker's encoder (conservative for Hebrew vs Gemini's tokenizer)."""
    from .chunk import ntokens

    return ntokens(text)


def _client_base_url(client) -> str:
    """Best-effort: the host google-genai will call (for logs and block diagnostics)."""
    try:
        return str(client._api_client._http_options.base_url)
    except Exception:
        return "(unknown)"


def _l2(vec: list[float]) -> list[float]:
    s = sum(x * x for x in vec) ** 0.5
    if s == 0:
        return vec
    return [x / s for x in vec]
