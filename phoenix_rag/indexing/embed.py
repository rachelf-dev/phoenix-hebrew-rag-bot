"""Embedding wrapper. Gemini (API or Vertex express via GOOGLE_GENAI_USE_VERTEXAI=1) is paced by a
tokens-per-minute budget, honours 429 retryDelay, and stops cleanly on a daily quota. Local
sentence-transformers loads offline from models/ (e5 query:/passage: prefixes are added automatically)."""

from __future__ import annotations

import logging
import os
import re
import time
from collections import deque
from typing import Sequence

from ..config import Config

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str, *, provider: str, batch_size: int = 32, dim: int = 0,
                 pause_seconds: float = 0.0, task_document: str = "RETRIEVAL_DOCUMENT",
                 task_query: str = "RETRIEVAL_QUERY", tokens_per_minute: int = 0, max_batch_tokens: int = 0):
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
        if provider == "google":
            from google import genai
            from google.genai import types

            key = os.getenv("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is missing; copy .env.example to .env")
            # Two Google gateways: Gemini API (generativelanguage.googleapis.com, default) and Vertex AI
            # express mode (aiplatform.googleapis.com). An "AQ." key can be bound to either, so the key
            # prefix does not decide; opt into Vertex with GOOGLE_GENAI_USE_VERTEXAI=1 (the SDK's own env var).
            self._vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")
            self._client = genai.Client(vertexai=True, api_key=key) if self._vertex else genai.Client(api_key=key)
            self._types = types
            if not self.dim:
                self.dim = int(os.getenv("EMBEDDING_DIM") or 768)
            self._base_url = _client_base_url(self._client)
            log.info("Gemini embedder %s dim=%d batch=%d tpm=%d batch_tokens=%d mode=%s endpoint=%s", model_name,
                     self.dim, batch_size, tokens_per_minute, max_batch_tokens,
                     "vertex-express" if self._vertex else "gemini-api", self._base_url)
        elif provider == "sentence-transformers":
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            # Local weights under models/; stay offline so sentence-transformers does not probe huggingface.co.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(model_name)
            dim_fn = getattr(self._st, "get_embedding_dimension", None) or self._st.get_sentence_embedding_dimension
            self.dim = int(dim_fn())
            self._is_e5 = "e5" in model_name.lower()
            log.info("local embedder %s dim=%d", model_name, self.dim)
        else:
            raise ValueError(f"unknown embedding provider: {provider}")

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
        return self._encode(texts, self.task_document, kind="passage", token_counts=token_counts)

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts, self.task_query, kind="query")

    def _encode(self, texts: Sequence[str], task: str, *, kind: str,
                token_counts: Sequence[int] | None = None) -> list[list[float]]:
        if self.provider == "sentence-transformers":
            prepared = list(texts)
            if getattr(self, "_is_e5", False):
                prefix = "query: " if kind == "query" else "passage: "
                prepared = [prefix + t for t in texts]
            vecs = self._st.encode(prepared, batch_size=self.batch_size,
                                   show_progress_bar=len(texts) > 32, normalize_embeddings=True)
            return [v.tolist() for v in vecs]
        return self._encode_gemini(texts, task, token_counts)

    # ------------------------------------------------------------------ gemini
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
        delay = 2.0
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                resp = self._client.models.embed_content(model=self.model_name, contents=batch, config=cfg)
                return [_l2(list(e.values)) for e in resp.embeddings]
            except Exception as exc:
                last = exc
                msg = str(exc)
                if "blockByNetFree" in msg or "418" in msg:
                    raise RuntimeError(
                        f"NetFree blocked the embedding request: {self.model_name} -> {self._base_url} "
                        f"(HTTP 418 Blocked by NetFree). Original error: {msg[:300]}"
                    ) from exc
                if any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "timeout", "Timeout")):
                    server_delay = _retry_delay(msg)
                    if server_delay > MAX_RETRY_WAIT_SECONDS:
                        # A multi-hour retryDelay means a *daily* quota (free tier: 1000 embedded texts/day/model).
                        # Stop now; progress is in the embedding cache and `index` resumes where it left off.
                        raise RuntimeError(
                            f"Gemini daily embedding quota exhausted for {self.model_name}; server asks to retry in "
                            f"{server_delay / 60:.0f} min. Re-run `index` later (cached progress is kept), enable billing, "
                            f"or switch embedding.provider to a local model. Detail: {_quota_summary(msg)}"
                        ) from exc
                    sleep_for = max(delay, server_delay)
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                        # A 429 means the whole minute budget is spent; make the throttle treat it that way.
                        self._window.append((time.monotonic(), self.tokens_per_minute or 0))
                    log.warning("embed retry %d/%d after %s (sleep %.1fs%s)", attempt, retries, msg[:120], sleep_for,
                                f", server asked {server_delay:.0f}s" if server_delay else "")
                    time.sleep(sleep_for)
                    delay = min(delay * 2, 60)
                    continue
                raise
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
