"""Full answer path: sanitize → rewrite → hybrid retrieve → Cohere rerank → Gemini.

Follow-ups inherit the last 4–6 turns before search. Weak rerank peaks skip
generation and return the *5040 not_found message. Gemini 429 exhaustion is a
soft fail (same message), not HTTP 500. Returns a structured Answer and logs
one JSONL line.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from ..config import Config
from ..indexing.embed import Embedder
from ..indexing.retrieve import Hit, search
from ..indexing.store import VectorIndex
from ..logging_utils import log_query
from .boundaries import Boundary, detect_with_history
from .context import history_window, history_window_size, retrieval_query
from .context import is_claim_sla_question
from .generate import generate_answer, grounded_action_fallback, rewrite_question
from .rerank import demote_mismatched_paths, drop_weak_hits, filter_hits_for_question, promote_topic_hits
from .rerank import rerank as cohere_rerank
from .sanitize import is_unintelligible_input, looks_like_injection, sanitize_history, sanitize_question

log = logging.getLogger(__name__)

_SPEC_TYPES = ("answered", "not_found", "personal_boundary", "out_of_scope")


@dataclass
class Answer:
    status: str
    text: str
    citations: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    question: str = ""
    rewritten: str = ""
    search_query: str = ""
    used_rerank: bool = False
    boundary: str = ""
    used_chunk_ids: list[str] = field(default_factory=list)

    @property
    def answer_type(self) -> str:
        """Spec field: answered | not_found | personal_boundary | out_of_scope."""
        if self.status in ("no_account_access", "no_advice"):
            return "personal_boundary"
        if self.status in _SPEC_TYPES:
            return self.status
        return "not_found"

    def to_json(self) -> dict:
        return {
            "answer_type": self.answer_type,
            "answer_text": self.text,
            "sources": self.citations,
            "used_chunk_ids": self.used_chunk_ids,
            "status": self.status,
            "rewritten": self.rewritten,
            "search_query": self.search_query,
        }


def _used_chunk_ids(hits: list[Hit], urls: list[str]) -> list[str]:
    needles = [u.rstrip("/") for u in urls if u]
    ids: list[str] = []
    seen: set[str] = set()
    for h in hits:
        u = (h.display_url or h.url or "").rstrip("/")
        if h.chunk_id in seen:
            continue
        if any(n and (n in u or u in n) for n in needles):
            ids.append(h.chunk_id)
            seen.add(h.chunk_id)
    return ids


@dataclass
class _PreparedAsk:
    """Retrieve/rerank result, or an early canned Answer that skips Gemini."""

    question: str
    rewritten: str = ""
    search_q: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    bound: Boundary = field(default_factory=Boundary)
    hits: list[Hit] = field(default_factory=list)
    used_rerank: bool = False
    early: Answer | None = None


class Pipeline:
    """Load the local embedder + Chroma once, then answer many questions."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._embedder: Embedder | None = None
        self._index: VectorIndex | None = None

    def ensure(self) -> tuple[Embedder, VectorIndex]:
        if self._embedder is None:
            self._embedder = Embedder.from_config(self.cfg)
            self._index = VectorIndex(self.cfg, self._embedder.model_name)
        assert self._index is not None
        return self._embedder, self._index

    def _max_question_chars(self) -> int:
        try:
            return int(self.cfg.retrieval.get("max_question_chars") or 1000)
        except (TypeError, ValueError):
            return 1000

    def _finish(self, ans: Answer) -> Answer:
        ans.used_chunk_ids = _used_chunk_ids(ans.hits, ans.citations)
        try:
            log_query(self.cfg, {
                "question": ans.question,
                "rewritten": ans.rewritten,
                "search_query": ans.search_query,
                "hits": [
                    {
                        "chunk_id": h.chunk_id,
                        "url": h.display_url or h.url,
                        "score": round(float(h.score or 0), 6),
                        "rerank_score": round(float(h.rerank_score or 0), 6),
                    }
                    for h in ans.hits
                ],
                "answer_type": ans.answer_type,
                "status": ans.status,
                "sources": ans.citations,
                "used_chunk_ids": ans.used_chunk_ids,
            })
        except Exception:
            log.exception("query log failed")
        return ans

    def _prepare(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        rerank: bool | None = None,
        rewrite: bool | None = None,
    ) -> _PreparedAsk:
        cfg = self.cfg
        msgs = cfg.messages
        cap = self._max_question_chars()
        q = sanitize_question(question, max_chars=cap)
        if not q:
            return _PreparedAsk(
                question=q,
                early=Answer(status="out_of_scope", text=msgs.get("out_of_scope", ""), question=q),
            )
        if looks_like_injection(q):
            return _PreparedAsk(
                question=q,
                early=Answer(
                    status="out_of_scope",
                    text=msgs.get("out_of_scope", ""),
                    question=q,
                    boundary="injection",
                ),
            )
        if is_unintelligible_input(q):
            return _PreparedAsk(
                question=q,
                early=Answer(
                    status="not_found",
                    text=msgs.get("unintelligible", msgs.get("not_found", "")),
                    question=q,
                    boundary="unintelligible",
                ),
            )

        history = sanitize_history(
            history_window(history or [], history_window_size(cfg)),
            max_chars=cap,
        )
        do_rewrite = cfg.retrieval.get("rewrite_with_history", True) if rewrite is None else rewrite
        rewritten = q
        if do_rewrite and history:
            rewritten = rewrite_question(q, history, cfg) or q

        bound = detect_with_history(q, rewritten, history)
        if bound.kind == "out_of_scope":
            return _PreparedAsk(
                question=q, rewritten=rewritten, history=history, bound=bound,
                early=Answer(status="out_of_scope", text=msgs.get("out_of_scope", ""), question=q,
                             rewritten=rewritten, boundary=bound.kind),
            )
        if bound.kind == "no_account_access":
            return _PreparedAsk(
                question=q, rewritten=rewritten, history=history, bound=bound,
                early=Answer(status="no_account_access", text=msgs.get("no_account_access", ""), question=q,
                             rewritten=rewritten, boundary=bound.kind),
            )

        embedder, index = self.ensure()
        rcfg = cfg.retrieval
        top_k = int(rcfg.get("top_k", 12))
        max_cand = int((rcfg.get("rerank") or {}).get("max_candidates") or top_k)
        do_rerank = bool((rcfg.get("rerank") or {}).get("provider")) if rerank is None else rerank
        search_q = retrieval_query(rewritten, history)
        hits = search(
            index,
            embedder,
            search_q,
            top_k=max_cand,
            hybrid=bool(rcfg.get("hybrid", True)),
            threshold=float(rcfg.get("relevance_threshold") or 0.0),
            max_per_url=int(rcfg.get("max_per_url") or 2),
        )
        hits = demote_mismatched_paths(search_q, hits, history=history)
        used_rerank = False
        keep = int((rcfg.get("rerank") or {}).get("top_n") or 5)
        if do_rerank and hits:
            rr = rcfg.get("rerank") or {}
            hits = cohere_rerank(
                search_q,
                hits,
                model=rr.get("model") or "rerank-multilingual-v3.0",
                top_n=len(hits),
            )
            used_rerank = any(h.rerank_score for h in hits)
            min_score = 0.0
            try:
                min_score = float(rr.get("min_score") or 0.0)
            except (TypeError, ValueError):
                min_score = 0.0
            hits = drop_weak_hits(hits, refuse_below=min_score if used_rerank else 0.0)
            hits = demote_mismatched_paths(search_q, hits, history=history)
        hits = filter_hits_for_question(search_q, hits, history=history)
        hits = promote_topic_hits(search_q, hits, history=history)[:keep]
        return _PreparedAsk(
            question=q, rewritten=rewritten, search_q=search_q, history=history,
            bound=bound, hits=hits, used_rerank=used_rerank,
        )

    def _complete(self, prep: _PreparedAsk) -> Answer:
        cfg = self.cfg
        msgs = cfg.messages
        bound = prep.bound
        hits = prep.hits
        gen_q = prep.rewritten
        if bound.kind in ("no_advice", "personal_boundary"):
            gen_q = (
                prep.rewritten
                + "\n\nענה רק מה כתוב באתר באופן כללי, בלי המלצה אישית ובלי התייחסות לפוליסה ספציפית."
            )
        try:
            data = generate_answer(
                gen_q, hits, cfg, history=prep.history, original=prep.question,
            )
        except Exception as exc:
            log.warning("Gemini generate failed; soft-fail to not_found: %s", str(exc)[:300])
            if is_claim_sla_question(prep.question):
                data = {"status": "not_found", "answer": "", "citation_urls": []}
            else:
                fallback = grounded_action_fallback(prep.rewritten, hits, history=prep.history)
                data = fallback or {"status": "not_found", "answer": "", "citation_urls": []}
        gen_status = data["status"]
        text = (data.get("answer") or "").strip()
        cites = list(data.get("citation_urls") or [])
        status = gen_status if gen_status == "answered" else "not_found"
        if status != "answered":
            text = msgs.get("not_found", text)
            cites = []

        if bound.kind == "personal_boundary":
            extra = msgs.get("personal_boundary", "")
            status = "personal_boundary"
            if gen_status != "answered" or not text:
                text = extra
                cites = []
            elif extra and extra not in text:
                text = f"{text}\n\n{extra}".strip()
        elif bound.kind == "no_advice":
            extra = msgs.get("no_advice", "")
            status = "no_advice"
            if gen_status != "answered" or not text:
                text = extra
                cites = []
            elif extra and extra not in text:
                text = f"{text}\n\n{extra}".strip()

        return Answer(
            status=status,
            text=text,
            citations=cites,
            hits=hits,
            question=prep.question,
            rewritten=prep.rewritten,
            search_query=prep.search_q,
            used_rerank=prep.used_rerank,
            boundary=bound.kind,
        )

    def ask(self, question: str, *, history: list[dict[str, str]] | None = None,
            rerank: bool | None = None, rewrite: bool | None = None) -> Answer:
        prep = self._prepare(question, history=history, rerank=rerank, rewrite=rewrite)
        if prep.early is not None:
            return self._finish(prep.early)
        return self._finish(self._complete(prep))

    def ask_events(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        rerank: bool | None = None,
        rewrite: bool | None = None,
    ) -> Iterator[tuple[str, Answer | dict[str, str]]]:
        """Yield ('status', {stage}) then ('done', Answer) for the SSE chat path."""
        yield ("status", {"stage": "searching"})
        prep = self._prepare(question, history=history, rerank=rerank, rewrite=rewrite)
        if prep.early is not None:
            yield ("done", self._finish(prep.early))
            return
        yield ("status", {"stage": "generating"})
        yield ("done", self._finish(self._complete(prep)))


def format_cli(ans: Answer, cfg: Config) -> str:
    """Answer body + source URLs + disclaimer, for the CLI."""
    parts = [ans.text.rstrip()]
    if ans.citations:
        parts.append("")
        parts.append("מקורות:")
        parts.extend(ans.citations)
    disc = cfg.messages.get("disclaimer")
    if disc:
        parts.append("")
        parts.append(disc)
    return "\n".join(parts)
