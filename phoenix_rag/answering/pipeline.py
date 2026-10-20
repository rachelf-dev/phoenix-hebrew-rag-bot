"""Full answer path: rewrite → hybrid retrieve → Cohere rerank → drop weak hits → action-page promote → Gemini.

Returns a structured Answer (spec JSON: answer_type, answer_text, sources, used_chunk_ids)
and appends one JSONL line per query.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import Config
from ..indexing.embed import Embedder
from ..indexing.retrieve import Hit, search
from ..indexing.store import VectorIndex
from ..logging_utils import log_query
from .boundaries import detect_with_history
from .generate import generate_answer, rewrite_question
from .rerank import demote_mismatched_paths, drop_weak_hits, promote_topic_hits
from .rerank import rerank as cohere_rerank

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
        }


def _retrieval_query(rewritten: str, history: list[dict[str, str]]) -> str:
    """Short follow-ups: widen the search query with recent user context from the thread."""
    if not history:
        return rewritten
    users = [(t.get("content") or "").strip() for t in history if t.get("role") == "user"]
    users = [u for u in users if u]
    if not users:
        return rewritten
    short = len(rewritten) < 90 or len(rewritten.split()) <= 14
    if not short:
        return rewritten
    anchor = users[-1]
    if len(anchor) < 40 and len(users) >= 2:
        anchor = f"{users[-2]} {users[-1]}".strip()
    combined = f"{anchor} {rewritten}".strip()
    return combined[:900]


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

    def _finish(self, ans: Answer) -> Answer:
        ans.used_chunk_ids = _used_chunk_ids(ans.hits, ans.citations)
        try:
            log_query(self.cfg, {
                "question": ans.question,
                "rewritten": ans.rewritten,
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

    def ask(self, question: str, *, history: list[dict[str, str]] | None = None,
            rerank: bool | None = None, rewrite: bool | None = None) -> Answer:
        cfg = self.cfg
        msgs = cfg.messages
        q = (question or "").strip()
        if not q:
            return self._finish(Answer(status="out_of_scope", text=msgs.get("out_of_scope", ""), question=q))

        history = history or []
        do_rewrite = cfg.retrieval.get("rewrite_with_history", True) if rewrite is None else rewrite
        rewritten = q
        if do_rewrite and history:
            rewritten = rewrite_question(q, history, cfg) or q

        bound = detect_with_history(rewritten, history)
        if bound.kind == "out_of_scope":
            return self._finish(Answer(status="out_of_scope", text=msgs.get("out_of_scope", ""), question=q,
                                       rewritten=rewritten, boundary=bound.kind))
        if bound.kind == "no_account_access":
            return self._finish(Answer(status="no_account_access", text=msgs.get("no_account_access", ""), question=q,
                                       rewritten=rewritten, boundary=bound.kind))

        embedder, index = self.ensure()
        rcfg = cfg.retrieval
        top_k = int(rcfg.get("top_k", 12))
        max_cand = int((rcfg.get("rerank") or {}).get("max_candidates") or top_k)
        do_rerank = bool((rcfg.get("rerank") or {}).get("provider")) if rerank is None else rerank
        search_q = _retrieval_query(rewritten, history)
        hits = search(
            index,
            embedder,
            search_q,
            top_k=max_cand,
            hybrid=bool(rcfg.get("hybrid", True)),
            threshold=float(rcfg.get("relevance_threshold") or 0.0),
            max_per_url=int(rcfg.get("max_per_url") or 2),
        )
        hits = demote_mismatched_paths(search_q, hits)
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
            hits = drop_weak_hits(hits)
            hits = demote_mismatched_paths(search_q, hits)
        hits = promote_topic_hits(search_q, hits)[:keep]

        gen_q = search_q if search_q != rewritten else rewritten
        if bound.kind in ("no_advice", "personal_boundary"):
            gen_q = (
                rewritten
                + "\n\nענה רק מה כתוב באתר באופן כללי, בלי המלצה אישית ובלי התייחסות לפוליסה ספציפית."
            )
        data = generate_answer(gen_q, hits, cfg)
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

        return self._finish(Answer(
            status=status,
            text=text,
            citations=cites,
            hits=hits,
            question=q,
            rewritten=rewritten,
            used_rerank=used_rerank,
            boundary=bound.kind,
        ))


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
