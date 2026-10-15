"""Full answer path: rewrite → hybrid retrieve → Cohere rerank → drop weak hits → action-page promote → Gemini."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..config import Config
from ..indexing.embed import Embedder
from ..indexing.retrieve import Hit, search
from ..indexing.store import VectorIndex
from .boundaries import detect
from .generate import generate_answer, rewrite_question
from .rerank import drop_weak_hits, promote_topic_hits
from .rerank import rerank as cohere_rerank

log = logging.getLogger(__name__)


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

    def ask(self, question: str, *, history: list[dict[str, str]] | None = None,
            rerank: bool | None = None, rewrite: bool | None = None) -> Answer:
        cfg = self.cfg
        msgs = cfg.messages
        q = (question or "").strip()
        if not q:
            return Answer(status="out_of_scope", text=msgs.get("out_of_scope", ""), question=q)

        history = history or []
        do_rewrite = cfg.retrieval.get("rewrite_with_history", True) if rewrite is None else rewrite
        rewritten = q
        if do_rewrite and history:
            rewritten = rewrite_question(q, history, cfg) or q

        bound = detect(rewritten)
        if bound.kind == "out_of_scope":
            return Answer(status="out_of_scope", text=msgs.get("out_of_scope", ""), question=q,
                          rewritten=rewritten, boundary=bound.kind)
        if bound.kind == "no_account_access":
            return Answer(status="no_account_access", text=msgs.get("no_account_access", ""), question=q,
                          rewritten=rewritten, boundary=bound.kind)

        embedder, index = self.ensure()
        rcfg = cfg.retrieval
        top_k = int(rcfg.get("top_k", 12))
        max_cand = int((rcfg.get("rerank") or {}).get("max_candidates") or top_k)
        do_rerank = bool((rcfg.get("rerank") or {}).get("provider")) if rerank is None else rerank
        hits = search(index, embedder, rewritten, top_k=max_cand, hybrid=bool(rcfg.get("hybrid", True)))
        used_rerank = False
        keep = int((rcfg.get("rerank") or {}).get("top_n") or 5)
        if do_rerank and hits:
            rr = rcfg.get("rerank") or {}
            hits = cohere_rerank(
                rewritten,
                hits,
                model=rr.get("model") or "rerank-multilingual-v3.0",
                top_n=len(hits),
            )
            used_rerank = any(h.rerank_score for h in hits)
            hits = drop_weak_hits(hits)
        hits = promote_topic_hits(rewritten, hits)[:keep]

        gen_q = rewritten
        if bound.kind in ("no_advice", "personal_boundary"):
            gen_q = (
                rewritten
                + "\n\nענה רק מה כתוב באתר באופן כללי, בלי המלצה אישית ובלי התייחסות לפוליסה ספציפית."
            )
        data = generate_answer(gen_q, hits, cfg)
        status = data["status"]
        text = (data.get("answer") or "").strip()
        cites = list(data.get("citation_urls") or [])
        if status != "answered":
            status = "not_found"
            text = msgs.get("not_found", text)
            cites = []

        extra = ""
        if bound.kind == "personal_boundary":
            extra = msgs.get("personal_boundary", "")
            status = "personal_boundary" if status == "answered" else status
        elif bound.kind == "no_advice":
            extra = msgs.get("no_advice", "")
            status = "no_advice" if status == "answered" else status
        if extra and extra not in text:
            text = f"{text}\n\n{extra}".strip() if text and status != "not_found" else extra

        return Answer(
            status=status,
            text=text,
            citations=cites,
            hits=hits,
            question=q,
            rewritten=rewritten,
            used_rerank=used_rerank,
            boundary=bound.kind,
        )


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
