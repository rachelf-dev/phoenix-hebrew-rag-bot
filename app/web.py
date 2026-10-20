# FastAPI app: serves app/static/index.html (עיצוב.txt) and wires the chat widget to Phoenix RAG.

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.citation_labels import norm_url_key, page_title_index, sources_for_answer, title_for_citation_url
from phoenix_rag.answering.context import history_window, history_window_size
from phoenix_rag.answering.pipeline import Pipeline
from phoenix_rag.config import Config
from phoenix_rag.indexing.retrieve import Hit

log = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
INDEX_HTML = STATIC_DIR / "index.html"


class HistoryTurn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[HistoryTurn] = Field(default_factory=list)


def _hits_payload(hits: list[Hit], cfg: Config, limit: int = 5) -> list[dict[str, Any]]:
    idx = page_title_index(cfg)
    out: list[dict[str, Any]] = []
    for h in hits[:limit]:
        url = h.display_url or h.url
        page_h1 = idx.get(norm_url_key(url), "") if url else ""
        if not page_h1 and url:
            page_h1 = title_for_citation_url(url, hits, cfg) or ""
        preview = (h.text or "").strip().replace("\n", " ")
        if len(preview) > 420:
            preview = preview[:417] + "…"
        out.append({
            "chunk_id": h.chunk_id,
            "url": url,
            "page_h1": page_h1,
            "page_title": h.page_title or "",
            "heading_path": h.heading_path or "",
            "score": round(float(h.score or 0), 4),
            "rerank_score": round(float(h.rerank_score or 0), 4) if h.rerank_score else None,
            "preview": preview,
        })
    return out


def create_app(cfg: Config) -> FastAPI:
    if not INDEX_HTML.is_file():
        raise FileNotFoundError(f"missing UI template: {INDEX_HTML}")

    pipe = Pipeline(cfg)
    log.info("loading embedder + index for web UI…")
    pipe.ensure()
    page_title_index(cfg)
    lock = threading.Lock()
    disclaimer = cfg.messages.get("disclaimer") or ""

    app = FastAPI(title="Phoenix RAG", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(INDEX_HTML, media_type="text/html; charset=utf-8")

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        pages = 0
        chunks = 0
        try:
            if cfg.db_path.is_file():
                from phoenix_rag.crawler.store import PageStore

                store = PageStore(cfg.db_path)
                pages = store.counts().get("active", 0)
                store.close()
        except Exception:
            log.exception("page count failed")
        try:
            if cfg.chunks_path.is_file():
                with cfg.chunks_path.open("r", encoding="utf-8") as fh:
                    chunks = sum(1 for _ in fh)
        except OSError:
            pass
        return {
            "pages": pages,
            "chunks": chunks,
            "embedding_model": cfg.embedding.get("model", ""),
        }

    @app.post("/api/ask")
    def ask(body: AskRequest) -> dict[str, Any]:
        q = (body.question or "").strip()
        if not q:
            raise HTTPException(status_code=400, detail="empty question")
        history = history_window(
            [{"role": h.role, "content": h.content} for h in body.history],
            history_window_size(cfg),
        )
        with lock:
            ans = pipe.ask(q, history=history)
        source_cards = sources_for_answer(
            ans.citations, ans.hits, cfg, status=ans.status
        )
        return {
            "status": ans.status,
            "answer": ans.text,
            "answer_text": ans.text,
            "answer_type": ans.answer_type,
            "citations": ans.citations,
            "sources": source_cards,
            "used_chunk_ids": ans.used_chunk_ids,
            "hits": _hits_payload(ans.hits, cfg),
            "disclaimer": disclaimer,
        }

    return app
