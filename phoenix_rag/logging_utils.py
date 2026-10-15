"""Per-query JSONL logs for the answer pipeline.

Writes rewritten question, retrieved chunk ids/scores, answer_type and sources.
Does not store names, ID numbers, policy numbers, or other personal identifiers.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config

log = logging.getLogger(__name__)

# Redact obvious ID-like tokens so a typed policy/claim number is not persisted.
_ID_RE = re.compile(r"\b\d{6,}\b")


def queries_path(cfg: Config) -> Path:
    cfg.ensure_dirs()
    rel = (cfg.raw.get("logging") or {}).get("queries_path") or "logs/queries.jsonl"
    return cfg.root / rel


def _scrub(text: str) -> str:
    return _ID_RE.sub("[id]", text or "")


def log_query(cfg: Config, record: dict[str, Any]) -> Path:
    """Append one query record to logs/queries.jsonl. Returns the log path."""
    path = queries_path(cfg)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": _scrub(str(record.get("question") or "")),
        "rewritten": _scrub(str(record.get("rewritten") or "")),
        "hits": record.get("hits") or [],
        "answer_type": record.get("answer_type") or "",
        "status": record.get("status") or "",
        "sources": record.get("sources") or [],
        "used_chunk_ids": record.get("used_chunk_ids") or [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.debug("query log -> %s", path)
    return path
