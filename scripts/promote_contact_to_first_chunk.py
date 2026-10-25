# Copy in-page contact lines into each URL's :0000: chunk so max_per_url=2 still reaches phones.
"""Usage: python scripts/promote_contact_to_first_chunk.py"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phoenix_rag.config import load_config
from phoenix_rag.indexing.chunk import ntokens

CONTACT_MARKERS = ("5432", "9893455", "נשמח לשוחח")


def _contact_block(text: str) -> str:
    if "נשמח לשוחח" in text:
        return text[text.index("נשמח לשוחח") :].strip()
    lines: list[str] = []
    for line in text.splitlines():
        if any(m in line for m in ("5432", "9893455", "WhatsApp", "וואטסאפ", "בצ'אט", "בטלפון")):
            lines.append(line.strip())
    return "\n".join(lines).strip()


def main() -> int:
    cfg = load_config()
    path = cfg.index_chunks_path
    rows: list[dict] = json.loads(path.read_text(encoding="utf-8"))
    by_url: dict[str, list[dict]] = {}
    for row in rows:
        key = (row.get("display_url") or row.get("url") or "").rstrip("/")
        by_url.setdefault(key, []).append(row)

    touched = 0
    for _key, group in by_url.items():
        contact_src = ""
        for row in group:
            t = row.get("original_text") or ""
            if any(m in t for m in CONTACT_MARKERS):
                block = _contact_block(t)
                if block:
                    contact_src = f"{contact_src}\n{block}".strip() if contact_src else block
        if contact_src and "5432" not in contact_src:
            for row in group:
                if "5432" in (row.get("original_text") or ""):
                    extra = _contact_block(row["original_text"])
                    if extra:
                        contact_src = f"{contact_src}\n{extra}".strip()
                    break
        if not contact_src:
            continue
        first = min(group, key=lambda r: r.get("chunk_id") or "")
        if contact_src in (first.get("original_text") or ""):
            continue
        orig = (first.get("original_text") or "").rstrip()
        first["original_text"] = f"{orig}\n\n{contact_src}".strip()
        prefix = (first.get("embed_text") or "").split("\n\n", 1)[0]
        body = first["original_text"]
        first["embed_text"] = f"{prefix}\n\n{body}" if prefix else body
        first["token_count"] = ntokens(first["embed_text"])
        touched += 1

    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"promoted contact into first chunk for {touched} URLs -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
