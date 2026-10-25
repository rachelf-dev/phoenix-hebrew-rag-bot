# Re-clean/chunk pages that lost in-page contact blocks; patch index_chunks.json in place (no re-embed).
"""Usage: python scripts/patch_contact_index.py [--dry-run]"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phoenix_rag.cleaning.clean import clean_html
from phoenix_rag.config import load_config
from phoenix_rag.crawler.urls import url_id
from phoenix_rag.indexing.chunk import Chunk, chunk_page, ntokens


def _contact_tail(plain: str) -> str:
    if "נשמח לשוחח" not in plain:
        return ""
    tail = plain.split("נשמח לשוחח", 1)[1].strip()
    return ("נשמח לשוחח\n\n" + tail).strip() if tail else ""


def _merge_chunks(a: Chunk, b: Chunk) -> Chunk:
    orig = f"{a.original_text.rstrip()}\n\n{b.original_text.lstrip()}".strip()
    embed = f"{a.embed_text.rstrip()}\n\n{b.original_text.lstrip()}".strip()
    return Chunk(
        chunk_id=a.chunk_id,
        url=a.url,
        display_url=a.display_url,
        page_title=a.page_title,
        heading_path=a.heading_path,
        fetch_date=a.fetch_date,
        content_hash=a.content_hash,
        section_type=a.section_type,
        language=a.language,
        original_text=orig,
        embed_text=embed,
        token_count=ntokens(embed),
        block_types=f"{a.block_types},{b.block_types}",
        form_links=a.form_links or b.form_links,
    )


def _fit_chunk_count(chunks: list[Chunk], target: int) -> list[Chunk]:
    out = list(chunks)
    while len(out) > target and len(out) >= 2:
        out[-2] = _merge_chunks(out[-2], out[-1])
        out.pop()
    while len(out) < target and out:
        out.append(out[-1])
    return out[:target]


def _load_pages(db_path: Path) -> list[tuple[str, str, str]]:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT url, url_id, raw_path FROM pages WHERE state='active' AND raw_path IS NOT NULL"
    ).fetchall()
    conn.close()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    index_path = cfg.index_chunks_path
    rows: list[dict] = json.loads(index_path.read_text(encoding="utf-8"))
    by_url: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        key = (row.get("display_url") or row.get("url") or "").rstrip("/")
        by_url.setdefault(key, []).append(i)

    db = cfg.data_dir / "pages.sqlite"
    if not db.is_file():
        raise SystemExit(f"missing {db}")

    patched_urls: list[str] = []
    id_changes: list[tuple[str, str, str]] = []
    skipped = 0

    for url, uid, raw_rel in _load_pages(db):
        raw_path = cfg.root / raw_rel
        if not raw_path.is_file():
            continue
        html = gzip.open(raw_path, "rt", encoding="utf-8", errors="replace").read()
        if "5432" not in html and "9893455" not in html:
            continue
        page = clean_html(html, url=url, cfg=cfg, section_type="service")
        plain = page.plain_text()
        if not _contact_tail(plain):
            continue
        key = (page.display_url or page.url).rstrip("/")
        slots = by_url.get(key)
        if not slots:
            skipped += 1
            continue
        old_slice = [rows[i] for i in slots]
        if any("5432" in (r.get("original_text") or "") for r in old_slice):
            continue

        clean_path = cfg.clean_dir / f"{uid}.json"
        if not args.dry_run:
            clean_path.write_text(page.to_json(), encoding="utf-8")

        new_chunks = chunk_page(page, cfg)
        fitted = _fit_chunk_count(new_chunks, len(slots))
        for slot_i, idx in enumerate(slots):
            old_id = rows[idx]["chunk_id"]
            nc = fitted[slot_i]
            new_dict = asdict(nc)
            new_dict["chunk_id"] = old_id  # keep row/chunk_id stable for index.npy alignment
            if new_dict.get("original_text") != rows[idx].get("original_text"):
                patched_urls.append(url)
                if old_id != nc.chunk_id:
                    id_changes.append((url, old_id, nc.chunk_id))
            if not args.dry_run:
                rows[idx] = new_dict

    print(f"patched_urls={len(set(patched_urls))}  skipped_no_index={skipped}")
    print(f"chunk_id would have changed (kept old id in index): {len(id_changes)}")
    if id_changes[:5]:
        for u, old, new in id_changes[:5]:
            print(f"  {u}\n    kept {old}\n    new text hash id would be {new}")

    if not args.dry_run and patched_urls:
        index_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        mirror = cfg.root / "index_artifacts" / "index_chunks.json"
        if mirror != index_path and mirror.is_file():
            mirror.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {index_path} ({len(rows)} rows)")
        import subprocess

        subprocess.run([sys.executable, str(ROOT / "scripts" / "promote_contact_to_first_chunk.py")], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
