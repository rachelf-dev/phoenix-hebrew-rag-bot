# Audit: pages whose raw HTML has "נשמח לשוחח" but cleaned text drops phone/WhatsApp.
"""Usage: python scripts/audit_contact_blocks.py"""

from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phoenix_rag.cleaning.clean import clean_html
from phoenix_rag.config import load_config

CANONICAL_RE = re.compile(r'<link rel="canonical" href="([^"]+)"', re.I)


def main() -> int:
    cfg = load_config()
    raw_dir = cfg.root / "data" / "raw"
    lost: list[str] = []
    kept: list[str] = []
    with_phrase = 0
    for gz in raw_dir.glob("*.html.gz"):
        html = gzip.open(gz, "rt", encoding="utf-8", errors="replace").read()
        if "נשמח לשוחח" not in html:
            continue
        with_phrase += 1
        m = CANONICAL_RE.search(html)
        url = m.group(1) if m else gz.name
        cp = clean_html(html, url=url, cfg=cfg, section_type="service")
        plain = cp.plain_text()
        raw_has_phone = "5432" in html or "9893455" in html
        clean_has_contact = any(x in plain for x in ("5432", "9893455", "נשמח"))
        if raw_has_phone and not clean_has_contact:
            lost.append(url)
        else:
            kept.append(url)
    print(f"raw pages with 'נשמח לשוחח': {with_phrase}")
    print(f"lost contact phones in clean: {len(lost)}")
    print(f"kept/partial: {len(kept)}")
    for u in lost:
        print(f"  LOST {u}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
