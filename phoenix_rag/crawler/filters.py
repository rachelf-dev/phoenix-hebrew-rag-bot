"""Crawl-time filters: NetFree/Guard interstitials, empty <main>, and URL/hash duplicates.

Used by the httpx crawler so blocked or duplicate pages never become `active` and
do not reach clean/chunk/index. Playwright is not required (fnx.co.il is SSR).
"""

from __future__ import annotations

import hashlib
import re

# Visible <main> shorter than this, with a foreign/filter title, is treated as a block page.
MIN_MAIN_CHARS = 40
EMPTY_HASH = hashlib.sha256(b"").hexdigest()

_PHOENIX_TITLE = re.compile(r"הפניקס|fnx|phoenix", re.I)

_FILTER_TITLE = re.compile(
    r"נטפרי|netfree|נחסם|חסום|blocked|access denied|captcha|just a moment|"
    r"cloudflare|attention required|guard",
    re.I,
)

_FILTER_BODY = re.compile(
    r"blockByNetFree|netfree\.(org|link|co\.il)|blocked by netfree|"
    r"האתר נחסם|האתר חסום|נחסם על ידי|חסימת אתר|"
    r"cf-browser-verification|cdn-cgi/challenge|g-recaptcha|"
    r"attention required! \| cloudflare",
    re.I,
)


def looks_like_phoenix_title(title: str | None) -> bool:
    return bool(title and _PHOENIX_TITLE.search(title))


def filter_reason(html_text: str, title: str | None, main_text: str) -> str | None:
    """Return a short reason when the fetched HTML is a filter/interstitial, else None."""
    head = f"{title or ''}\n{(html_text or '')[:12_000]}"
    if _FILTER_BODY.search(head) or (title and _FILTER_TITLE.search(title)):
        return "filter_fingerprint"
    text = re.sub(r"\s+", " ", main_text or "").strip()
    if len(text) >= MIN_MAIN_CHARS:
        return None
    if title and _FILTER_TITLE.search(title):
        return "empty_main_filter_title"
    if not looks_like_phoenix_title(title) and not text:
        return "empty_main_foreign_title"
    return None


_BAD_CANONICAL = {"", "undefined", "null", "none"}


def usable_canonical(canonical: str | None) -> str | None:
    """Drop JS leftovers like 'undefined' that are not real URLs."""
    c = (canonical or "").strip()
    if not c or c.lower() in _BAD_CANONICAL:
        return None
    if not c.startswith("http"):
        return None
    return c


def canonical_target(url: str, canonical: str | None) -> str | None:
    """Canonical URL to defer to when it differs from the fetched URL."""
    canonical = usable_canonical(canonical)
    if not url or not canonical:
        return None
    if url.rstrip("/") == canonical.rstrip("/"):
        return None
    return canonical


def pick_canonical_keeper(canonical: str, urls: list[str]) -> str:
    """Keep the canonical URL itself when present; otherwise the shortest URL."""
    want = (canonical or "").rstrip("/")
    for u in urls:
        if (u or "").rstrip("/") == want:
            return u
    return min(urls, key=lambda u: (len(u), u))


def pick_hash_keeper(urls: list[str]) -> str:
    """First-wins among identical hashes: shortest normalised URL, then lexicographic."""
    return min(urls, key=lambda u: (len(u), u))


def plan_duplicates(
    rows: list[tuple[str, str | None, str | None]],
) -> list[tuple[str, str, str]]:
    """Decide which active pages are aliases.

    Each row is ``(url, canonical_url, content_hash)``. Returns
    ``(url, event, detail)`` for pages that should become ``duplicate``.
    Canonical groups are resolved first so a hash-identical canonical owner is kept.
    """
    marks: list[tuple[str, str, str]] = []
    dropped: set[str] = set()

    by_canon: dict[str, list[str]] = {}
    for url, canonical, _hash in rows:
        key = (usable_canonical(canonical) or url or "").rstrip("/")
        if not key:
            continue
        by_canon.setdefault(key, []).append(url)
    for key, urls in by_canon.items():
        uniq = list(dict.fromkeys(urls))
        if len(uniq) < 2:
            continue
        keeper = pick_canonical_keeper(key, uniq)
        for u in uniq:
            if u != keeper and u not in dropped:
                marks.append((u, "duplicate_canonical", f"canonical={key} keeper={keeper}"))
                dropped.add(u)

    by_hash: dict[str, list[str]] = {}
    for url, _canonical, h in rows:
        if url in dropped or not h or h == EMPTY_HASH:
            continue
        by_hash.setdefault(h, []).append(url)
    for h, urls in by_hash.items():
        uniq = list(dict.fromkeys(urls))
        if len(uniq) < 2:
            continue
        keeper = pick_hash_keeper(uniq)
        for u in uniq:
            if u != keeper and u not in dropped:
                marks.append((u, "duplicate_hash", f"hash={h[:16]} keeper={keeper}"))
                dropped.add(u)
    return marks
