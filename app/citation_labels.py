# Resolve citation URLs to official Hebrew page titles (h1) for web source cards.

from __future__ import annotations

import json
from urllib.parse import urlparse, unquote

from phoenix_rag.config import Config
from phoenix_rag.crawler.store import PageStore
from phoenix_rag.indexing.retrieve import Hit

_GENERIC = frozenset({"מקור באתר הפניקס", "עמוד באתר הפניקס"})

_index_by_url: dict[str, str] | None = None
_index_stamp: float = 0.0


def norm_url_key(url: str) -> str:
    p = urlparse((url or "").strip())
    host = (p.netloc or "").lower().removeprefix("www.")
    path = unquote(p.path.rstrip("/").lower())
    return f"{host}{path}"


def _url_variants(url: str) -> list[str]:
    """All www/non-www and trailing-slash variants for SQLite page lookup."""
    u = (url or "").strip()
    if not u:
        return []
    p = urlparse(u)
    if not p.netloc:
        return [u]
    host = p.netloc.lower()
    bare = host.removeprefix("www.")
    hosts = {host, bare, f"www.{bare}"}
    path = p.path or "/"
    paths = {path.rstrip("/") or "/", path.rstrip("/") + "/" if path != "/" else "/"}
    scheme = p.scheme or "https"
    out: set[str] = set()
    for h in hosts:
        for path_i in paths:
            out.add(f"{scheme}://{h}{path_i}")
    return list(out)


def _index_stamp_for(cfg: Config) -> float:
    if not cfg.clean_dir.is_dir():
        return 0.0
    try:
        return max(f.stat().st_mtime for f in cfg.clean_dir.glob("*.json"))
    except ValueError:
        return 0.0


def page_title_index(cfg: Config) -> dict[str, str]:
    """url-key -> official h1 (fallback: document title). Built once from data/clean."""
    global _index_by_url, _index_stamp
    stamp = _index_stamp_for(cfg)
    if _index_by_url is not None and stamp == _index_stamp:
        return _index_by_url

    idx: dict[str, str] = {}
    if cfg.clean_dir.is_dir():
        for path in cfg.clean_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            label = (data.get("h1") or data.get("title") or "").strip()
            if not label:
                continue
            for field in ("url", "display_url", "canonical_url"):
                raw = data.get(field)
                if raw:
                    idx[norm_url_key(raw)] = label

    _index_by_url = idx
    _index_stamp = stamp
    return idx


def label_from_hit(hit: Hit | None) -> str | None:
    """Document-level title only — never chunk heading_path (often a sentence, not h1)."""
    if not hit:
        return None
    return (hit.page_title or "").strip() or None


def _label_from_store(url: str, store: PageStore, cfg: Config) -> str | None:
    for variant in _url_variants(url):
        row = store.get(variant)
        if not row:
            continue
        clean_path = cfg.clean_dir / f"{row.url_id}.json"
        if clean_path.is_file():
            try:
                data = json.loads(clean_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            h1 = (data.get("h1") or data.get("title") or "").strip()
            if h1:
                return h1
        if row.title:
            return row.title.strip()
    return None


def title_for_citation_url(url: str, hits: list[Hit], cfg: Config) -> str | None:
    key = norm_url_key(url)
    idx = page_title_index(cfg)
    if key in idx:
        return idx[key]

    by_key: dict[str, Hit] = {}
    for h in hits:
        for raw in (h.display_url, h.url):
            if raw:
                by_key[norm_url_key(raw)] = h
    label = label_from_hit(by_key.get(key))
    if label:
        return label

    if cfg.db_path.is_file():
        store = PageStore(cfg.db_path)
        try:
            return _label_from_store(url, store, cfg)
        finally:
            store.close()
    return None


def resolve_citation_sources(citations: list[str], hits: list[Hit], cfg: Config) -> list[dict[str, str]]:
    """Return [{url, title}, ...] — only entries with a resolved official page title."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for url in citations:
        if not url or not str(url).strip():
            continue
        key = norm_url_key(url)
        if key in seen:
            continue
        label = title_for_citation_url(url, hits, cfg)
        if not label or label in _GENERIC:
            continue
        seen.add(key)
        out.append({"url": url.strip(), "title": label})
    return out


def sources_for_answer(
    citations: list[str],
    hits: list[Hit],
    cfg: Config,
    *,
    status: str = "",
) -> list[dict[str, str]]:
    """Source cards for the UI: citations first, then top retrieved hits if needed."""
    cards = resolve_citation_sources(citations, hits, cfg)
    if cards:
        return cards
    if status != "answered":
        return []
    seen: set[str] = set()
    for h in hits[:5]:
        url = (h.display_url or h.url or "").strip()
        if not url:
            continue
        key = norm_url_key(url)
        if key in seen:
            continue
        label = title_for_citation_url(url, hits, cfg)
        if not label or label in _GENERIC:
            continue
        seen.add(key)
        cards.append({"url": url, "title": label})
    return cards
