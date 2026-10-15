"""URL normalisation and scope rules for fnx.co.il.

Normalisation rules (spec 6.1: "normalize URLs, strip tracking parameters and fragments, de-duplicate"):
  * HTML entities in the sitemap (``&#x5E7;``) are unescaped.
  * scheme -> https, host -> www.fnx.co.il, lowercase host.
  * fragment dropped; query string dropped (site does not use queries for content;
    robots.txt disallows ``?page=``, ``?tag=`` ...).
  * path percent-encoded consistently (Hebrew slugs), trailing slash added
    (the sitemap uses trailing slashes everywhere).
"""

from __future__ import annotations

import hashlib
import html
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from ..config import Config

CANONICAL_HOST = "www.fnx.co.il"


def normalize_url(url: str, cfg: Config | None = None) -> str | None:
    """Return the normalised URL, or None if it is not an http(s) URL on fnx.co.il."""
    if not url:
        return None
    url = html.unescape(url.strip())
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https", ""):
        return None
    host = (parts.hostname or "").lower()
    allowed = set((cfg.crawl.get("allowed_hosts") if cfg else None) or [CANONICAL_HOST, "fnx.co.il"])
    if host and host not in allowed:
        return None
    if not host:
        return None  # relative links are resolved by the caller with urljoin before reaching here
    path = unquote(parts.path or "/")
    path = quote(path, safe="/-_.~()!*'")
    if not path.endswith("/") and "." not in path.rsplit("/", 1)[-1]:
        path += "/"
    drop_query = True if cfg is None else cfg.crawl.get("drop_query_params", True)
    query = "" if drop_query else parts.query
    return urlunsplit(("https", CANONICAL_HOST, path, query, ""))


def display_url(url: str) -> str:
    """Human-readable form (Hebrew slugs decoded) for citations."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, unquote(parts.path), parts.query, ""))


def url_path(url: str) -> str:
    return unquote(urlsplit(url).path) or "/"


def url_id(url: str) -> str:
    """Stable id used as the raw-HTML file name."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def classify(url: str, cfg: Config) -> tuple[bool, str, int, str]:
    """Decide whether *url* is in scope.

    Returns (allowed, section_type, priority, reason).  ``reason`` explains an exclusion
    ("excluded:/en/") or a match ("include:/claims-customer-service/").
    """
    path = url_path(url)
    if path.lower().endswith((".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".xml", ".zip", ".doc", ".docx", ".xls", ".xlsx")):
        return False, "", 0, "excluded:binary"
    for ex in cfg.seeds.exclude:
        if path.startswith(ex.prefix):
            return False, "", 0, f"excluded:{ex.prefix}"
    for inc in cfg.seeds.include:
        if inc.exact:
            if path == inc.prefix:
                return True, inc.section_type, inc.priority, f"include:{inc.prefix}"
            continue
        if path.startswith(inc.prefix):
            return True, refine_section_type(path, inc.section_type, cfg), inc.priority, f"include:{inc.prefix}"
    return False, "", 0, "excluded:not-in-seed-areas"


def refine_section_type(path: str, default: str, cfg: Config) -> str:
    """Apply the `section_type_rules` fragments from seeds.yaml on top of the area default."""
    lowered = path.lower()
    for stype, fragments in cfg.seeds.section_type_rules.items():
        for frag in fragments:
            if frag.lower() in lowered:
                return stype
    return default
