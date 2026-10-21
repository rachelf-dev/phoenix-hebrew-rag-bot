"""Cohere Rerank over hybrid hits, plus a light how-to boost for self-service pages.

Uses httpx so we do not add a Cohere SDK. A 418/NetFree or missing key disables
rerank for the rest of the process and the hybrid ranking is kept. After scores
arrive, near-zero hits are dropped so Gemini is not fed unrelated chunks.
"""

from __future__ import annotations

import logging
import os
import time
from urllib.parse import urlparse

import httpx

from ..indexing.hebrew_lex import tokenize
from ..indexing.retrieve import Hit
from .context import claim_path_markers, extract_thread_slots

log = logging.getLogger(__name__)

_TYPE_HE = {
    "service": "דף פעולה / שירות עצמי",
    "faq": "שאלות ותשובות",
    "form": "טופס",
    "product": "דף מוצר",
    "guide": "מדריך / הסבר",
    "article": "מאמר",
    "claim": "דף תביעה",
}

_V2 = "https://api.cohere.com/v2/rerank"
_V1 = "https://api.cohere.com/v1/rerank"

_disabled_reason: str | None = None


def disabled_reason() -> str | None:
    return _disabled_reason


def _url_path(hit: Hit) -> str:
    raw = hit.display_url or hit.url or ""
    path = urlparse(raw).path.lower()
    return path if path.endswith("/") else f"{path}/"


def _is_howto(question: str) -> bool:
    q = (question or "").lower()
    return any(m in q for m in (
        "איך", "כיצד", "how do", "how to", "how can",
        "למלא", "טופס", "טפסים", "מה עושים", "מה הצעד",
        "לאן שולח", "איפה מגיש",
    ))


def _is_define(question: str) -> bool:
    """Coverage / 'what is this product' questions — not a bare 'האם' (too common on unanswerable)."""
    return any(m in (question or "") for m in ("מה כולל", "מה זה", "מיועד", "איזה ביטוח יש"))


def _title_overlap(question: str, hit: Hit, *, with_body: bool = False) -> int:
    q_toks = {t for t in tokenize(question or "") if len(t) >= 3}
    path = _url_path(hit).replace("-", " ").replace("_", " ")
    parts = [hit.page_title, hit.heading_path, path]
    if with_body:
        parts.append((hit.text or "")[:240])
    blob = " ".join(p for p in parts if p)
    return len(q_toks & {t for t in tokenize(blob) if len(t) >= 3})


def _is_claim_howto(question: str, history: list[dict[str, str]] | None = None) -> bool:
    blob = question or ""
    slots = extract_thread_slots(history, question)
    if slots.damages or slots.wants_claim or slots.wants_form:
        return True
    return any(w in blob for w in (
        "תביע", "נזילה", "נזיל", "דוד", "צנרת", "שרברב", "נזקי מים", "נזק למבנה", "שכן", "טופס",
    ))


def _is_action_hit(hit: Hit, question: str = "", history: list[dict[str, str]] | None = None) -> bool:
    """Self-service how-to page: claims URLs for claim questions, else a deep /customer-service/ page."""
    path = _url_path(hit)
    parts = [p for p in path.strip("/").split("/") if p]
    if _is_claim_howto(question, history):
        return path.startswith("/claims-customer-service/")
    return path.startswith("/customer-service/") and hit.section_type == "service" and len(parts) >= 3


def promote_claim_path_hits(
    question: str,
    hits: list[Hit],
    history: list[dict[str, str]] | None = None,
    *,
    limit: int = 3,
) -> list[Hit]:
    """Boost apartment/car claim pages that match thread damage type (plumber, boiler, third party)."""
    if not hits:
        return hits
    markers = claim_path_markers(question, history)
    if not markers:
        return hits
    chosen: list[Hit] = []
    rest: list[Hit] = []
    seen: set[str] = set()
    for h in hits:
        path = _url_path(h)
        url = (h.display_url or h.url or "").rstrip("/")
        if len(chosen) < limit and any(m in path for m in markers) and url not in seen:
            chosen.append(h)
            if url:
                seen.add(url)
        else:
            rest.append(h)
    return chosen + rest if chosen else hits


def promote_action_hits(
    question: str,
    hits: list[Hit],
    *,
    limit: int = 2,
    history: list[dict[str, str]] | None = None,
) -> list[Hit]:
    """On how-to questions, move self-service action pages to the front of the list."""
    if not hits:
        return hits
    hits = promote_claim_path_hits(question, hits, history, limit=limit)
    if not _is_howto(question) and not _is_claim_howto(question, history):
        return hits
    chosen: list[Hit] = []
    rest: list[Hit] = []
    seen: set[str] = set()
    for h in hits:
        url = (h.display_url or h.url or "").rstrip("/")
        overlap = _title_overlap(question, h, with_body=True)
        min_ov = 1 if _is_claim_howto(question, history) else 2
        if (
            len(chosen) < limit
            and _is_action_hit(h, question, history)
            and overlap >= min_ov
            and url not in seen
        ):
            chosen.append(h)
            if url:
                seen.add(url)
        else:
            rest.append(h)
    return chosen + rest if chosen else hits


def preferred_action_urls(question: str, hits: list[Hit]) -> list[str]:
    """Self-service action URLs worth citing when the user asked how to do something."""
    if not _is_howto(question):
        return []
    urls: list[str] = []
    for h in hits:
        if not _is_action_hit(h, question) or _title_overlap(question, h, with_body=True) < 2:
            continue
        u = h.display_url or h.url
        if u and u not in urls:
            urls.append(u)
    return urls


def _canonical_product_hits(question: str, hits: list[Hit]) -> list[Hit]:
    """Product landing pages whose title overlaps the question; shorter URL wins."""
    scored: list[tuple[int, int, Hit]] = []
    seen: set[str] = set()
    for h in hits:
        if h.section_type != "product":
            continue
        ov = _title_overlap(question, h)
        if ov < 3:
            continue
        path = _url_path(h)
        if any(x in path for x in ("/faq/", "/articles/", "what-is-and-is-not")):
            continue
        url = (h.display_url or h.url or "").rstrip("/")
        if not url or url in seen:
            continue
        seen.add(url)
        parts = [p for p in path.strip("/").split("/") if p]
        scored.append((ov, len(parts), h))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    specific = [h for ov, n, h in scored if n >= 2]
    chosen = specific or [h for _, _, h in scored]
    qlow = (question or "").lower()
    if "טסלה" in (question or "") or "tesla" in qlow:
        tesla = [h for h in chosen if "tesla" in _url_path(h)]
        if tesla:
            return tesla
    return chosen


def demote_mismatched_paths(question: str, hits: list[Hit]) -> list[Hit]:
    """Apartment/home questions should not rank business-liability pages first."""
    q = question or ""
    if "דירה" in q and "עסק" not in q:
        bad_markers = (
            "/claims-business-insurance/",
            "/claims-liability-insurance/business",
            "business-water",
            "business-burglary",
            "business-fire",
        )
        good, tail = [], []
        for h in hits:
            path = _url_path(h)
            if any(m in path for m in bad_markers):
                tail.append(h)
            else:
                good.append(h)
        if good:
            return good + tail
    qlow = q.lower()
    if ("טסלה" in q or "tesla" in qlow) and "טעינ" not in q and "charging" not in qlow:
        tesla, rest, chargers = [], [], []
        for h in hits:
            path = _url_path(h)
            if "tesla" in path or "טסלה" in (h.page_title or ""):
                tesla.append(h)
            elif "charging-station" in path or "charging" in path:
                chargers.append(h)
            else:
                rest.append(h)
        ordered = tesla + rest + chargers
        if tesla or rest:
            return ordered
    return hits


def promote_topic_hits(
    question: str,
    hits: list[Hit],
    *,
    limit: int = 2,
    history: list[dict[str, str]] | None = None,
) -> list[Hit]:
    """Move the most relevant action or product landing page to the front of the list."""
    if not hits:
        return hits
    hits = promote_claim_path_hits(question, hits, history, limit=max(limit, 3))
    if _is_howto(question) or _is_claim_howto(question, history):
        return promote_action_hits(question, hits, limit=limit, history=history)
    if not _is_define(question):
        return hits
    chosen = _canonical_product_hits(question, hits)[:limit]
    if not chosen:
        return hits
    ids = {h.chunk_id for h in chosen}
    rest = [h for h in hits if h.chunk_id not in ids]
    return chosen + rest


def preferred_cite_urls(question: str, hits: list[Hit]) -> list[str]:
    """URLs the answer should cite: action pages on how-to, product pages on 'what is / what includes'."""
    if _is_howto(question):
        return preferred_action_urls(question, hits)
    if not _is_define(question):
        return []
    urls: list[str] = []
    for h in _canonical_product_hits(question, hits)[:2]:
        u = h.display_url or h.url
        if u and u not in urls:
            urls.append(u)
    return urls


def drop_weak_hits(
    hits: list[Hit],
    *,
    relative: float = 0.08,
    floor: float = 0.03,
    refuse_below: float = 0.0,
) -> list[Hit]:
    """Keep Cohere hits near the best score; empty list means 'too weak to answer'.

    When rerank did not assign scores, the hybrid order is kept. `refuse_below`
    is the peak-score gate (config `rerank.min_score`); it does not force-keep
    a leftover weak hit.
    """
    scored = [h.rerank_score for h in hits if h.rerank_score]
    if not scored:
        return hits
    peak = max(scored)
    if refuse_below > 0 and peak < refuse_below:
        return []
    if peak < floor:
        return hits
    thresh = max(floor, peak * relative)
    return [h for h in hits if h.rerank_score >= thresh]


def _doc_text(hit: Hit) -> str:
    url = hit.display_url or hit.url
    kind = _TYPE_HE.get(hit.section_type, hit.section_type)
    head = " | ".join(p for p in (hit.page_title, hit.heading_path) if p)
    body = (hit.text or "").strip()
    extra = f"\nטפסים:\n{hit.form_links}" if hit.form_links else ""
    cue = ""
    if hit.section_type == "service" and "/customer-service/" in (url or ""):
        cue = "דף פעולה לביצוע בשירות עצמי (לא FAQ, לא הצטרפות, לא מאמר).\n"
    return f"{cue}כתובת: {url}\nסוג עמוד: {kind}\nכותרת: {head}\n{body}{extra}"[:4000]


def rerank(question: str, hits: list[Hit], *, model: str, top_n: int, timeout: float = 30.0) -> list[Hit]:
    """Reorder `hits` by Cohere relevance. Returns at most `top_n` hits (or the original list on failure)."""
    global _disabled_reason
    if not hits:
        return []
    if _disabled_reason:
        return hits[:top_n]
    key = (os.getenv("COHERE_API_KEY") or "").strip()
    if not key:
        _disabled_reason = "COHERE_API_KEY missing"
        log.warning("rerank skipped: %s", _disabled_reason)
        return hits[:top_n]

    docs = [_doc_text(h) for h in hits]
    query = question
    if _is_howto(question):
        query = f"{question}\nחפש דף פעולה / שירות עצמי לביצוע הפעולה, לא FAQ ולא דף הצטרפות."
    payload_v2 = {"model": model, "query": query, "documents": [{"text": t} for t in docs], "top_n": min(top_n, len(docs))}
    payload_v1 = {"model": model, "query": query, "documents": docs, "top_n": min(top_n, len(docs))}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    delay = 2.0
    last_err = ""
    for attempt in range(1, 5):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                resp = client.post(_V2, headers=headers, json=payload_v2)
                if resp.status_code in (404, 400):
                    resp = client.post(_V1, headers=headers, json=payload_v1)
            if resp.status_code == 418 or "blockByNetFree" in resp.text:
                _disabled_reason = f"NetFree blocked Cohere ({resp.status_code})"
                log.warning("rerank disabled: %s", _disabled_reason)
                return hits[:top_n]
            if resp.status_code == 429:
                wait = _retry_after(resp) or delay
                log.warning("rerank 429, sleep %.1fs (attempt %d)", wait, attempt)
                time.sleep(wait)
                delay = min(delay * 2, 30)
                continue
            if resp.status_code in (401, 403):
                _disabled_reason = f"Cohere auth failed ({resp.status_code})"
                log.warning("rerank disabled: %s", _disabled_reason)
                return hits[:top_n]
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("results") or []
            out: list[Hit] = []
            for row in rows:
                idx = int(row.get("index", -1))
                if idx < 0 or idx >= len(hits):
                    continue
                hit = hits[idx]
                hit.rerank_score = float(row.get("relevance_score") or 0.0)
                hit.score = hit.rerank_score
                out.append(hit)
            return out[:top_n] or hits[:top_n]
        except httpx.HTTPError as exc:
            last_err = str(exc)
            log.warning("rerank attempt %d failed: %s", attempt, last_err[:200])
            time.sleep(delay)
            delay = min(delay * 2, 30)
    log.warning("rerank gave up (%s); keeping hybrid order", last_err[:200])
    return hits[:top_n]


def _retry_after(resp: httpx.Response) -> float:
    raw = resp.headers.get("retry-after") or resp.headers.get("x-ratelimit-reset")
    try:
        return float(raw) if raw else 0.0
    except ValueError:
        return 0.0
