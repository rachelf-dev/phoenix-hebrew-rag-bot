"""Polite, sitemap-driven crawler for fnx.co.il with incremental refresh.

Usage (via CLI):  python -m phoenix_rag.cli crawl [--limit N] [--prefix /claims-customer-service/] [--force]

Design
------
* Frontier = sitemap URLs (with <lastmod>) filtered by seeds.yaml, plus in-content links
  discovered inside <main> of fetched pages (same host, allowed prefixes).
* One request at a time, fixed delay between requests, descriptive User-Agent, robots.txt
  honoured (Protego, wildcard-aware).
* Incremental refresh: a URL is re-fetched only if its sitemap <lastmod> changed, or (when no
  lastmod is known) via conditional GET (ETag / If-Modified-Since).  After fetching, the
  content hash of the visible text in <main> decides whether the page is "changed".
* Raw HTML is stored gzip-compressed under data/raw/<sha1(url)>.html.gz.
* Filter/interstitial HTML (NetFree, Guard, empty foreign <main>) is stored as
  state=blocked and never indexed. Canonical and content-hash duplicates are
  marked state=duplicate at the end of the run so they do not reach chunk/index.
* Every decision is logged to the crawl_log table and to logs/crawl_<run>.log.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from ..config import Config
from .filters import filter_reason, plan_duplicates
from .robots import Robots
from .store import PageRow, PageStore, utcnow
from .urls import classify, normalize_url, url_id

log = logging.getLogger(__name__)

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
MAIN_RE = re.compile(r"<main\b.*?</main>", re.S | re.I)


@dataclass
class Candidate:
    url: str
    section_type: str
    priority: int
    lastmod: str | None
    discovered_from: str


class Crawler:
    def __init__(self, cfg: Config, *, limit: int = 0, prefix: str | None = None, force: bool = False,
                 discover: bool | None = None, retry_failed: bool = False, delay: float | None = None):
        self.cfg = cfg
        self.limit = limit or int(cfg.crawl.get("max_pages", 0) or 0)
        self.prefix = prefix
        self.force = force
        self.retry_failed = retry_failed
        self.discover = cfg.crawl.get("discover_links", True) if discover is None else discover
        self.delay = float(delay if delay is not None else cfg.crawl.get("delay_seconds", 1.5))
        self.timeout = float(cfg.crawl.get("timeout_seconds", 30))
        self.max_retries = int(cfg.crawl.get("max_retries", 3))
        self.ua = cfg.crawl["user_agent"]
        self.base_url = cfg.crawl["base_url"]
        cfg.ensure_dirs()
        self.store = PageStore(cfg.db_path)
        self.client = httpx.Client(
            headers={"User-Agent": self.ua, "Accept-Language": "he-IL,he;q=0.9"},
            timeout=self.timeout,
            follow_redirects=True,
        )
        self.robots = Robots(self.base_url, self.ua, self.client, respect=cfg.crawl.get("respect_robots", True))
        rd = self.robots.crawl_delay()
        if rd and rd > self.delay:
            log.info("robots.txt Crawl-delay=%s overrides configured delay %s", rd, self.delay)
            self.delay = float(rd)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.stats: Counter[str] = Counter()
        self._last_request_ts = 0.0
        self._seen: set[str] = set()
        self._fetched_count = 0

    # ------------------------------------------------------------------ public
    def run(self) -> dict:
        mode = "test" if (self.limit or self.prefix) else ("full" if self.force else "incremental")
        self.store.start_run(self.run_id, mode)
        log.info("run %s mode=%s delay=%.1fs discover=%s", self.run_id, mode, self.delay, self.discover)

        if self.retry_failed:
            mode = "test"
            frontier = {
                r.url: Candidate(r.url, r.section_type or "general", r.priority or 3, r.sitemap_lastmod, r.discovered_from or "retry")
                for r in self.store.iter_pages("failed")
            }
            log.info("retrying %d previously failed URLs", len(frontier))
        else:
            frontier = self._frontier_from_sitemap()
            log.info("frontier from sitemap: %d URLs in scope", len(frontier))
        queue: list[Candidate] = sorted(frontier.values(), key=lambda c: (c.priority, c.url))
        seen_in_run: set[str] = set(frontier)

        while queue:
            cand = queue.pop(0)
            if self.limit and self._fetched_count >= self.limit:
                log.info("limit of %d fetched pages reached", self.limit)
                break
            new_links = self._process(cand)
            for link in new_links:
                if link.url in seen_in_run:
                    continue
                seen_in_run.add(link.url)
                queue.append(link)
                queue.sort(key=lambda c: (c.priority, c.url))

        if mode != "test":
            removed = self.store.mark_removed(self.run_id, seen_in_run)
            self.stats["removed"] = len(removed)
            if removed:
                log.info("%d pages marked removed (no longer in sitemap/discovery)", len(removed))

        self._sweep_duplicates()

        stats = dict(self.stats)
        stats["store_counts"] = self.store.counts()
        self.store.finish_run(self.run_id, json.dumps(stats, ensure_ascii=False))
        log.info("done: %s", json.dumps(stats, ensure_ascii=False))
        self.client.close()
        self.store.close()
        return stats

    # --------------------------------------------------------------- frontier
    def _frontier_from_sitemap(self) -> dict[str, Candidate]:
        sitemap_urls = [self.cfg.crawl.get("sitemap_url")] or self.robots.sitemaps()
        found: dict[str, Candidate] = {}
        for sm_url in sitemap_urls:
            if not sm_url:
                continue
            for loc, lastmod in self._read_sitemap(sm_url):
                url = normalize_url(loc, self.cfg)
                if not url:
                    continue
                allowed, stype, prio, reason = classify(url, self.cfg)
                if allowed and self.prefix and not url.startswith(self.base_url.rstrip("/") + self.prefix):
                    self.stats["outside_prefix"] += 1  # test-run filter; not an exclusion worth logging
                    continue
                if not allowed:
                    self.stats["excluded"] += 1
                    self.store.log(self.run_id, url, "excluded", reason)
                    continue
                if url not in found:
                    found[url] = Candidate(url, stype, prio, lastmod, "sitemap")
        self.store.conn.commit()
        return found

    def _read_sitemap(self, sm_url: str) -> list[tuple[str, str | None]]:
        out: list[tuple[str, str | None]] = []
        resp = self._get(sm_url)
        if resp is None or resp.status_code != 200:
            log.error("sitemap %s not available", sm_url)
            return out
        root = ET.fromstring(resp.content)
        tag = root.tag.lower()
        if tag.endswith("sitemapindex"):
            for sm in root.findall("sm:sitemap/sm:loc", SITEMAP_NS):
                out.extend(self._read_sitemap(sm.text.strip()))
            return out
        for u in root.findall("sm:url", SITEMAP_NS):
            loc = u.find("sm:loc", SITEMAP_NS)
            lm = u.find("sm:lastmod", SITEMAP_NS)
            if loc is not None and loc.text:
                out.append((loc.text.strip(), lm.text.strip() if lm is not None and lm.text else None))
        log.info("sitemap %s: %d urls", sm_url, len(out))
        return out

    # ---------------------------------------------------------------- process
    def _process(self, cand: Candidate) -> list[Candidate]:
        url = cand.url
        if url in self._seen:
            return []
        self._seen.add(url)

        if not self.robots.allowed(url):
            self.stats["blocked"] += 1
            self.store.log(self.run_id, url, "blocked", "robots.txt")
            self.store.upsert(PageRow(url=url, url_id=url_id(url), state="blocked", section_type=cand.section_type,
                                      priority=cand.priority, discovered_from=cand.discovered_from,
                                      last_seen_run=self.run_id, sitemap_lastmod=cand.lastmod))
            return []

        existing = self.store.get(url)
        if existing and not self.force and existing.state == "active":
            if cand.lastmod and existing.sitemap_lastmod == cand.lastmod and existing.raw_path:
                existing.last_seen_run = self.run_id
                self.store.upsert(existing)
                self.stats["unchanged_lastmod"] += 1
                self.store.log(self.run_id, url, "unchanged", "sitemap lastmod identical")
                return self._links_from_disk(existing) if self.discover else []

        headers = {}
        if existing and not self.force and existing.state == "active":
            if existing.etag:
                headers["If-None-Match"] = existing.etag
            if existing.last_modified:
                headers["If-Modified-Since"] = existing.last_modified

        resp = self._get(url, headers=headers)
        self._fetched_count += 1
        if resp is None:
            self._fail(cand, existing, None, "network error after retries")
            return []
        if resp.status_code == 304 and existing:
            existing.last_seen_run = self.run_id
            existing.sitemap_lastmod = cand.lastmod or existing.sitemap_lastmod
            self.store.upsert(existing)
            self.stats["unchanged_304"] += 1
            self.store.log(self.run_id, url, "unchanged", "HTTP 304")
            return self._links_from_disk(existing) if self.discover else []
        if resp.status_code in (403, 418):
            self._block(cand, existing, resp.status_code, f"HTTP {resp.status_code} filter")
            return []
        if resp.status_code != 200:
            self._fail(cand, existing, resp.status_code, f"HTTP {resp.status_code}")
            return []
        if "text/html" not in resp.headers.get("content-type", ""):
            self._fail(cand, existing, resp.status_code, f"non-HTML content-type {resp.headers.get('content-type')}")
            return []

        final_url = normalize_url(str(resp.url), self.cfg) or url
        html_text = resp.text
        main_html = self._main_html(html_text)
        soup_main = BeautifulSoup(main_html, "lxml") if main_html else None
        main_text = self._main_text(soup_main)
        content_hash = hashlib.sha256(main_text.encode("utf-8")).hexdigest()
        title, canonical = self._title_and_canonical(html_text)

        if canonical:
            canonical = normalize_url(canonical, self.cfg)
        if final_url != url:
            self.store.log(self.run_id, url, "redirected", final_url)

        blocked = filter_reason(html_text, title, main_text)
        if blocked:
            raw_path = self._save_raw(url, html_text)
            self._block(
                cand, existing, resp.status_code, blocked,
                title=title, canonical=canonical or final_url,
                content_hash=content_hash, raw_path=str(raw_path.relative_to(self.cfg.root)),
            )
            return []

        changed = not existing or existing.content_hash != content_hash
        raw_path = self._save_raw(url, html_text)
        row = PageRow(
            url=url,
            url_id=url_id(url),
            canonical_url=canonical or final_url,
            title=title,
            http_status=resp.status_code,
            fetched_at=utcnow(),
            content_hash=content_hash,
            sitemap_lastmod=cand.lastmod,
            etag=resp.headers.get("etag"),
            last_modified=resp.headers.get("last-modified"),
            raw_path=str(raw_path.relative_to(self.cfg.root)),
            section_type=cand.section_type,
            priority=cand.priority,
            discovered_from=cand.discovered_from,
            state="active",
            error=None,
            last_seen_run=self.run_id,
            changed_in_run=self.run_id if changed else (existing.changed_in_run if existing else self.run_id),
        )
        self.store.upsert(row)
        event = "fetched" if changed else "unchanged"
        self.stats["fetched_changed" if changed else "fetched_unchanged"] += 1
        self.store.log(self.run_id, url, event, f"{len(main_html)} main bytes; title={title!r}")
        log.info("%s %s [%s] %s", event, url, cand.section_type, title)

        return self._extract_links(soup_main, url, cand.priority) if (self.discover and soup_main) else []

    def _block(
        self,
        cand: Candidate,
        existing: PageRow | None,
        status: int | None,
        msg: str,
        *,
        title: str | None = None,
        canonical: str | None = None,
        content_hash: str | None = None,
        raw_path: str | None = None,
    ) -> None:
        self.stats["blocked_filter"] += 1
        self.store.log(self.run_id, cand.url, "blocked_filter", msg)
        log.warning("blocked_filter %s: %s", cand.url, msg)
        row = existing or PageRow(url=cand.url, url_id=url_id(cand.url))
        row.http_status = status
        row.error = msg
        row.title = title or row.title
        row.canonical_url = canonical or row.canonical_url
        row.content_hash = content_hash or row.content_hash
        row.raw_path = raw_path or row.raw_path
        row.section_type = row.section_type or cand.section_type
        row.priority = row.priority or cand.priority
        row.discovered_from = row.discovered_from or cand.discovered_from
        row.last_seen_run = self.run_id
        row.state = "blocked"
        self.store.upsert(row)

    def _mark_duplicate(self, row: PageRow, event: str, detail: str) -> None:
        self.stats[event] += 1
        self.store.log(self.run_id, row.url, event, detail)
        log.info("%s %s (%s)", event, row.url, detail)
        self.store.set_state(row.url, "duplicate", detail)

    def _sweep_duplicates(self) -> None:
        """Collapse canonical aliases and identical <main> hashes among active pages."""
        actives = list(self.store.iter_pages("active"))
        by_url = {r.url: r for r in actives}
        for url, event, detail in plan_duplicates(
            [(r.url, r.canonical_url, r.content_hash) for r in actives]
        ):
            row = by_url.get(url)
            if row:
                self._mark_duplicate(row, event, detail)

    def _fail(self, cand: Candidate, existing: PageRow | None, status: int | None, msg: str) -> None:
        self.stats["failed"] += 1
        self.store.log(self.run_id, cand.url, "failed", msg)
        log.warning("failed %s: %s", cand.url, msg)
        row = existing or PageRow(url=cand.url, url_id=url_id(cand.url))
        row.http_status = status
        row.error = msg
        row.section_type = row.section_type or cand.section_type
        row.priority = row.priority or cand.priority
        row.discovered_from = row.discovered_from or cand.discovered_from
        row.last_seen_run = self.run_id
        # A previously good page that now fails keeps its content until a full run removes it.
        if not (existing and existing.state == "active" and existing.raw_path):
            row.state = "failed"
        self.store.upsert(row)

    # ------------------------------------------------------------------ http
    def _get(self, url: str, headers: dict | None = None) -> httpx.Response | None:
        for attempt in range(1, self.max_retries + 1):
            self._respect_delay()
            try:
                resp = self.client.get(url, headers=headers)
            except httpx.HTTPError as exc:
                log.warning("attempt %d/%d %s: %s", attempt, self.max_retries, url, exc)
                time.sleep(self.delay * attempt)
                continue
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                wait = self.delay * (2 ** attempt)
                log.warning("HTTP %s on %s; backing off %.1fs", resp.status_code, url, wait)
                time.sleep(wait)
                continue
            return resp
        return None

    def _respect_delay(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request_ts = time.monotonic()

    # --------------------------------------------------------------- parsing
    @staticmethod
    def _main_html(html_text: str) -> str:
        m = MAIN_RE.search(html_text)
        return m.group(0) if m else ""

    @staticmethod
    def _main_text(soup_main: BeautifulSoup | None) -> str:
        if soup_main is None:
            return ""
        for t in soup_main(["script", "style", "noscript", "svg"]):
            t.decompose()
        return re.sub(r"\s+", " ", soup_main.get_text(" ", strip=True))

    @classmethod
    def _hash_main(cls, soup_main: BeautifulSoup | None) -> str:
        text = cls._main_text(soup_main)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _title_and_canonical(html_text: str) -> tuple[str | None, str | None]:
        head_end = html_text.find("</head>")
        head = html_text[: head_end if head_end > 0 else 200_000]
        soup = BeautifulSoup(head, "lxml")
        title = soup.title.get_text(strip=True) if soup.title else None
        link = soup.find("link", rel=lambda v: v and "canonical" in v)
        canonical = link.get("href") if link else None
        return title, canonical

    def _extract_links(self, soup_main: BeautifulSoup, base: str, parent_priority: int) -> list[Candidate]:
        out: list[Candidate] = []
        for a in soup_main.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue
            url = normalize_url(urljoin(base, href), self.cfg)
            if not url or url in self._seen:
                continue
            allowed, stype, prio, _ = classify(url, self.cfg)
            if not allowed:
                continue
            if self.prefix and not url.startswith(self.base_url.rstrip("/") + self.prefix):
                continue
            out.append(Candidate(url, stype, max(prio, parent_priority), None, base))
        return out

    def _links_from_disk(self, row: PageRow) -> list[Candidate]:
        """For unchanged pages we still want their links in the frontier (cheap, no network)."""
        try:
            html_text = read_raw(self.cfg.root / row.raw_path)
        except OSError:
            return []
        main_html = self._main_html(html_text)
        if not main_html:
            return []
        return self._extract_links(BeautifulSoup(main_html, "lxml"), row.url, row.priority or 3)

    # --------------------------------------------------------------- storage
    def _save_raw(self, url: str, html_text: str) -> Path:
        path = self.cfg.raw_dir / f"{url_id(url)}.html.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(html_text)
        return path


def read_raw(path: Path) -> str:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return fh.read()
    return path.read_text(encoding="utf-8")
