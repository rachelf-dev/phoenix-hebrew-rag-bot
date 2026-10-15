"""robots.txt handling with wildcard support (Protego, the parser used by Scrapy)."""

from __future__ import annotations

import logging

import httpx
from protego import Protego

log = logging.getLogger(__name__)


class Robots:
    def __init__(self, base_url: str, user_agent: str, client: httpx.Client, respect: bool = True):
        self.user_agent = user_agent
        self.respect = respect
        self._rp: Protego | None = None
        if not respect:
            log.warning("respect_robots=false: robots.txt will be ignored (not recommended)")
            return
        url = base_url.rstrip("/") + "/robots.txt"
        try:
            resp = client.get(url)
            if resp.status_code == 200:
                self._rp = Protego.parse(resp.text)
                log.info("robots.txt loaded (%d bytes)", len(resp.text))
            else:
                log.warning("robots.txt returned HTTP %s; treating everything as allowed", resp.status_code)
        except httpx.HTTPError as exc:  # pragma: no cover - network
            log.warning("robots.txt fetch failed (%s); treating everything as allowed", exc)

    def allowed(self, url: str) -> bool:
        if not self.respect or self._rp is None:
            return True
        # Check both our UA token and the generic one; Protego picks the matching group.
        return self._rp.can_fetch(url, self.user_agent)

    def crawl_delay(self) -> float | None:
        if self._rp is None:
            return None
        return self._rp.crawl_delay(self.user_agent)

    def sitemaps(self) -> list[str]:
        if self._rp is None:
            return []
        return list(self._rp.sitemaps)
