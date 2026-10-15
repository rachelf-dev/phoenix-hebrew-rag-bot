"""SQLite page store: one row per normalised URL + raw HTML on disk.

Spec 6.1: "For each page store: URL, canonical URL, title, HTTP status, fetch timestamp,
content hash (for change detection), raw HTML. Support incremental refresh ... Log skipped,
failed, and blocked URLs."
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    url             TEXT PRIMARY KEY,
    url_id          TEXT NOT NULL,
    canonical_url   TEXT,
    title           TEXT,
    http_status     INTEGER,
    fetched_at      TEXT,
    content_hash    TEXT,            -- sha256 of <main> HTML (scripts removed)
    sitemap_lastmod TEXT,
    etag            TEXT,
    last_modified   TEXT,
    raw_path        TEXT,
    section_type    TEXT,
    priority        INTEGER,
    discovered_from TEXT,            -- 'sitemap' or the URL where the link was found
    state           TEXT NOT NULL,   -- active | removed | failed | blocked | skipped
    error           TEXT,
    last_seen_run   TEXT,            -- run id in which the URL was last listed/discovered
    changed_in_run  TEXT             -- run id in which the content hash last changed
);
CREATE INDEX IF NOT EXISTS idx_pages_state ON pages(state);
CREATE INDEX IF NOT EXISTS idx_pages_section ON pages(section_type);

CREATE TABLE IF NOT EXISTS crawl_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT NOT NULL,
    ts        TEXT NOT NULL,
    url       TEXT NOT NULL,
    event     TEXT NOT NULL,         -- fetched | unchanged | skipped | blocked | failed | removed | excluded
    detail    TEXT
);
CREATE INDEX IF NOT EXISTS idx_log_run ON crawl_log(run_id, event);

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    mode        TEXT,                -- full | incremental | test
    stats       TEXT                 -- json
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class PageRow:
    url: str
    url_id: str
    canonical_url: str | None = None
    title: str | None = None
    http_status: int | None = None
    fetched_at: str | None = None
    content_hash: str | None = None
    sitemap_lastmod: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    raw_path: str | None = None
    section_type: str | None = None
    priority: int | None = None
    discovered_from: str | None = None
    state: str = "skipped"
    error: str | None = None
    last_seen_run: str | None = None
    changed_in_run: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "PageRow":
        return cls(**{k: row[k] for k in row.keys()})


class PageStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # --- runs -----------------------------------------------------------------
    def start_run(self, run_id: str, mode: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO runs(run_id, started_at, mode) VALUES (?,?,?)", (run_id, utcnow(), mode)
        )
        self.conn.commit()

    def finish_run(self, run_id: str, stats_json: str) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at=?, stats=? WHERE run_id=?", (utcnow(), stats_json, run_id)
        )
        self.conn.commit()

    # --- pages ----------------------------------------------------------------
    def get(self, url: str) -> PageRow | None:
        row = self.conn.execute("SELECT * FROM pages WHERE url=?", (url,)).fetchone()
        return PageRow.from_row(row) if row else None

    def upsert(self, page: PageRow) -> None:
        cols = [k for k in page.__dataclass_fields__]
        placeholders = ",".join("?" for _ in cols)
        updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != "url")
        self.conn.execute(
            f"INSERT INTO pages({','.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(url) DO UPDATE SET {updates}",
            [getattr(page, c) for c in cols],
        )
        self.conn.commit()

    def iter_pages(self, state: str | None = "active") -> Iterator[PageRow]:
        if state:
            cur = self.conn.execute("SELECT * FROM pages WHERE state=? ORDER BY priority, url", (state,))
        else:
            cur = self.conn.execute("SELECT * FROM pages ORDER BY priority, url")
        for row in cur:
            yield PageRow.from_row(row)

    def mark_removed(self, run_id: str, keep_urls: Iterable[str]) -> list[str]:
        """Pages that were active but not listed/discovered in this full run are marked removed."""
        keep = set(keep_urls)
        removed: list[str] = []
        for row in self.conn.execute("SELECT url FROM pages WHERE state='active'"):
            if row["url"] not in keep:
                removed.append(row["url"])
        for url in removed:
            self.conn.execute("UPDATE pages SET state='removed' WHERE url=?", (url,))
            self.log(run_id, url, "removed", "not present in sitemap/discovery of a full run")
        self.conn.commit()
        return removed

    def counts(self) -> dict[str, int]:
        return {
            row["state"]: row["n"]
            for row in self.conn.execute("SELECT state, COUNT(*) AS n FROM pages GROUP BY state")
        }

    # --- log ------------------------------------------------------------------
    def log(self, run_id: str, url: str, event: str, detail: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO crawl_log(run_id, ts, url, event, detail) VALUES (?,?,?,?,?)",
            (run_id, utcnow(), url, event, detail),
        )

    def run_events(self, run_id: str) -> dict[str, int]:
        return {
            row["event"]: row["n"]
            for row in self.conn.execute(
                "SELECT event, COUNT(*) AS n FROM crawl_log WHERE run_id=? GROUP BY event", (run_id,)
            )
        }

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
