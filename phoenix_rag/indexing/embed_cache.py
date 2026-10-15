"""On-disk embedding cache (sqlite) keyed by model, dimension and a hash of the embedded text, so a long,
rate-limited index run can be interrupted and resumed without re-embedding what is already done."""

from __future__ import annotations

import array
import hashlib
import sqlite3
from pathlib import Path
from typing import Iterable


class EmbeddingCache:
    def __init__(self, path: Path, model: str, dim: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model, self.dim = model, dim
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB NOT NULL)")
        self.conn.commit()

    def key(self, text: str) -> str:
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return f"{self.model}:{self.dim}:{h}"

    def get_many(self, texts: Iterable[str]) -> dict[str, list[float]]:
        """Return {text_key: vector} for every text already cached."""
        keys = [self.key(t) for t in texts]
        found: dict[str, list[float]] = {}
        for i in range(0, len(keys), 900):  # sqlite parameter limit
            sl = keys[i : i + 900]
            q = f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(sl))})"
            for k, blob in self.conn.execute(q, sl):
                a = array.array("f")
                a.frombytes(blob)
                found[k] = a.tolist()
        return found

    def put_many(self, texts: list[str], vectors: list[list[float]]) -> None:
        rows = [(self.key(t), array.array("f", v).tobytes()) for t, v in zip(texts, vectors)]
        self.conn.executemany("INSERT OR REPLACE INTO emb (key, vec) VALUES (?, ?)", rows)
        self.conn.commit()

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM emb WHERE key LIKE ?", (f"{self.model}:{self.dim}:%",)).fetchone()[0])

    def close(self) -> None:
        self.conn.close()
