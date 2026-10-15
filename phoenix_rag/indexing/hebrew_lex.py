"""Lightweight Hebrew tokenisation for BM25 (spec 6.4: prefixes ה/ל/ב hurt pure lexical match)."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[א-ת]+|[a-z0-9]+", re.I)
_PREFIXES = tuple("הבלמכשוד")


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        out.append(tok)
        if len(tok) >= 4 and tok[0] in _PREFIXES:
            stem = tok[1:]
            if len(stem) >= 2:
                out.append(stem)
    return out
