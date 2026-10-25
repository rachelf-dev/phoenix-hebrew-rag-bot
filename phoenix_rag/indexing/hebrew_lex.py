"""Lightweight Hebrew tokenisation for BM25 (spec 6.4: prefixes ה/ל/ב hurt pure lexical match).

Normalises common spelling variants (אויר/אוויר, defective/full spelling, final letters)
so queries and index text match the same token forms.
"""

from __future__ import annotations

import re

from ..cleaning.hebrew import normalize_text

_TOKEN = re.compile(r"[א-ת]+|[a-z0-9]+", re.I)
_PREFIXES = tuple("הבלמכשוד")

# Canonical map for whole-token substitution after normalization.
_SPELLING_VARIANTS = (
    ("אוויר", "אויר"),
    ("מזגאוויר", "מזגאויר"),
    ("ממזגאוויר", "ממזגאויר"),
)
_FINAL_FORMS = str.maketrans("ךםןףץ", "כמנפצ")


def normalize_hebrew_lex(text: str) -> str:
    """Collapse spelling variants before BM25 tokenisation."""
    s = normalize_text(text or "").casefold()
    s = s.translate(_FINAL_FORMS)
    for full, defect in _SPELLING_VARIANTS:
        s = s.replace(defect, full)
    # Common site typo / variant on claim pages
    s = s.replace("מזג אויר", "מזג אוויר")
    s = s.replace("ממזג אויר", "ממזג אוויר")
    s = s.replace("נזקי מזג אויר", "נזקי מזג אוויר")
    return s


def tokenize(text: str) -> list[str]:
    norm = normalize_hebrew_lex(text)
    out: list[str] = []
    for tok in _TOKEN.findall(norm):
        out.append(tok)
        if len(tok) >= 4 and tok[0] in _PREFIXES:
            stem = tok[1:]
            if len(stem) >= 2:
                out.append(stem)
    return out
