"""Hebrew text normalisation (spec 6.2).

* keep nikud-free text as is (we do not add or strip nikud; it is rare on the site)
* normalise quote / geresh / gershayim variants
* remove zero-width and bidi control characters that break matching but keep RTL text intact
* collapse whitespace
"""

from __future__ import annotations

import re
import unicodedata

# Characters that are invisible and only confuse tokenisers / embeddings.
_INVISIBLE = re.compile(
    "[\u200b\u200c\u200d\u200e\u200f"  # zero-width space/joiners, LRM/RLM
    "\u202a\u202b\u202c\u202d\u202e"  # LRE RLE PDF LRO RLO
    "\u2066\u2067\u2068\u2069"  # LRI RLI FSI PDI
    "\ufeff\u00ad]"  # BOM, soft hyphen
)
_DOUBLE_QUOTES = {"\u05f4": '"', "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u00ab": '"', "\u00bb": '"', "\u2033": '"'}
_SINGLE_QUOTES = {"\u05f3": "'", "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u2032": "'", "\u00b4": "'", "\u0060": "'"}
_DASHES = {"\u2013": "-", "\u2014": "-", "\u2212": "-"}
_TRANS = str.maketrans({**_DOUBLE_QUOTES, **_SINGLE_QUOTES, **_DASHES, "\u00a0": " ", "\u2009": " ", "\u202f": " "})
_WS = re.compile(r"[ \t\r\f\v]+")
_MULTI_NL = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE.sub("", text)
    text = text.translate(_TRANS)
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def normalize_inline(text: str) -> str:
    """Single-line variant (headings, list items, table cells)."""
    return re.sub(r"\s+", " ", normalize_text(text)).strip()
