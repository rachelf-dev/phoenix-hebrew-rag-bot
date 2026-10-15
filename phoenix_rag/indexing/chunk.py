"""Heading-aware chunker (spec 6.3).

Split by heading structure first, then by size. FAQ pairs, numbered procedures and
tables stay intact when they fit. Every chunk is prefixed with page title + heading
path for embedding; the original site text is stored separately for quoting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator  # noqa: I001

import tiktoken

from ..cleaning.clean import Block, CleanPage, Link
from ..config import Config
from ..crawler.urls import url_id

log = logging.getLogger(__name__)

_ENC = tiktoken.get_encoding("cl100k_base")
_SENT_SPLIT = re.compile(r"(?<=[\.!?…։۔。])\s+|(?<=\n)")
ATOMIC = {"faq", "table", "steps"}


def ntokens(text: str) -> int:
    if not text:
        return 0
    return len(_ENC.encode(text))


@dataclass
class Chunk:
    chunk_id: str
    url: str
    display_url: str
    page_title: str
    heading_path: str
    fetch_date: str | None
    content_hash: str | None
    section_type: str
    language: str
    original_text: str          # exactly the site text, for quoting
    embed_text: str             # context-prefixed, used for the vector
    token_count: int
    block_types: str            # comma-separated
    form_links: str = ""        # "label|href" lines, for "where to submit" answers

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class _Unit:
    text: str
    btype: str
    heading_path: list[str]
    tokens: int = 0

    def __post_init__(self) -> None:
        self.tokens = ntokens(self.text)


def _force_cap(prefix: str, text: str, max_tok: int, overlap: int) -> list[str]:
    """Guarantee prefix+text stays <= max_tok by cutting on token ids if needed."""
    if ntokens(prefix + text) <= max_tok:
        return [text]
    room = max(max_tok - ntokens(prefix) - 1, 32)
    ids = _ENC.encode(text)
    step = max(room - overlap, 32)
    return [_ENC.decode(ids[i : i + room]) for i in range(0, len(ids), step)] or [text]


def context_prefix(title: str, heading_path: list[str]) -> str:
    path = " > ".join(heading_path) if heading_path else title
    return f"כותרת: {title}\nנתיב: {path}\n\n"


def _useful_links(links: list[Link]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for lk in links:
        if lk.kind not in ("pdf", "form") or not lk.href or lk.href in seen:
            continue
        seen.add(lk.href)
        label = lk.text or lk.href
        lines.append(f"{label}|{lk.href}")
        if len(lines) >= 8:
            break
    return "\n".join(lines)


def chunk_page(page: CleanPage, cfg: Config) -> list[Chunk]:
    if page.too_short:
        return []
    max_tok = int(cfg.chunking.get("max_tokens", 600))
    overlap = int(cfg.chunking.get("overlap_tokens", 60))
    keep_faq = bool(cfg.chunking.get("keep_faq_pairs_intact", True))

    units: list[_Unit] = []
    for b in page.blocks:
        if b.type == "heading" or not (b.text or "").strip():
            continue
        units.append(_Unit(text=b.text.strip(), btype=b.type, heading_path=list(b.heading_path or [])))

    packed = _pack(units, max_tok=max_tok, overlap=overlap, keep_faq=keep_faq)
    links = _useful_links(page.links)
    out: list[Chunk] = []
    uid = url_id(page.url)
    i = 0
    for path, text, btypes in packed:
        prefix = context_prefix(page.title, path)
        pieces = _force_cap(prefix, text, max_tok, overlap)
        for text_i in pieces:
            embed = prefix + text_i
            cid = hashlib.sha1(f"{page.url}\0{i}\0{text_i}".encode("utf-8")).hexdigest()[:16]
            out.append(
                Chunk(
                    chunk_id=f"{uid}:{i:04d}:{cid}",
                    url=page.url,
                    display_url=page.display_url,
                    page_title=page.title,
                    heading_path=" > ".join(path),
                    fetch_date=page.fetched_at,
                    content_hash=page.content_hash,
                    section_type=page.section_type,
                    language=page.language,
                    original_text=text_i,
                    embed_text=embed,
                    token_count=ntokens(embed),
                    block_types=",".join(btypes),
                    form_links=links,
                )
            )
            i += 1
    return out


def _pack(units: list[_Unit], *, max_tok: int, overlap: int, keep_faq: bool) -> list[tuple[list[str], str, list[str]]]:
    """Return (heading_path, original_text, block_types) tuples."""
    groups: list[list[_Unit]] = []
    current_key: tuple[str, ...] | None = None
    buf: list[_Unit] = []
    for u in units:
        key = tuple(u.heading_path)
        if current_key is None:
            current_key = key
        if key != current_key:
            if buf:
                groups.append(buf)
            buf = []
            current_key = key
        buf.append(u)
    if buf:
        groups.append(buf)

    out: list[tuple[list[str], str, list[str]]] = []
    for group in groups:
        path = group[0].heading_path
        pieces = _fill(group, max_tok=max_tok, overlap=overlap, keep_faq=keep_faq)
        out.extend((path, text, types) for text, types in pieces)
    return out


def _fill(group: list[_Unit], *, max_tok: int, overlap: int, keep_faq: bool) -> list[tuple[str, list[str]]]:
    # Reserve room for the title/heading_path prefix that is prepended later.
    budget = max(max_tok - 80, 120)
    chunks: list[tuple[str, list[str]]] = []
    bag: list[_Unit] = []
    bag_tok = 0

    def emit() -> None:
        nonlocal bag, bag_tok
        if not bag:
            return
        chunks.append(("\n\n".join(u.text for u in bag), [u.btype for u in bag]))
        bag, bag_tok = [], 0

    for u in group:
        atomic = (u.btype == "faq" and keep_faq) or u.btype in ("table", "steps")
        if u.tokens > budget:
            emit()
            if atomic and u.btype == "faq":
                chunks.extend(_split_faq(u, budget))
            else:
                chunks.extend(_split_text(u.text, u.btype, budget, overlap))
            continue
        extra = u.tokens + (2 if bag else 0)
        if bag and bag_tok + extra > budget:
            emit()
        bag.append(u)
        bag_tok += extra
    emit()
    return chunks


def _split_faq(u: _Unit, max_tok: int) -> list[tuple[str, list[str]]]:
    q, a = u.text, ""
    if "\nתשובה:" in u.text:
        q, a = u.text.split("\nתשובה:", 1)
        a = "תשובה:" + a
    parts = _window(a or u.text, max_tok - ntokens(q) - 2)
    if not parts:
        return [(u.text, ["faq"])]
    out = []
    for i, p in enumerate(parts):
        head = q if i == 0 else q + "\n(המשך)"
        out.append((head + "\n" + p, ["faq"]))
    return out


def _split_text(text: str, btype: str, max_tok: int, overlap: int) -> list[tuple[str, list[str]]]:
    return [(p, [btype]) for p in _window(text, max_tok, overlap)]


def _window(text: str, max_tok: int, overlap: int = 40) -> list[str]:
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    if not sents:
        # last resort: hard-cut by tokens
        ids = _ENC.encode(text)
        step = max(max_tok - overlap, 1)
        return [_ENC.decode(ids[i : i + max_tok]) for i in range(0, len(ids), step)]
    out: list[str] = []
    bag: list[str] = []
    tok = 0
    for s in sents:
        st = ntokens(s)
        if st > max_tok:
            if bag:
                out.append(" ".join(bag))
                bag, tok = [], 0
            ids = _ENC.encode(s)
            for i in range(0, len(ids), max_tok):
                out.append(_ENC.decode(ids[i : i + max_tok]))
            continue
        if bag and tok + st > max_tok:
            out.append(" ".join(bag))
            # overlap last sentence if it fits
            if overlap and bag:
                last = bag[-1]
                bag, tok = [last], ntokens(last)
            else:
                bag, tok = [], 0
        bag.append(s)
        tok += st
    if bag:
        out.append(" ".join(bag))
    return out


def load_clean_pages(clean_dir: Path, *, skip_short: bool = True) -> Iterator[CleanPage]:
    for path in sorted(clean_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        page = CleanPage.from_dict(raw)
        if skip_short and page.too_short:
            continue
        yield page


def write_chunks(chunks: list[Chunk], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(c.to_json() + "\n")


def read_chunks(path: Path) -> Iterator[Chunk]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield Chunk(**json.loads(line))
