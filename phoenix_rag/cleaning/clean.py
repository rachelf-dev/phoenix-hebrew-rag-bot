"""HTML -> structured clean page (spec 6.2).

Keeps: title, H1-H4 hierarchy, paragraphs, lists, tables (as markdown), FAQ question/answer
pairs (kept together), links to forms / PDFs as metadata.
Drops: navigation, header/footer, breadcrumbs (kept as metadata), contact widgets, lead forms,
scripts, styles, repeated legal disclaimer (kept once as metadata).

Every block carries its ``heading_path`` so the chunker can prefix chunks with context.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Iterator
from urllib.parse import urlsplit

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from ..config import Config
from ..crawler.urls import display_url
from .hebrew import normalize_inline, normalize_text

MAIN_RE = re.compile(r"<main\b.*?</main>", re.S | re.I)
TITLE_SUFFIX_RE = re.compile(r"\s*\|\s*הפניקס\s*$")
LEAF_TAGS = {"p", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre"}
INLINE_TAGS = {"a", "span", "strong", "b", "em", "i", "u", "br", "sup", "sub", "small", "mark", "img", "button"}


@dataclass
class Block:
    type: str                      # heading | paragraph | list | steps | table | faq
    text: str                      # rendered text (markdown-ish for tables/lists)
    heading_path: list[str] = field(default_factory=list)
    level: int | None = None       # headings only
    question: str | None = None    # faq only
    answer: str | None = None      # faq only
    items: list[str] | None = None # list / steps only


@dataclass
class Link:
    text: str
    href: str
    kind: str                      # pdf | form | internal | external


@dataclass
class CleanPage:
    url: str
    display_url: str
    canonical_url: str | None
    title: str
    h1: str | None
    breadcrumbs: list[str]
    section_type: str
    fetched_at: str | None
    content_hash: str | None
    language: str
    blocks: list[Block]
    links: list[Link]
    disclaimer: str | None
    text_chars: int
    too_short: bool
    iframe_srcs: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)

    def plain_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    @classmethod
    def from_dict(cls, raw: dict) -> "CleanPage":
        blocks = [Block(**{k: b.get(k) for k in Block.__dataclass_fields__}) for b in raw.get("blocks") or []]
        links = [Link(**{k: lk.get(k) for k in Link.__dataclass_fields__}) for lk in raw.get("links") or []]
        return cls(
            url=raw["url"],
            display_url=raw.get("display_url") or raw["url"],
            canonical_url=raw.get("canonical_url"),
            title=raw.get("title") or raw["url"],
            h1=raw.get("h1"),
            breadcrumbs=raw.get("breadcrumbs") or [],
            section_type=raw.get("section_type") or "general",
            fetched_at=raw.get("fetched_at"),
            content_hash=raw.get("content_hash"),
            language=raw.get("language") or "he",
            blocks=blocks,
            links=links,
            disclaimer=raw.get("disclaimer"),
            text_chars=int(raw.get("text_chars") or 0),
            too_short=bool(raw.get("too_short")),
            iframe_srcs=raw.get("iframe_srcs") or [],
        )


# ----------------------------------------------------------------------------- public API
def clean_html(html_text: str, *, url: str, cfg: Config, section_type: str, canonical_url: str | None = None,
               fetched_at: str | None = None, content_hash: str | None = None) -> CleanPage:
    m = MAIN_RE.search(html_text)
    main_html = m.group(0) if m else html_text
    soup = BeautifulSoup(main_html, "lxml")
    main = soup.find("main") or soup

    head = BeautifulSoup(html_text[: max(html_text.find("</head>"), 0) or 100_000], "lxml")
    title = normalize_inline(head.title.get_text() if head.title else "")
    title = TITLE_SUFFIX_RE.sub("", title)
    lang = (head.find("html") or {}).get("lang") if head.find("html") else None

    breadcrumbs = _extract_breadcrumbs(main)
    links = _extract_links(main, url)
    # Some pages embed their real content in an iframe (legacy .aspx under /SPF/, calculators);
    # that content is not crawlable (robots.txt) so we only record the fact for the limitations report.
    iframe_srcs = [html.unescape(i.get("src", "")) for i in main.find_all("iframe") if i.get("src")]
    _drop_noise(main, cfg)

    h1_tag = main.find("h1")
    h1 = normalize_inline(h1_tag.get_text(" ")) if h1_tag else None

    walker = _Walker(cfg)
    blocks = list(walker.walk(main))
    blocks, disclaimer = _pull_disclaimer(blocks, cfg)
    blocks = _dedupe(blocks)

    text_chars = sum(len(b.text) for b in blocks if b.type != "heading")
    faq_chars = sum(len(b.text) for b in blocks if b.type == "faq")
    if faq_chars and faq_chars >= 0.5 * max(text_chars, 1):
        section_type = "faq"
    min_chars = int(cfg.crawl.get("min_clean_chars", 200))
    too_short = text_chars < min_chars and section_type not in ("faq", "form")

    return CleanPage(
        url=url,
        display_url=display_url(url),
        canonical_url=canonical_url,
        title=title or h1 or url,
        h1=h1,
        breadcrumbs=breadcrumbs,
        section_type=section_type,
        fetched_at=fetched_at,
        content_hash=content_hash,
        language=(lang or "he").split("-")[0],
        blocks=blocks,
        links=links,
        disclaimer=disclaimer,
        text_chars=text_chars,
        too_short=too_short,
        iframe_srcs=iframe_srcs,
    )


# ----------------------------------------------------------------------------- helpers
def _extract_breadcrumbs(main: Tag) -> list[str]:
    nav = main.select_one("nav.breadcrumbs")
    if not nav:
        return []
    crumbs = [normalize_inline(li.get_text(" ")) for li in nav.find_all("li")]
    return [c for c in crumbs if c]


def _extract_links(main: Tag, page_url: str) -> list[Link]:
    by_href: dict[str, Link] = {}
    page_host = urlsplit(page_url).hostname or ""
    for a in main.find_all("a", href=True):
        href = html.unescape(a["href"].strip())
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        text = normalize_inline(a.get_text(" ")) or normalize_inline(a.get("title") or "")
        if text == href:
            text = ""
        if href in by_href:
            if text and not by_href[href].text:
                by_href[href].text = text
            continue
        host = (urlsplit(href).hostname or page_host).lower()
        low = href.lower()
        if low.endswith(".pdf") or "/documents/" in low or "/sites/docs/" in low:
            kind = "pdf"
        elif host != page_host and any(k in low for k in ("claim", "form", "tofes", "natav", "smart.fnx", "my.fnx")):
            kind = "form"
        elif host.endswith("fnx.co.il"):
            kind = "internal"
        else:
            kind = "external"
        by_href[href] = Link(text=text, href=href, kind=kind)
    return list(by_href.values())


def _is_site_wide_contacts(tag: Tag) -> bool:
    """Drop only the site-wide contact matrix, not per-page 'נשמח לשוחח' blocks."""
    text = normalize_text(tag.get_text(" "))
    if "כל הדרכים לדבר איתנו" in text or "כל הדרכים ליצור איתנו קשר" in text:
        return True
    if text.count("054-9893455") >= 4 and "מוקד שירות" in text:
        return True
    return False


def _drop_noise(main: Tag, cfg: Config) -> None:
    for sel in cfg.cleaning.get("drop_selectors", []):
        for t in main.select(sel):
            t.decompose()
    drop_classes = cfg.cleaning.get("drop_section_classes", [])
    for t in list(main.find_all(True, class_=True)):
        if t.decomposed or t.attrs is None:
            continue
        classes = " ".join(t.get("class", []))
        if "contacts-section" in classes:
            if _is_site_wide_contacts(t):
                t.decompose()
            continue
        if any(dc in classes for dc in drop_classes):
            t.decompose()
    # Mobile/desktop duplicates: responsive variants hidden on large screens are copies.
    for t in list(main.select(".d-lg-none")):
        if not t.decomposed:
            t.decompose()


class _Walker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.stack: list[tuple[int, str]] = []  # (level, text)

    def path(self) -> list[str]:
        return [t for _, t in self.stack]

    def _push_heading(self, level: int, text: str) -> None:
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text))

    def walk(self, node: Tag) -> Iterator[Block]:
        for child in list(node.children):
            if isinstance(child, NavigableString):
                if isinstance(child, Comment):
                    continue
                txt = normalize_text(str(child))
                if len(txt) > 1:
                    yield Block("paragraph", txt, self.path())
                continue
            if not isinstance(child, Tag):
                continue
            classes = " ".join(child.get("class", []))
            if "collapse-group-item" in classes:
                blk = self._faq(child)
                if blk:
                    yield blk
                continue
            name = child.name.lower()
            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                text = normalize_inline(child.get_text(" "))
                if text:
                    level = int(name[1])
                    self._push_heading(level, text)
                    yield Block("heading", text, self.path()[:-1], level=level)
                continue
            if name in ("ul", "ol"):
                items = [normalize_inline(li.get_text(" ")) for li in child.find_all("li", recursive=False)]
                items = [i for i in items if i]
                if items:
                    btype = "steps" if name == "ol" else "list"
                    bullet = (lambda i, s: f"{i + 1}. {s}") if name == "ol" else (lambda i, s: f"- {s}")
                    yield Block(btype, "\n".join(bullet(i, s) for i, s in enumerate(items)), self.path(), items=items)
                continue
            if name == "table":
                text = _table_to_markdown(child)
                if text:
                    yield Block("table", text, self.path())
                continue
            if name in ("p", "blockquote", "pre") or (name == "div" and _is_text_leaf(child)):
                text = normalize_text(child.get_text(" "))
                if not text or text.isdigit() or len(text) < 3:
                    continue  # step-number bubbles ("1", "2") and stray punctuation
                # Rich-text editor "subtitle" spans are the site's way of writing sub-headings.
                sub_level = _rte_subtitle_level(child)
                if sub_level and len(text) <= 120:
                    self._push_heading(sub_level, text)
                    yield Block("heading", text, self.path()[:-1], level=sub_level)
                else:
                    yield Block("paragraph", text, self.path())
                continue
            if name in INLINE_TAGS:
                text = normalize_inline(child.get_text(" ")) if name != "img" else ""
                if text and name not in ("a", "button"):
                    yield Block("paragraph", text, self.path())
                continue
            yield from self.walk(child)

    def _faq(self, item: Tag) -> Block | None:
        q_tag = item.select_one(".question") or item.select_one(".collapse-group-head")
        a_tag = item.select_one(".answer") or item.select_one(".collapse-group-body")
        q = normalize_inline(q_tag.get_text(" ")) if q_tag else ""
        if a_tag:
            sub = _Walker(self.cfg)
            parts = [b.text for b in sub.walk(a_tag)]
            a = normalize_text("\n".join(parts)) if parts else normalize_text(a_tag.get_text("\n"))
        else:
            a = ""
        if not q or not a:
            return None
        return Block("faq", f"שאלה: {q}\nתשובה: {a}", self.path(), question=q, answer=a)


_RTE_SUBTITLE_RE = re.compile(r"rte-subtitle-(\d)")


def _rte_subtitle_level(p: Tag) -> int | None:
    """<p><span class="rte-subtitle-2">…</span></p>  ->  heading level 3 (subtitle-1 -> 2, subtitle-3 -> 4)."""
    children = [c for c in p.children if not (isinstance(c, NavigableString) and not str(c).strip())]
    if len(children) != 1 or not isinstance(children[0], Tag):
        return None
    m = _RTE_SUBTITLE_RE.search(" ".join(children[0].get("class", [])))
    if not m:
        return None
    return min(int(m.group(1)) + 1, 6)


def _is_text_leaf(div: Tag) -> bool:
    """A div with inline content only (no block children) is treated as a paragraph."""
    for c in div.children:
        if isinstance(c, Tag) and c.name.lower() not in INLINE_TAGS:
            return False
    return bool(normalize_inline(div.get_text(" ")))


def _table_to_markdown(table: Tag) -> str:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [normalize_inline(td.get_text(" ")) for td in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _pull_disclaimer(blocks: list[Block], cfg: Config) -> tuple[list[Block], str | None]:
    prefixes = tuple(cfg.cleaning.get("boilerplate_prefixes", []))
    kept: list[Block] = []
    disclaimer: str | None = None
    for b in blocks:
        if b.type == "paragraph" and prefixes and b.text.startswith(prefixes):
            disclaimer = disclaimer or b.text
            continue
        kept.append(b)
    return kept, disclaimer


def _dedupe(blocks: list[Block]) -> list[Block]:
    seen: set[tuple[str, str]] = set()
    out: list[Block] = []
    for b in blocks:
        key = (b.type, b.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
    # drop headings that have no content below them (e.g. widget titles)
    pruned: list[Block] = []
    for i, b in enumerate(out):
        if b.type == "heading":
            nxt = out[i + 1] if i + 1 < len(out) else None
            if nxt is None or (nxt.type == "heading" and (nxt.level or 9) <= (b.level or 1)):
                continue
        pruned.append(b)
    return pruned
