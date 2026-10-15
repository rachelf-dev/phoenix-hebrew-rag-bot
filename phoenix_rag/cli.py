"""Command line entry point.

    python -m phoenix_rag.cli map                  # classify sitemap URLs, print the area map (no page fetches)
    python -m phoenix_rag.cli crawl [--limit N] [--prefix /claims-customer-service/] [--force] [--no-discover]
    python -m phoenix_rag.cli clean [--url URL] [--limit N]   # raw HTML -> data/clean/*.json
    python -m phoenix_rag.cli chunk [--limit N]               # cleaned pages -> data/chunks.jsonl
    python -m phoenix_rag.cli index [--rechunk]               # embed + Chroma + BM25, then probe
    python -m phoenix_rag.cli probe [-k 12]                   # Hebrew retrieval smoke test
    python -m phoenix_rag.cli compare-embeddings [--models ...]   # embedding bake-off on a Hebrew subset
    python -m phoenix_rag.cli ask "שאלה" [--json]              # one grounded answer (rerank + Gemini)
    python -m phoenix_rag.cli chat                             # interactive Hebrew chat
    python -m phoenix_rag.cli eval                             # 20 grounded + 10 unanswerable
    python -m phoenix_rag.cli scenarios                        # Day-3 A/B/C
    python -m phoenix_rag.cli serve [--port 7860]              # local chat UI
    python -m phoenix_rag.cli status               # store counts and last runs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import Config, load_config


def _setup_logging(cfg: Config, name: str) -> Path:
    cfg.ensure_dirs()
    # Windows consoles default to a legacy code page; force UTF-8 so Hebrew titles are readable.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = cfg.logs_dir / f"{name}_{ts}.log"
    handlers: list[logging.Handler] = [logging.FileHandler(path, encoding="utf-8")]
    stream = logging.StreamHandler(sys.stdout)
    handlers.append(stream)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers)
    return path


# --------------------------------------------------------------------------- map
def cmd_map(cfg: Config, args: argparse.Namespace) -> int:
    """Fetch only the sitemap and show how seeds.yaml classifies it."""
    import httpx

    from .crawler.crawl import Crawler
    from .crawler.urls import classify, normalize_url

    _setup_logging(cfg, "map")
    client = httpx.Client(headers={"User-Agent": cfg.crawl["user_agent"]}, timeout=30, follow_redirects=True)
    resp = client.get(cfg.crawl["sitemap_url"])
    resp.raise_for_status()
    import xml.etree.ElementTree as ET

    root = ET.fromstring(resp.content)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locs = [u.text.strip() for u in root.findall("sm:url/sm:loc", ns) if u.text]
    included: dict[str, Counter] = defaultdict(Counter)
    excluded: Counter = Counter()
    seen: set[str] = set()
    for loc in locs:
        url = normalize_url(loc, cfg)
        if not url or url in seen:
            continue
        seen.add(url)
        allowed, stype, prio, reason = classify(url, cfg)
        if allowed:
            included[reason.split(":", 1)[1]][stype] += 1
        else:
            excluded[reason.split(":", 1)[1]] += 1
    total_in = sum(sum(c.values()) for c in included.values())
    print(f"\nsitemap URLs: {len(locs)}  unique: {len(seen)}  in scope: {total_in}  excluded: {sum(excluded.values())}\n")
    print("INCLUDED (prefix -> section_type counts)")
    for prefix, counter in sorted(included.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {sum(counter.values()):4d}  {prefix:45s} {dict(counter)}")
    print("\nEXCLUDED (prefix/reason -> count)")
    for reason, n in excluded.most_common():
        print(f"  {n:4d}  {reason}")
    if args.write:
        out = cfg.data_dir / "sitemap_classification.json"
        cfg.ensure_dirs()
        out.write_text(json.dumps({"included": included, "excluded": excluded}, ensure_ascii=False, indent=1, default=dict), encoding="utf-8")
        print(f"\nwritten {out}")
    return 0


# ------------------------------------------------------------------------- crawl
def cmd_crawl(cfg: Config, args: argparse.Namespace) -> int:
    from .crawler.crawl import Crawler

    log_path = _setup_logging(cfg, "crawl")
    crawler = Crawler(cfg, limit=args.limit, prefix=args.prefix, force=args.force,
                      discover=False if args.no_discover else None,
                      retry_failed=args.retry_failed, delay=args.delay)
    stats = crawler.run()
    print("\nCRAWL SUMMARY")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  log: {log_path}")
    return 0


# ------------------------------------------------------------------------- clean
def cmd_clean(cfg: Config, args: argparse.Namespace) -> int:
    from .cleaning.clean import clean_html
    from .crawler.crawl import read_raw
    from .crawler.store import PageStore

    _setup_logging(cfg, "clean")
    log = logging.getLogger("clean")
    store = PageStore(cfg.db_path)
    cfg.ensure_dirs()
    n = short = 0
    types: Counter = Counter()
    short_reasons: Counter = Counter()
    rows = [store.get(args.url)] if args.url else list(store.iter_pages("active"))
    for row in rows:
        if row is None or not row.raw_path:
            continue
        if args.limit and n >= args.limit:
            break
        html_text = read_raw(cfg.root / row.raw_path)
        page = clean_html(html_text, url=row.url, cfg=cfg, section_type=row.section_type or "general",
                          canonical_url=row.canonical_url, fetched_at=row.fetched_at, content_hash=row.content_hash)
        out = cfg.clean_dir / f"{row.url_id}.json"
        out.write_text(page.to_json(), encoding="utf-8")
        n += 1
        types[page.section_type] += 1
        if page.too_short:
            short += 1
            reason = ("iframe-only" if page.iframe_srcs else
                      "empty-page" if page.text_chars == 0 else "short-hub")
            short_reasons[reason] += 1
            log.info("too_short/%s (%d chars) %s", reason, page.text_chars, row.url)
        if args.verbose:
            print("=" * 100)
            print(page.display_url, "|", page.title, "|", page.section_type, "|", page.text_chars, "chars")
            print(" > ".join(page.breadcrumbs))
            print(page.plain_text()[: args.verbose])
            print("links:", [(l.kind, l.text, l.href) for l in page.links if l.kind in ("pdf", "form")][:8])
    stale = 0
    if not args.url and not args.limit:
        # Refresh semantics (spec 6.1): pages no longer active must not survive in the index. Drop their
        # cleaned JSON so the next `chunk` / `index --rechunk` forgets them.
        active_ids = {row.url_id for row in rows if row is not None}
        for path in cfg.clean_dir.glob("*.json"):
            if path.stem not in active_ids:
                path.unlink()
                stale += 1
        if stale:
            log.info("removed %d stale clean files (pages no longer active)", stale)
    print(f"\ncleaned {n} pages -> {cfg.clean_dir}  (too_short: {short} {dict(short_reasons)})  "
          f"stale removed: {stale}  section types: {dict(types)}")
    store.close()
    return 0


# ----------------------------------------------------------------------- inspect
def cmd_inspect(cfg: Config, args: argparse.Namespace) -> int:
    """Debug helper: show the <main> block structure of a stored page (what the cleaner sees)."""
    import re

    from bs4 import BeautifulSoup

    from .crawler.crawl import MAIN_RE, read_raw
    from .crawler.store import PageStore

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    store = PageStore(cfg.db_path)
    for url in args.urls:
        row = store.get(url)
        if row is None or not row.raw_path:
            print(f"not in store: {url}")
            continue
        html_text = read_raw(cfg.root / row.raw_path)
        m = MAIN_RE.search(html_text)
        soup = BeautifulSoup(m.group(0) if m else html_text, "lxml")
        print("=" * 100)
        print(url, "| title:", row.title, "| section_type:", row.section_type)
        for s in soup.select(".section-item"):
            txt = re.sub(r"\s+", " ", s.get_text(" ", strip=True))
            print(f"  -- {' '.join(s.get('class', []))[:70]:70s} | {len(txt):5d} chars | {txt[:200]}")
        print("  iframes:", [i.get("src", "")[:100] for i in soup.find_all("iframe")])
        print("  tables:", len(soup.find_all("table")),
              "| pdf links:", len([a for a in soup.find_all("a", href=True) if ".pdf" in a["href"].lower()]))
    store.close()
    return 0


# ------------------------------------------------------------------------- chunk
def cmd_chunk(cfg: Config, args: argparse.Namespace) -> int:
    from .indexing.chunk import chunk_page, load_clean_pages, write_chunks

    _setup_logging(cfg, "chunk")
    log = logging.getLogger("chunk")
    chunks = []
    pages = n_short = 0
    types: Counter = Counter()
    tok = []
    for page in load_clean_pages(cfg.clean_dir, skip_short=True):
        pages += 1
        if args.limit and pages > args.limit:
            pages -= 1
            break
        made = chunk_page(page, cfg)
        chunks.extend(made)
        types[page.section_type] += len(made)
        tok.extend(c.token_count for c in made)
        if args.verbose and made:
            print("=" * 80)
            print(page.display_url, page.title, f"-> {len(made)} chunks")
            print(made[0].embed_text[: args.verbose])
    write_chunks(chunks, cfg.chunks_path)
    tok.sort()
    p50 = tok[len(tok) // 2] if tok else 0
    log.info("pages=%d chunks=%d p50_tokens=%d max_tokens=%d by_type=%s", pages, len(chunks), p50, (tok[-1] if tok else 0), dict(types))
    print(f"\nchunked {pages} pages -> {len(chunks)} chunks  p50={p50} tokens  max={tok[-1] if tok else 0}")
    print(f"  wrote {cfg.chunks_path}")
    print(f"  by section_type: {dict(types)}")
    return 0


# ------------------------------------------------------------------------- index
def cmd_index(cfg: Config, args: argparse.Namespace) -> int:
    from .indexing.chunk import chunk_page, load_clean_pages, read_chunks, write_chunks
    from .indexing.embed import Embedder
    from .indexing.store import VectorIndex

    log_path = _setup_logging(cfg, "index")
    log = logging.getLogger("index")
    if args.rechunk or not cfg.chunks_path.exists():
        log.info("building chunks from %s", cfg.clean_dir)
        chunks = []
        for page in load_clean_pages(cfg.clean_dir, skip_short=True):
            chunks.extend(chunk_page(page, cfg))
        write_chunks(chunks, cfg.chunks_path)
        log.info("wrote %d chunks", len(chunks))
    else:
        chunks = list(read_chunks(cfg.chunks_path))
        log.info("loaded %d chunks from %s", len(chunks), cfg.chunks_path)
    if args.limit:
        chunks = chunks[: args.limit]
        log.info("limit: using first %d chunks", len(chunks))

    embedder = Embedder.from_config(cfg)
    vectors = _embed_with_cache(cfg, embedder, chunks, log)
    index = VectorIndex(cfg, embedder.model_name)
    index.replace_all(chunks, vectors)
    print(f"\nindexed {len(chunks)} chunks  model={embedder.model_name}  dim={embedder.dim}  chroma={index.count()}")
    print(f"  log: {log_path}")
    if not args.no_probe:
        _run_probe(cfg, embedder, index, k=int(cfg.retrieval.get("top_k", 12)), hybrid=True)
    return 0


def _embed_with_cache(cfg: Config, embedder, chunks: list, log: logging.Logger) -> list[list[float]]:
    """Embed only chunks missing from the on-disk cache, persisting after every group so a long,
    rate-limited run can be stopped and resumed."""
    from .indexing.embed_cache import EmbeddingCache

    cache_rel = cfg.embedding.get("cache_path") or "data/embeddings_cache.sqlite"
    cache = EmbeddingCache(cfg.root / cache_rel, embedder.model_name, embedder.dim)
    texts = [c.embed_text for c in chunks]
    have = cache.get_many(texts)
    todo = [i for i, t in enumerate(texts) if cache.key(t) not in have]
    log.info("embeddings: %d cached, %d to embed (cache %s)", len(texts) - len(todo), len(todo), cache_rel)

    group = 200  # ~4 API batches; each group is committed to the cache before moving on
    total_tokens = sum(int(chunks[i].token_count) for i in todo)
    done_tokens = 0
    started = time.monotonic()
    for g in range(0, len(todo), group):
        idx = todo[g : g + group]
        g_texts = [texts[i] for i in idx]
        g_counts = [int(chunks[i].token_count) for i in idx]
        vecs = embedder.encode_passages(g_texts, token_counts=g_counts)
        cache.put_many(g_texts, vecs)
        for t, v in zip(g_texts, vecs):
            have[cache.key(t)] = v
        done_tokens += sum(g_counts)
        elapsed = time.monotonic() - started
        rate = done_tokens / elapsed if elapsed > 0 else 0
        eta = (total_tokens - done_tokens) / rate / 60 if rate else 0
        log.info("progress %d/%d chunks  %d/%d tokens  %.0f tok/min  eta %.0f min",
                 min(g + group, len(todo)), len(todo), done_tokens, total_tokens, rate * 60, eta)
    cache.close()
    return [have[cache.key(t)] for t in texts]


# ------------------------------------------------------------------------- probe
def _run_probe(cfg: Config, embedder, index, *, k: int, hybrid: bool) -> int:
    import yaml

    from .config import CONFIG_DIR
    from .indexing.retrieve import search

    spec = yaml.safe_load((CONFIG_DIR / "probe.yaml").read_text(encoding="utf-8"))
    hits_n = 0
    print(f"\nPROBE  model={embedder.model_name}  k={k}  hybrid={hybrid}  vectors={index.count()}\n")
    for i, item in enumerate(spec["questions"], 1):
        q, expect = item["q"], item["expect"]
        results = search(index, embedder, q, top_k=k, hybrid=hybrid)
        urls = [h.url for h in results]
        ok = any(expect.rstrip("/") in u for u in urls)
        rank = next((j for j, u in enumerate(urls, 1) if expect.rstrip("/") in u), None)
        hits_n += int(ok)
        mark = "HIT" if ok else "MISS"
        print(f"{i:2d} {mark:4s} rank={str(rank or '-'):>3}  {q}")
        print(f"      expect: {expect}")
        if results:
            print(f"      top:    {results[0].url}  [{results[0].page_title} | {results[0].heading_path}]")
        if not ok:
            for h in results[:3]:
                print(f"      alt:    {h.url}")
    n = len(spec["questions"])
    print(f"\nhit rate @ {k}: {hits_n}/{n} = {hits_n / n:.0%}")
    return hits_n


def cmd_probe(cfg: Config, args: argparse.Namespace) -> int:
    """Run the Hebrew smoke questions in config/probe.yaml against the live index."""
    from .indexing.embed import Embedder
    from .indexing.store import VectorIndex

    _setup_logging(cfg, "probe")
    embedder = Embedder.from_config(cfg)
    index = VectorIndex(cfg, embedder.model_name)
    _run_probe(cfg, embedder, index, k=int(args.k), hybrid=args.hybrid)
    return 0


# ----------------------------------------------------------------- compare-embeddings
def cmd_compare(cfg: Config, args: argparse.Namespace) -> int:
    """Spec 6.4: compare embedding candidates on Hebrew retrieval hit rate (semantic only, subset)."""
    from .indexing.compare import run_comparison

    _setup_logging(cfg, "compare")
    run_comparison(cfg, args.models, distractors=args.distractors, k=args.k)
    return 0


# --------------------------------------------------------------------------- ask / chat
def cmd_ask(cfg: Config, args: argparse.Namespace) -> int:
    from .answering.pipeline import Pipeline, format_cli

    _setup_logging(cfg, "ask")
    pipe = Pipeline(cfg)
    question = " ".join(args.question).strip()
    ans = pipe.ask(question, rerank=not args.no_rerank)
    if args.json:
        print(json.dumps(ans.to_json(), ensure_ascii=False, indent=2))
    else:
        print("\n" + format_cli(ans, cfg))
    return 0


def cmd_chat(cfg: Config, args: argparse.Namespace) -> int:
    from .answering.pipeline import Pipeline, format_cli

    _setup_logging(cfg, "chat")
    pipe = Pipeline(cfg)
    pipe.ensure()
    print("צ'אט הפניקס (רק מתוכן האתר). יציאה: שורה ריקה / exit")
    history: list[dict[str, str]] = []
    while True:
        try:
            question = input("\nשאלה: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question or question.lower() in ("exit", "quit", "יציאה"):
            return 0
        ans = pipe.ask(question, history=history, rerank=not args.no_rerank)
        print(format_cli(ans, cfg))
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": ans.text})


def cmd_scenarios(cfg: Config, args: argparse.Namespace) -> int:
    from .answering.scenarios import run_scenarios

    _setup_logging(cfg, "scenarios")
    rows = run_scenarios(cfg, rerank=not args.no_rerank)
    return 0 if all(r.ok for r in rows) else 1


def cmd_eval(cfg: Config, args: argparse.Namespace) -> int:
    from .answering.eval_run import run_eval

    _setup_logging(cfg, "eval")
    rows = run_eval(cfg, limit=args.limit, rerank=not args.no_rerank)
    failed = sum(1 for r in rows if not r.ok)
    return 1 if failed else 0


def cmd_serve(cfg: Config, args: argparse.Namespace) -> int:
    from .answering.ui import serve

    _setup_logging(cfg, "serve")
    serve(cfg, host=args.host, port=args.port)
    return 0


# ------------------------------------------------------------------------ status
def cmd_status(cfg: Config, args: argparse.Namespace) -> int:
    from .crawler.store import PageStore

    if not cfg.db_path.exists():
        print("no store yet; run `crawl` first")
        return 0
    store = PageStore(cfg.db_path)
    print("pages by state:", store.counts())
    by_type = Counter(r.section_type for r in store.iter_pages("active"))
    print("active pages by section_type:", dict(by_type))
    n_clean = len(list(cfg.clean_dir.glob("*.json"))) if cfg.clean_dir.exists() else 0
    n_chunks = sum(1 for _ in cfg.chunks_path.open(encoding="utf-8")) if cfg.chunks_path.exists() else 0
    print(f"clean json: {n_clean}  chunks: {n_chunks}")
    if cfg.chroma_dir.exists():
        try:
            from .indexing.store import VectorIndex
            idx = VectorIndex(cfg, cfg.embedding["model"])
            print(f"chroma vectors: {idx.count()}  model={cfg.embedding['model']}")
        except Exception as exc:
            print(f"chroma: {exc}")
    print("\nlast runs:")
    for row in store.conn.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT 5"):
        print(f"  {row['run_id']} mode={row['mode']} started={row['started_at']} finished={row['finished_at']}")
        if row["stats"]:
            print(f"     {row['stats']}")
    store.close()
    return 0


# -------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phoenix_rag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("map", help="classify sitemap URLs according to seeds.yaml")
    p.add_argument("--write", action="store_true", help="write data/sitemap_classification.json")
    p.set_defaults(func=cmd_map)

    p = sub.add_parser("crawl", help="crawl in-scope pages into data/")
    p.add_argument("--limit", type=int, default=0, help="stop after N fetched pages (test runs)")
    p.add_argument("--prefix", default=None, help="only URLs under this path prefix, e.g. /claims-customer-service/")
    p.add_argument("--force", action="store_true", help="re-fetch everything, ignore lastmod/ETag")
    p.add_argument("--no-discover", action="store_true", help="do not follow in-content links")
    p.add_argument("--retry-failed", action="store_true", help="only re-try URLs whose state is 'failed'")
    p.add_argument("--delay", type=float, default=None, help="override crawl.delay_seconds for this run")
    p.set_defaults(func=cmd_crawl)

    p = sub.add_parser("clean", help="turn raw HTML into structured JSON")
    p.add_argument("--url", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--verbose", type=int, default=0, metavar="CHARS", help="print the first CHARS of each cleaned page")
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("chunk", help="split cleaned pages into context-prefixed chunks")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--verbose", type=int, default=0, metavar="CHARS")
    p.set_defaults(func=cmd_chunk)

    p = sub.add_parser("index", help="embed chunks and write the Chroma + BM25 index")
    p.add_argument("--rechunk", action="store_true", help="rebuild chunks.jsonl from cleaned pages")
    p.add_argument("--limit", type=int, default=0, help="index only the first N chunks (smoke test)")
    p.add_argument("--no-probe", action="store_true", help="skip the Hebrew retrieval smoke test")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("probe", help="run the Hebrew retrieval smoke questions")
    p.add_argument("-k", type=int, default=12)
    p.add_argument("--no-hybrid", dest="hybrid", action="store_false")
    p.set_defaults(func=cmd_probe, hybrid=True)

    p = sub.add_parser("compare-embeddings", help="bake-off of embedding models on a Hebrew subset (spec 6.4)")
    p.add_argument("--models", nargs="+", default=["st:models/multilingual-e5-base", "gemini-embedding-001"])
    p.add_argument("--distractors", type=int, default=400)
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("inspect", help="show the <main> block structure of stored pages (cleaner debugging)")
    p.add_argument("urls", nargs="+")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("ask", help="answer one Hebrew question from the site index")
    p.add_argument("question", nargs="+")
    p.add_argument("--json", action="store_true", help="print spec JSON (answer_type, sources, used_chunk_ids)")
    p.add_argument("--no-rerank", action="store_true", help="skip Cohere Rerank")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="interactive Hebrew chat over the site index")
    p.add_argument("--no-rerank", action="store_true")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("scenarios", help="run Day-3 reference scenarios A/B/C")
    p.add_argument("--no-rerank", action="store_true")
    p.set_defaults(func=cmd_scenarios)

    p = sub.add_parser("eval", help="run config/eval.yaml (grounded + unanswerable)")
    p.add_argument("--limit", type=int, default=0, help="only the first N items (smoke)")
    p.add_argument("--no-rerank", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="local Hebrew chat UI in the browser")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("status", help="show store counts and recent runs")
    p.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
