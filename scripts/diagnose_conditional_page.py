# Step-0 diagnostics: hybrid retrieval + pipeline chunk selection for water-heater cases.
"""Usage: python scripts/diagnose_conditional_page.py [--no-rerank]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phoenix_rag.answering.context import retrieval_query
from phoenix_rag.answering.generate import rewrite_question
from phoenix_rag.answering.pipeline import Pipeline, _LLM_CHUNK_CAP
from phoenix_rag.config import load_config
from phoenix_rag.indexing.retrieve import search

WH = "/claims-customer-service/claims-appartment-insurance/water-heater/"
SOLAR_HISTORY = [
    {"role": "user", "content": "נשבר לי דוד השמש, מה עושים?"},
    {
        "role": "assistant",
        "content": (
            "במידה והנזק לדוד השמש אינו כתוצאה מנזקי מזג אויר ונרכשה הרחבה לדוודים, "
            "יש לפנות לנותן השירות."
        ),
    },
]
QUOTE_FU = (
    "בהתייחס לציטוט הבא מהתשובה: «במידה והנזק לדוד השמש אינו כתוצאה מנזקי מזג אויר "
    "ונרכשה הרחבה לדוודים, יש לפנות לנותן השירות.»\n\n"
    "הנזק הוא ממזג אויר"
)
CASES = [
    ("bug1_weather_quote", QUOTE_FU, SOLAR_HISTORY),
    ("bug2_solar_unspecified", "נשבר לי דוד השמש, מה עושים?", None),
]


def _preview(text: str, n: int = 150) -> str:
    t = (text or "").replace("\n", " ").strip()
    return t[:n] + ("…" if len(t) > n else "")


def _print_bm25_top(index, query: str, k: int = 25) -> None:
    pairs = index.lexical_query(query, k)
    print(f"  BM25 top-{k} (score = BM25Okapi):")
    branch_a = False
    for rank, (cid, score) in enumerate(pairs, 1):
        stored = index.chunk_by_id(cid)
        if not stored:
            continue
        url = stored.display_url or stored.url or ""
        if WH in url and any(
            x in (stored.original_text or "") for x in ("מזג", "מבנה", "תמונ", "טופס")
        ):
            branch_a = branch_a or ("מזג" in (stored.original_text or "") and "אינו" not in (stored.heading_path or ""))
        mark = " [WH]" if WH in url else ""
        print(
            f"    {rank:2d}  score={score:7.3f}  id={cid[:48]}…{mark}\n"
            f"        path={stored.heading_path!r}\n"
            f"        {_preview(stored.original_text)}"
        )
    wh_any = any(WH in (index.chunk_by_id(cid).display_url or index.chunk_by_id(cid).url or "") for cid, _ in pairs)
    print(f"  water-heater URL in BM25 top-{k}: {wh_any}")
    print(f"  branch-A-ish chunk in BM25 top-{k}: {branch_a}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipe = Pipeline(cfg)
    embedder, index = pipe.ensure()
    rcfg = cfg.retrieval
    max_cand = int((rcfg.get("rerank") or {}).get("max_candidates") or 40)
    keep = int((rcfg.get("rerank") or {}).get("top_n") or 5)

    for label, question, history in CASES:
        print("=" * 72)
        print(label)
        print(f"question: {question[:200]}{'…' if len(question) > 200 else ''}")
        rewritten = question
        if history and rcfg.get("rewrite_with_history", True):
            rewritten = rewrite_question(question, history, cfg) or question
        search_q = retrieval_query(rewritten, history)
        print(f"rewritten: {rewritten}")
        print(f"retrieval_query: {search_q!r}")

        _print_bm25_top(index, search_q)

        prep = pipe._prepare(
            question,
            history=history,
            rerank=not args.no_rerank,
            rewrite=True,
        )
        if prep.early:
            print(f"early exit: {prep.early.status}")
            continue
        print(f"post-rerank/filter top-{keep} (RRF score on Hit.score):")
        branch_a_retrieved = False
        for i, h in enumerate(prep.hits[:max_cand], 1):
            url = h.display_url or h.url or ""
            if WH in url and "מזג" in (h.text or "") and "5420" not in (h.text or ""):
                branch_a_retrieved = True
            print(
                f"  {i:2d}  rrf={h.score:.4f}  rerank={h.rerank_score:.4f}  id={h.chunk_id[:48]}…\n"
                f"      path={h.heading_path!r}\n"
                f"      {_preview(h.text)}"
            )
        llm = prep.hits[:_LLM_CHUNK_CAP]
        print(f"chunks sent to LLM ({len(llm)}):")
        for h in llm:
            print(f"  - {h.chunk_id[:48]}… | {h.heading_path!r} | {_preview(h.text, 100)}")
        print(f"branch-A (weather claim path) in retrieved top-{max_cand}: {branch_a_retrieved}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
