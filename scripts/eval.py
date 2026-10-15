#!/usr/bin/env python3
"""Run the Day-4 evaluation set (eval/questions.jsonl) and write metrics to eval/results/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phoenix_rag.answering.eval_set import (  # noqa: E402
    load_questions,
    run_eval_set,
    write_results,
)
from phoenix_rag.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Phoenix RAG evaluation (25+5 questions, full metrics)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--questions", type=Path, default=None, help="default: eval/questions.jsonl")
    parser.add_argument("--tag", default="run", help="label for results (e.g. before, after)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-hidden", action="store_true", help="skip hidden holdout questions")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--calibrate-only", action="store_true", help="retrieval scores only, no LLM calls")
    args = parser.parse_args()

    cfg = load_config(args.config)
    qpath = args.questions or (cfg.root / "eval" / "questions.jsonl")
    if not qpath.is_absolute():
        qpath = cfg.root / qpath
    questions = load_questions(qpath, include_hidden=not args.no_hidden)
    if args.limit:
        questions = questions[: args.limit]

    out_dir = cfg.root / "eval" / "results"
    pause = float(cfg.evaluation.get("pause_seconds") or 8)

    if args.calibrate_only:
        from phoenix_rag.answering.eval_set import _retrieval_hit, calibrate_rrf_threshold, score_question
        from phoenix_rag.answering.pipeline import Answer, Pipeline

        pipe = Pipeline(cfg)
        embedder, index = pipe.ensure()
        results = []
        for q in questions:
            retr_ok, top_rrf = _retrieval_hit(cfg, embedder, index, q.question, q.expected_url_contains, history=q.history)
            ans = Answer(status="skipped", text="", question=q.question)
            row = score_question(q, ans, retrieval_hit=retr_ok, top_rrf=top_rrf)
            row.detail = "calibrate-only"
            if q.expected_answer_type == "answered":
                row.ok = retr_ok
            else:
                row.ok = True
            results.append(row)
        from phoenix_rag.answering.eval_set import summarize

        summary = summarize(args.tag, results, questions)
        json_path, md_path = write_results(out_dir, args.tag, results, summary)
        print(f"retrieval-only; suggested RRF threshold: {summary.suggested_rrf_threshold}")
        print(f"  {summary.threshold_note}")
        print(f"wrote {json_path}\n      {md_path}")
        return 0

    results, summary = run_eval_set(cfg, questions, tag=args.tag, rerank=not args.no_rerank, pause_seconds=pause)
    for r in results:
        mark = "PASS" if r.ok else "FAIL"
        hid = " [hidden]" if r.hidden else ""
        print(f"{mark:4s} {r.id:4s} {r.category:18s} {r.status:20s}{hid}  {r.question[:70]}")
        if not r.ok or r.possible_hallucination:
            print(f"      {r.detail}")

    print(
        f"\n{summary.passed}/{summary.total} passed | "
        f"retrieval {summary.retrieval_hit_rate} | "
        f"citations {summary.citation_correctness} | "
        f"refusal {summary.correct_refusal_rate} | "
        f"false refusal {summary.false_refusal_rate} | "
        f"hallucination {summary.hallucination_rate}"
    )
    print(f"suggested relevance_threshold (RRF): {summary.suggested_rrf_threshold} — {summary.threshold_note}")

    json_path, md_path = write_results(out_dir, args.tag, results, summary)
    print(f"\nresults: {json_path}\n         {md_path}")
    return 0 if summary.passed == summary.total and summary.hallucination_rate == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
