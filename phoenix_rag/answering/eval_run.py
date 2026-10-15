"""Run config/eval.yaml: 20 grounded (must cite a real URL) + 10 unanswerable (must not invent)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import yaml

from ..config import Config
from .pipeline import Pipeline

log = logging.getLogger(__name__)


@dataclass
class ItemResult:
    kind: str
    question: str
    ok: bool
    status: str
    citations: list[str]
    detail: str


def load_eval(cfg: Config) -> tuple[list[dict], list[dict]]:
    rel = cfg.evaluation.get("path") or "config/eval.yaml"
    path = cfg.root / rel
    if path.suffix == ".jsonl":
        from .eval_set import load_questions

        items = load_questions(path, include_hidden=True)
        grounded, unans = [], []
        for q in items:
            row = {
                "q": q.question,
                "expect": q.expected_url_contains,
                "history": q.history,
                "expected_type": q.expected_answer_type,
            }
            if q.expected_answer_type == "answered":
                grounded.append(row)
            else:
                unans.append(row)
        return grounded, unans
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(spec.get("grounded") or []), list(spec.get("unanswerable") or [])


def _cite_hit(expect: str | list[str], citations: list[str]) -> bool:
    needles = expect if isinstance(expect, list) else [expect]
    needles = [n.rstrip("/") for n in needles if n]
    if not needles:
        return bool(citations)
    return any(n in (u or "") for n in needles for u in citations)


def run_eval(cfg: Config, *, limit: int = 0, rerank: bool = True) -> list[ItemResult]:
    grounded, unans = load_eval(cfg)
    items: list[tuple[str, dict]] = [("grounded", x) for x in grounded] + [("unanswerable", x) for x in unans]
    if limit:
        items = items[:limit]
    pipe = Pipeline(cfg)
    pipe.ensure()
    pause = float(cfg.evaluation.get("pause_seconds") or 8)
    out: list[ItemResult] = []
    for i, (kind, item) in enumerate(items):
        q = item["q"]
        if i:
            time.sleep(pause)
        try:
            hist = item.get("history") or None
            ans = pipe.ask(q, history=hist, rerank=rerank, rewrite=bool(hist))
            status, citations = ans.status, list(ans.citations)
            exp = item.get("expected_type")
            if kind == "grounded":
                ok = status in ("answered", "personal_boundary", "no_advice") and _cite_hit(item.get("expect", ""), citations)
                detail = "cited " + (citations[0] if citations else "NONE")
            elif exp in ("personal_boundary", "no_advice", "no_account_access", "out_of_scope", "not_found"):
                ok = status == exp or (exp == "not_found" and status in ("not_found", "out_of_scope", "no_account_access"))
                if exp == "personal_boundary" and item.get("expect"):
                    ok = ok and _cite_hit(item.get("expect", ""), citations)
                detail = f"expected={exp} status={status}"
            else:
                ok = status in ("not_found", "out_of_scope", "no_account_access")
                detail = f"status={status}"
        except Exception as exc:
            ok, status, citations, detail = False, "error", [], str(exc)[:240]
            log.exception("eval item failed: %s", q)
        mark = "PASS" if ok else "FAIL"
        log.info("%s %s %s | %s", mark, kind, q, detail)
        out.append(ItemResult(kind=kind, question=q, ok=ok, status=status,
                              citations=citations, detail=detail))
        print(f"{mark:4s} {kind:13s} {status:20s}  {q}")
        if not ok:
            print(f"      {detail}")
            if citations:
                print(f"      cites: {citations[:3]}")
    g = [r for r in out if r.kind == "grounded"]
    u = [r for r in out if r.kind == "unanswerable"]
    print(f"\ngrounded: {sum(r.ok for r in g)}/{len(g)}   unanswerable (no invent): {sum(r.ok for r in u)}/{len(u)}")
    return out
