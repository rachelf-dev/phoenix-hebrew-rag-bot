"""Load eval/questions.jsonl, score answers, aggregate metrics, and suggest RRF threshold."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..indexing.embed import Embedder
from ..indexing.retrieve import search
from ..indexing.store import VectorIndex
from .context import retrieval_query
from .generate import rewrite_question
from .pipeline import Answer, Pipeline

REFUSAL_STATUSES = frozenset({"not_found", "out_of_scope", "no_account_access"})
BOUNDARY_STATUSES = frozenset({"personal_boundary", "no_advice"})


@dataclass
class EvalQuestion:
    id: str
    category: str
    question: str
    expected_answer_type: str
    expected_url_contains: list[str] = field(default_factory=list)
    forbidden_url_contains: list[str] = field(default_factory=list)
    required_facts: list[str] = field(default_factory=list)
    history: list[dict[str, str]] = field(default_factory=list)
    hidden: bool = False
    notes: str = ""


@dataclass
class QuestionResult:
    id: str
    category: str
    hidden: bool
    question: str
    ok: bool
    status: str
    answer_type: str
    citations: list[str]
    retrieval_hit: bool
    citation_ok: bool
    possible_hallucination: bool
    latency_ms: float
    detail: str
    top_rrf_score: float = 0.0


@dataclass
class EvalSummary:
    tag: str
    total: int
    passed: int
    retrieval_hit_rate: float
    citation_correctness: float
    correct_refusal_rate: float
    false_refusal_rate: float
    hallucination_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    suggested_rrf_threshold: float
    threshold_note: str
    by_category: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_questions(path: Path, *, include_hidden: bool = True) -> list[EvalQuestion]:
    out: list[EvalQuestion] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        raw = json.loads(line)
        if raw.get("hidden") and not include_hidden:
            continue
        urls = raw.get("expected_url_contains") or raw.get("expect") or []
        if isinstance(urls, str):
            urls = [urls]
        forbidden = raw.get("forbidden_url_contains") or []
        if isinstance(forbidden, str):
            forbidden = [forbidden]
        out.append(
            EvalQuestion(
                id=str(raw.get("id") or f"q{len(out)+1:02d}"),
                category=str(raw.get("category") or "direct"),
                question=str(raw["question"]),
                expected_answer_type=str(raw.get("expected_answer_type") or "answered"),
                expected_url_contains=[u.rstrip("/") for u in urls if u],
                forbidden_url_contains=[u.rstrip("/") for u in forbidden if u],
                required_facts=list(raw.get("required_facts") or []),
                history=list(raw.get("history") or []),
                hidden=bool(raw.get("hidden")),
                notes=str(raw.get("notes") or ""),
            )
        )
    return out


def _url_hit(needles: list[str], urls: list[str], *, require_all: bool = False) -> bool:
    if not needles:
        return bool(urls)
    if require_all:
        return all(any(n in (u or "") for u in urls) for n in needles)
    return any(n in (u or "") for n in needles for u in urls)


def _url_forbidden(needles: list[str], urls: list[str]) -> bool:
    if not needles:
        return False
    return any(n in (u or "") for n in needles for u in urls)


def _retrieval_hit(
    cfg: Config,
    embedder: Embedder,
    index: VectorIndex,
    question: str,
    needles: list[str],
    *,
    history: list[dict[str, str]] | None = None,
) -> tuple[bool, float]:
    rcfg = cfg.retrieval
    rewritten = question
    if history and rcfg.get("rewrite_with_history", True):
        rewritten = rewrite_question(question, history, cfg) or question
    search_q = retrieval_query(rewritten, history)
    top_k = int(rcfg.get("top_k", 12))
    max_cand = int((rcfg.get("rerank") or {}).get("max_candidates") or top_k)
    hits = search(
        index,
        embedder,
        search_q,
        top_k=max_cand,
        hybrid=bool(rcfg.get("hybrid", True)),
        threshold=0.0,
        max_per_url=int(rcfg.get("max_per_url") or 2),
    )
    top = float(hits[0].score) if hits else 0.0
    urls = [(h.display_url or h.url or "") for h in hits]
    return _url_hit(needles, urls), top


def _possible_hallucination(q: EvalQuestion, ans: Answer) -> bool:
    if q.expected_answer_type in ("not_found", "out_of_scope", "no_account_access"):
        if ans.status == "answered" and ans.citations:
            return True
        if ans.status == "answered" and (ans.text or "").strip():
            return True
    if ans.status == "answered" and not ans.citations:
        return True
    if ans.status == "answered" and ans.citations:
        retrieved = {(h.display_url or h.url or "").rstrip("/") for h in ans.hits}
        for cite in ans.citations:
            c = (cite or "").rstrip("/")
            if not any(c in r or r in c for r in retrieved if r):
                return True
    return False


def score_question(q: EvalQuestion, ans: Answer, *, retrieval_hit: bool, top_rrf: float) -> QuestionResult:
    exp = q.expected_answer_type
    cites = list(ans.citations)
    citation_ok = True
    if q.expected_url_contains:
        citation_ok = _url_hit(q.expected_url_contains, cites)
    if citation_ok and q.forbidden_url_contains:
        citation_ok = not _url_forbidden(q.forbidden_url_contains, cites)
    halluc = _possible_hallucination(q, ans)

    if exp == "answered":
        ok = ans.status in ("answered", *BOUNDARY_STATUSES) and citation_ok and not halluc
        detail = "cited " + (cites[0] if cites else "NONE")
    elif exp in REFUSAL_STATUSES:
        ok = ans.status == exp or (exp == "not_found" and ans.status in REFUSAL_STATUSES)
        ok = ok and not halluc
        detail = f"status={ans.status}"
    elif exp in BOUNDARY_STATUSES:
        ok = ans.status in BOUNDARY_STATUSES or ans.answer_type == "personal_boundary"
        ok = ok and not halluc
        if q.expected_url_contains:
            ok = ok and citation_ok
        detail = f"status={ans.status} boundary"
    else:
        ok = ans.answer_type == exp and not halluc
        detail = f"type={ans.answer_type}"

    return QuestionResult(
        id=q.id,
        category=q.category,
        hidden=q.hidden,
        question=q.question,
        ok=ok,
        status=ans.status,
        answer_type=ans.answer_type,
        citations=cites,
        retrieval_hit=retrieval_hit,
        citation_ok=citation_ok,
        possible_hallucination=halluc,
        latency_ms=0.0,
        detail=detail,
        top_rrf_score=top_rrf,
    )


def calibrate_rrf_threshold(results: list[QuestionResult], questions: list[EvalQuestion]) -> tuple[float, str]:
    """Pick an RRF floor from top-1 hybrid scores: keep all answerable, drop weak unanswerable when possible."""
    by_id = {q.id: q for q in questions}
    answerable: list[float] = []
    unanswerable: list[float] = []
    for r in results:
        q = by_id.get(r.id)
        if not q:
            continue
        if q.expected_answer_type == "answered" or q.expected_url_contains:
            answerable.append(r.top_rrf_score)
        elif q.expected_answer_type in ("not_found", "out_of_scope", "no_account_access"):
            unanswerable.append(r.top_rrf_score)

    if not answerable:
        return 0.0, "no answerable items for calibration; keeping threshold 0.0"

    floor = min(answerable) * 0.85
    if not unanswerable:
        return round(floor, 6), f"floor={floor:.6f} from min answerable RRF (no unanswerable in run)"

    # Raise threshold only if it still keeps every answerable top score.
    candidates = sorted(set(unanswerable))
    best = 0.0
    for t in candidates:
        if t >= min(answerable):
            break
        if all(s >= t for s in answerable):
            best = max(best, t)
    if best > 0:
        return round(best, 6), f"RRF threshold {best:.6f} separates {sum(1 for s in unanswerable if s < best)}/{len(unanswerable)} weak unanswerable hits"
    return round(floor, 6), f"conservative floor {floor:.6f} (unanswerable scores overlap answerable)"


def summarize(tag: str, results: list[QuestionResult], questions: list[EvalQuestion]) -> EvalSummary:
    by_id = {q.id: q for q in questions}
    answerable = [r for r in results if by_id[r.id].expected_answer_type == "answered"]
    refuse_expected = [
        r
        for r in results
        if by_id[r.id].expected_answer_type in ("not_found", "out_of_scope", "no_account_access", *BOUNDARY_STATUSES)
    ]
    should_refuse = [r for r in results if by_id[r.id].expected_answer_type in ("not_found", "out_of_scope", "no_account_access")]
    latencies = sorted(r.latency_ms for r in results if r.latency_ms > 0)

    def rate(num: int, den: int) -> float:
        return round(num / den, 4) if den else 0.0

    thr, thr_note = calibrate_rrf_threshold(results, questions)
    by_cat: dict[str, dict[str, Any]] = {}
    for r in results:
        bucket = by_cat.setdefault(r.category, {"total": 0, "passed": 0})
        bucket["total"] += 1
        if r.ok:
            bucket["passed"] += 1

    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = latencies[int(len(latencies) * 0.95)] if len(latencies) >= 2 else (latencies[0] if latencies else 0.0)

    return EvalSummary(
        tag=tag,
        total=len(results),
        passed=sum(1 for r in results if r.ok),
        retrieval_hit_rate=rate(sum(1 for r in answerable if r.retrieval_hit), len(answerable)),
        citation_correctness=rate(sum(1 for r in answerable if r.citation_ok), len(answerable)),
        correct_refusal_rate=rate(sum(1 for r in should_refuse if r.ok), len(should_refuse)),
        false_refusal_rate=rate(
            sum(1 for r in answerable if r.status == "not_found" and by_id[r.id].expected_answer_type == "answered"),
            len(answerable),
        ),
        hallucination_rate=rate(sum(1 for r in results if r.possible_hallucination), len(results)),
        latency_p50_ms=round(p50, 1),
        latency_p95_ms=round(p95, 1),
        suggested_rrf_threshold=thr,
        threshold_note=thr_note,
        by_category=by_cat,
    )


def run_eval_set(
    cfg: Config,
    questions: list[EvalQuestion],
    *,
    tag: str = "run",
    rerank: bool = True,
    pause_seconds: float = 0.0,
) -> tuple[list[QuestionResult], EvalSummary]:
    pipe = Pipeline(cfg)
    embedder, index = pipe.ensure()
    results: list[QuestionResult] = []
    for i, q in enumerate(questions):
        if i and pause_seconds > 0:
            time.sleep(pause_seconds)
        t0 = time.perf_counter()
        retr_ok, top_rrf = _retrieval_hit(cfg, embedder, index, q.question, q.expected_url_contains, history=q.history)
        try:
            ans = pipe.ask(q.question, history=q.history or None, rerank=rerank)
        except Exception as exc:
            ans = Answer(status="error", text=str(exc)[:200], question=q.question)
        elapsed = (time.perf_counter() - t0) * 1000.0
        row = score_question(q, ans, retrieval_hit=retr_ok, top_rrf=top_rrf)
        row.latency_ms = round(elapsed, 1)
        results.append(row)
    return results, summarize(tag, results, questions)


def write_results(out_dir: Path, tag: str, results: list[QuestionResult], summary: EvalSummary) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    json_path = out_dir / f"{stamp}_{tag}.json"
    md_path = out_dir / f"{stamp}_{tag}.md"
    payload = {
        "summary": asdict(summary),
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Eval run `{tag}` ({stamp})",
        "",
        f"| metric | value |",
        f"| --- | --- |",
        f"| passed | {summary.passed}/{summary.total} |",
        f"| retrieval hit rate | {summary.retrieval_hit_rate} |",
        f"| citation correctness | {summary.citation_correctness} |",
        f"| correct refusal rate | {summary.correct_refusal_rate} |",
        f"| false refusal rate | {summary.false_refusal_rate} |",
        f"| hallucination rate (heuristic) | {summary.hallucination_rate} |",
        f"| latency p50 (ms) | {summary.latency_p50_ms} |",
        f"| latency p95 (ms) | {summary.latency_p95_ms} |",
        f"| suggested RRF threshold | {summary.suggested_rrf_threshold} |",
        "",
        summary.threshold_note,
        "",
        "## By category",
        "",
    ]
    for cat, stats in sorted(summary.by_category.items()):
        lines.append(f"- **{cat}**: {stats['passed']}/{stats['total']}")
    lines.extend(["", "## Failures", ""])
    for r in results:
        if not r.ok or r.possible_hallucination:
            lines.append(f"- `{r.id}` ({r.category}): {r.detail} — {r.question[:80]}")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
