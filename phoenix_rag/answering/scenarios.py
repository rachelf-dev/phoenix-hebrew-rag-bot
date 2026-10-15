"""Day-3 reference scenarios A/B/C: claim how-to, missing SLA follow-up, personal-policy boundary."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from .pipeline import Pipeline

A_Q = "יש לי נזילה בבית, איך מגישים תביעה?"
B_Q = "תוך כמה זמן מקבלים תשובה?"
C_Q = "האם הפוליסה שלי מכסה את זה?"

A_URLS = (
    "/claims-customer-service/claims-appartment-insurance/pipeline-damage/",
    "/claims-customer-service/claims-appartment-insurance/damage-to-the-apartment/",
)


@dataclass
class ScenarioResult:
    sid: str
    question: str
    ok: bool
    status: str
    answer_type: str
    citations: list[str]
    detail: str


def _cite_ok(citations: list[str]) -> bool:
    return any(n.rstrip("/") in (u or "") for n in A_URLS for u in citations)


def run_scenarios(cfg: Config, *, rerank: bool = True) -> list[ScenarioResult]:
    """Run A, then B and C as follow-ups on the same conversation. Prints PASS/FAIL."""
    pipe = Pipeline(cfg)
    pipe.ensure()
    history: list[dict[str, str]] = []
    out: list[ScenarioResult] = []

    # A — grounded how-to
    a = pipe.ask(A_Q, history=history, rerank=rerank)
    a_ok = a.status == "answered" and _cite_ok(a.citations)
    a_detail = "cited " + (a.citations[0] if a.citations else "NONE")
    out.append(ScenarioResult("A", A_Q, a_ok, a.status, a.answer_type, list(a.citations), a_detail))
    history.append({"role": "user", "content": A_Q})
    history.append({"role": "assistant", "content": a.text})

    # B — follow-up: claim decision time is not on the page
    b = pipe.ask(B_Q, history=history, rerank=rerank)
    b_ok = b.status == "not_found" or b.answer_type == "not_found"
    out.append(ScenarioResult("B", B_Q, b_ok, b.status, b.answer_type, list(b.citations),
                              f"rewritten={b.rewritten!r}"))
    history.append({"role": "user", "content": B_Q})
    history.append({"role": "assistant", "content": b.text})

    # C — personal policy boundary
    c = pipe.ask(C_Q, history=history, rerank=rerank)
    c_ok = c.status == "personal_boundary" or c.answer_type == "personal_boundary"
    out.append(ScenarioResult("C", C_Q, c_ok, c.status, c.answer_type, list(c.citations),
                              f"boundary={c.boundary}"))

    for r in out:
        mark = "PASS" if r.ok else "FAIL"
        print(f"{mark:4s} {r.sid}  status={r.status:20s} type={r.answer_type:20s}  {r.question}")
        if not r.ok:
            print(f"      {r.detail}")
            if r.citations:
                print(f"      cites: {r.citations[:3]}")
    print(f"\nscenarios A/B/C: {sum(r.ok for r in out)}/{len(out)}")
    return out
