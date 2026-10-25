"""Rule-based Hebrew intent gates (personal policy, account, advice, off-topic).

Follow-ups in an on-topic thread skip out_of_scope so RAG can run. Fixed refusal
strings live in config.yaml → messages and run before generation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .context import extract_thread_slots, is_followup, thread_has_insurance_slots, thread_has_topics
from .sanitize import is_unintelligible_input

# Personal-policy: we still retrieve general coverage, then append the boundary message.
_PERSONAL = (
    "הפוליסה שלי",
    "הפוליסה שלנו",
    "הכיסוי שלי",
    "הכיסוי שלנו",
    "האם אני מכוסה",
    "האם אני מכוסהת",
    "האם אני כלול",
    "מה מגיע לי בפוליסה",
    "לפי הפוליסה שלי",
    "בפוליסה שלי",
)

# Account / claim-status: no retrieval of personal records — we have none.
_ACCOUNT = (
    "התביעה שלי",
    "התביעה שלנו",
    "סטטוס התביעה",
    "מצב התביעה",
    "מה קורה עם התביעה",
    "מספר תביעה",
    "מספר הפוליסה שלי",
    "מספר פוליסה שלי",
    "היתרה שלי",
    "כמה כסף יש לי",
    "החשבון שלי",
    "באזור האישי כמה",
    "מה יתרת",
    "כמה צברתי",
    "מתי פג תוקף הפוליסה שלי",
    "תעודת הזהות שלי",
)

_ADVICE = (
    "כדאי לי",
    "מומלץ לי",
    "תבחר לי",
    "תמליץ לי",
    "מה לקנות",
    "איזה ביטוח הכי טוב בשבילי",
    "איזה ביטוח מתאים לי",
    "במה לבחור",
)

# Cheap on-topic lexicon so weather / sports / shopping skip the index.
_ON_TOPIC = (
    "ביטוח",
    "תביע",
    "פוליס",
    "הפניקס",
    "fnx",
    "כיסוי",
    "פרמי",
    "גמל",
    "פנס",
    "השתלמות",
    "משכנתא",
    "תכולה",
    "מקיף",
    "חובה",
    "נסיעות",
    "תאונ",
    "נזק",
    "נזיל",
    "צנרת",
    "דוד",
    "שרברב",
    "שרברבים",
    "הרחבה",
    "הרחבות",
    "מוסך",
    "שמאי",
    "פיצוי",
    "מבוטח",
    "מוטב",
    "עמית",
    "חיסכון",
    "שירות לקוחות",
    "סיעוד",
    "מחלות קשות",
    "ריסק",
    "נהג",
    "רכב",
    "גרר",
    "גריר",
    "שירותי דרך",
    "דירה",
    "תביעות",
    "קרן",
    "הלווא",
    "פרטים אישיים",
    "עדכון פרטים",
    "מעדכנים",
    "pdf",
    "PDF",
    "הצהר",
    "פדיון",
    "בריאות",
    # English insurance queries (spec: answer in Hebrew, still on-topic)
    "insurance",
    "travel",
    "claim",
    "policy",
    "phoenix",
    "extend",
    "abroad",
    "coverage",
    "car",
    "health",
)


@dataclass(frozen=True)
class Boundary:
    """Which canned message to use; empty kind means 'normal RAG path'."""

    kind: str = ""  # personal_boundary | no_advice | no_account_access | out_of_scope | ""
    message_key: str = ""


_STANDALONE_OFF_TOPIC = (
    "מזג האוויר",
    "מזג אוויר",
    "כדורגל",
    "ספורט",
    "מתכון",
    "בחירות",
)


def _followup_needs_context(question: str, history: list[dict[str, str]] | None = None) -> bool:
    q = (question or "").strip()
    if not q or any(p in q for p in _STANDALONE_OFF_TOPIC):
        return False
    return is_followup(q, history)


def is_on_topic(text: str) -> bool:
    """True when the text matches the cheap insurance / Phoenix site lexicon."""
    q = (text or "").strip()
    if not q:
        return False
    return any(p in q for p in _ON_TOPIC)


def detect(question: str) -> Boundary:
    q = (question or "").strip()
    if not q:
        return Boundary("out_of_scope", "out_of_scope")
    low = q  # Hebrew has no case; keep original for substring match
    if any(p in low for p in _ACCOUNT):
        return Boundary("no_account_access", "no_account_access")
    if any(p in low for p in _PERSONAL):
        return Boundary("personal_boundary", "personal_boundary")
    if any(p in low for p in _ADVICE):
        return Boundary("no_advice", "no_advice")
    if not is_on_topic(low):
        return Boundary("out_of_scope", "out_of_scope")
    return Boundary()


def _hard_boundary(bound: Boundary) -> bool:
    return bound.kind in ("no_account_access", "personal_boundary", "no_advice")


def detect_with_history(
    original: str,
    rewritten: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> Boundary:
    """Apply detect; follow-ups in an on-topic thread go to RAG instead of out_of_scope."""
    original = (original or "").strip()
    rewritten = (rewritten or original).strip()
    if is_unintelligible_input(original):
        return Boundary("out_of_scope", "out_of_scope")
    # Keep personal/account/advice from the user's actual wording first.
    orig_bound = detect(original)
    if _hard_boundary(orig_bound):
        return orig_bound
    rew_bound = detect(rewritten) if rewritten != original else orig_bound
    if _hard_boundary(rew_bound):
        return rew_bound
    if orig_bound.kind != "out_of_scope" or rew_bound.kind != "out_of_scope":
        return Boundary()

    history = history or []
    if not history:
        return orig_bound
    if any(p in original for p in _STANDALONE_OFF_TOPIC):
        return orig_bound

    if thread_has_insurance_slots(history, original) and _followup_needs_context(original, history):
        return Boundary()

    ctx = " ".join(
        (t.get("content") or "").strip()
        for t in history[-6:]
        if (t.get("content") or "").strip()
    )
    if ctx:
        widened = detect(f"{ctx} {rewritten}")
        # History may mention "הפוליסה שלי"; do not inherit that onto a later follow-up.
        if widened.kind != "out_of_scope" and not _hard_boundary(widened):
            return Boundary()

    for turn in reversed(history):
        text = (turn.get("content") or "").strip()
        if not text:
            continue
        prior = detect(text)
        if _hard_boundary(prior):
            continue
        if prior.kind != "out_of_scope" or is_on_topic(text):
            if _followup_needs_context(original, history):
                return Boundary()
    if extract_thread_slots(history, rewritten).product or extract_thread_slots(history, rewritten).damages:
        if _followup_needs_context(original, history):
            return Boundary()
    return orig_bound
