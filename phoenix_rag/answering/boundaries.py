"""Rule-based Hebrew intent gates (personal policy, account, advice, off-topic).

Fixed refusal strings live in config.yaml → messages. These run before generation so
the model is not asked to invent a personal answer.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    "דירה",
    "תביעות",
    "קרן",
    "הלווא",
    "פרטים אישיים",
    "עדכון פרטים",
    "מעדכנים",
    "טופס",
    "טפסים",
    "הורד",
    "הורדה",
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

_FOLLOWUP_MARKERS = (
    "איזה ",
    "איזו ",
    "ואיך",
    "ומה ",
    "גם ",
    "תוך כמה",
    "למלא",
    "טופס",
    "טפסים",
    "ממשיכ",
    "משם",
    "על זה",
    "בקשר ל",
    "קודם",
    "אמרת",
    "פרטי",
    "הסדר",
)


def _followup_needs_context(question: str) -> bool:
    q = (question or "").strip()
    if not q or any(p in q for p in _STANDALONE_OFF_TOPIC):
        return False
    words = q.split()
    short = len(q) <= 90 or len(words) <= 14
    if not short:
        return False
    if any(m in q for m in _FOLLOWUP_MARKERS):
        return True
    return "?" in q and len(words) <= 10


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


def detect_with_history(question: str, history: list[dict[str, str]] | None) -> Boundary:
    """Apply detect; for follow-ups in an on-topic thread, prefer RAG over out_of_scope."""
    rewritten = (question or "").strip()
    bound = detect(rewritten)
    if bound.kind in ("no_account_access", "personal_boundary", "no_advice"):
        return bound
    if bound.kind != "out_of_scope":
        return bound
    history = history or []
    if not history or not _followup_needs_context(rewritten):
        return bound

    ctx = " ".join((t.get("content") or "").strip() for t in history[-6:] if (t.get("content") or "").strip())
    if ctx:
        widened = detect(f"{ctx} {rewritten}")
        if widened.kind != "out_of_scope":
            return widened

    for turn in reversed(history):
        text = (turn.get("content") or "").strip()
        if not text:
            continue
        prior = detect(text)
        if prior.kind in ("no_account_access", "personal_boundary", "no_advice"):
            continue
        if prior.kind != "out_of_scope" or is_on_topic(text):
            return Boundary()
    return bound
