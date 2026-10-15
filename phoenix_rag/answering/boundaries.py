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
)


@dataclass(frozen=True)
class Boundary:
    """Which canned message to use; empty kind means 'normal RAG path'."""

    kind: str = ""  # personal_boundary | no_advice | no_account_access | out_of_scope | ""
    message_key: str = ""


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
    if not any(p in low for p in _ON_TOPIC):
        return Boundary("out_of_scope", "out_of_scope")
    return Boundary()
