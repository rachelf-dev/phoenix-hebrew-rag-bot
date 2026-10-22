"""Conversation window, thread slots, and follow-up query enrichment for RAG.

Keeps rewrite/retrieval/generation on the same 4–6 turn history so short
follow-ups inherit apartment / plumber / boiler context as standalone search queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Default: last 8 messages (4 Q&A pairs). Config may raise this to 12.
HISTORY_WINDOW_DEFAULT = 8
HISTORY_WINDOW_MIN = 4
HISTORY_WINDOW_MAX = 12

_PRODUCT_RULES = (
    ("ביטוח דירה", ("ביטוח דירה", "ביטוח מבנה", "ביטוח תכולה", "דירה", "מבנה", "תכולה", "דירת")),
    ("ביטוח רכב", ("ביטוח רכב", "רכב", "מקיף", "חובה", "מוסך", "תאונה", "תאונ")),
    ("ביטוח נסיעות", ("ביטוח נסיעות", "נסיעות", "travel")),
    ("ביטוח בריאות", ("ביטוח בריאות", "בריאות", "health")),
)

_COVERAGE_RULES = (
    ("שרברב פרטי", ("שרברב פרטי", "שרברבים פרטי", "בעל מקצוע פרטי")),
    ("שרברב הסדר", ("שרברב הסדר", "שרברבים הסדר", "נותן שירות", "נותן השירות")),
)

_DAMAGE_RULES = (
    ("נזקי מים", ("נזקי מים", "נזילה", "נזיל")),
    ("נזקי מזג אוויר", ("נזקי מזג אוויר", "מזג אוויר", "ממזג אוויר", "ממזג האוויר", "ממזג")),
    ("צנרת", ("צנרת", "צינור", "pipeline")),
    ("דוד שמש", ("דוד שמש", "דוד השמש", "דוד")),
    ("נזק למבנה", ("נזק למבנה", "נזק לתכולה", "נזק לתכולה")),
    ("נזק לצד שלישי", ("נזק לצד שלישי", "צד שלישי", "שכן", "השכן", "שכנ")),
)

# Phrases too broad to append to hybrid search (they pull FAQ / car / articles).
_GENERIC_ANCHORS = frozenset({
    "כיסוי",
    "תביעה",
    "תביעת",
    "טופס",
    "טפסים",
    "פוליסה",
    "דירה",
    "רכב",
    "נסיעות",
    "דוד",
    "שרברב",
    "שרברבים",
    "נזיל",
    "נזילה",
    "תכולה",
    "מקיף",
    "חובה",
    "שמאי",
    "פיצוי",
})

_TOPIC_PHRASES = (
    "שרברב הסדר",
    "שרברב פרטי",
    "נזק לצד שלישי",
    "צד שלישי",
    "ביטוח דירה",
    "ביטוח תכולה",
    "ביטוח רכב",
    "ביטוח נסיעות",
    "ביטוח בריאות",
    "נזקי מים",
    "דוד השמש",
    "דוד שמש",
    "מוסך הסדר",
    "שירותי דרך",
    "גרירה",
    "גרר",
    "עדכון פרטים",
    "שירות עצמי",
    "הארכת ביטוח",
    "נזק למבנה",
    "שרברב",
    "שרברבים",
    "צנרת",
    "נזילה",
    "נזיל",
    "תכולה",
    "מבוטח",
)

_FOLLOWUP_MARKERS = (
    "הנזק ",
    "הנזק הוא",
    "ממזג",
    "מזג אוויר",
    "איזה ",
    "איזו ",
    "ואיך",
    "ומה ",
    "וגם ",
    "ואם ",
    "ואז ",
    "גם ",
    "תוך כמה",
    "ממשיכ",
    "משם",
    "מכאן",
    "על זה",
    "האם זה",
    "האם גם",
    "בקשר ל",
    "מה לגבי",
    "ומה לגבי",
    "קודם",
    "אמרת",
    "הטופס",
    "הכיסוי",
    "התביעה",
    "הפעולה",
    "אותו",
    "אותה",
    "אותם",
    "כזה",
    "ככה",
    "משנה",
    "עוד משהו",
    "הצעד הבא",
    "מה עושים",
    "this",
    "that",
    "the form",
)

_DEICTIC = (
    " זה",
    "זה ",
    "זאת",
    "הזה",
    "שם",
    "כך",
    "ככה",
    "אותו",
    "אותה",
)

# Claim decision timeline — not the same as "how to file a claim" (must stay not_found if absent on site).
_CLAIM_SLA_MARKERS = (
    "תוך כמה זמן מקבלים תשובה",
    "תוך כמה זמן",
    "מתי התביעה מאושרת",
    "מתי מאושרת התביעה",
    "זמן טיפול בתביעה",
    "זמן אישור התביעה",
    "מקבלים תשובה על תביע",
    "כמה זמן לוקח התביעה",
    "כמה זמן עד",
)


def is_claim_sla_question(question: str) -> bool:
    """True when the user asks for claim decision / response time (usually not on the public site)."""
    q = (question or "").strip()
    if not q:
        return False
    if any(p in q for p in _CLAIM_SLA_MARKERS):
        return True
    if "תוך כמה" in q and any(w in q for w in ("תשובה", "תביע", "מאושר", "טיפול")):
        return True
    return False


@dataclass
class ThreadSlots:
    product: str = ""
    coverage: str = ""
    damages: list[str] = field(default_factory=list)
    wants_form: bool = False
    wants_claim: bool = False


def history_window_size(cfg: Any | None = None) -> int:
    """Last N chat messages sent to rewrite + Gemini (clamped to 4–12)."""
    raw = HISTORY_WINDOW_DEFAULT
    if cfg is not None:
        try:
            raw = int((getattr(cfg, "retrieval", None) or {}).get("history_window") or raw)
        except (TypeError, ValueError):
            raw = HISTORY_WINDOW_DEFAULT
    return max(HISTORY_WINDOW_MIN, min(raw, HISTORY_WINDOW_MAX))


def history_window(
    history: list[dict[str, str]] | None,
    n: int | None = None,
    *,
    user_limit: int = 500,
    assistant_limit: int = 500,
) -> list[dict[str, str]]:
    """Normalize roles and keep the last `n` non-empty turns, trimmed for tokens."""
    limit = HISTORY_WINDOW_DEFAULT if n is None else n
    limit = max(HISTORY_WINDOW_MIN, min(int(limit), HISTORY_WINDOW_MAX))
    out: list[dict[str, str]] = []
    for t in history or []:
        role = (t.get("role") or "").strip().lower()
        if role in ("human", "client"):
            role = "user"
        elif role in ("bot", "model", "ai"):
            role = "assistant"
        if role not in ("user", "assistant"):
            continue
        content = (t.get("content") or "").strip()
        if not content:
            continue
        cap = user_limit if role == "user" else assistant_limit
        if len(content) > cap:
            content = content[: cap - 1].rstrip() + "…"
        out.append({"role": role, "content": content})
    return out[-limit:]


def format_history_lines(turns: list[dict[str, str]]) -> list[str]:
    """Hebrew role labels for Gemini prompts."""
    lines: list[str] = []
    for t in turns:
        role = "משתמש" if t.get("role") == "user" else "עוזר"
        lines.append(f"{role}: {t.get('content', '')}")
    return lines


def _conversation_blob(history: list[dict[str, str]] | None, question: str = "") -> str:
    parts = [(t.get("content") or "") for t in (history or [])]
    if question:
        parts.append(question)
    return " ".join(parts)


def _user_blob(history: list[dict[str, str]] | None, question: str = "") -> str:
    """User turns only — avoids assistant boilerplate polluting damage/coverage slots."""
    parts: list[str] = []
    for t in history or []:
        role = (t.get("role") or "").strip().lower()
        if role in ("human", "client", "user"):
            parts.append(t.get("content") or "")
    if question:
        parts.append(question)
    return " ".join(parts)


def _is_short_continuation(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    return len(q.split()) <= 12 or len(q) <= 80


def _slots_from_blob(blob: str) -> ThreadSlots:
    slots = ThreadSlots()
    for label, keys in _PRODUCT_RULES:
        if any(k in blob for k in keys):
            slots.product = label
            break
    for label, keys in _COVERAGE_RULES:
        if any(k in blob for k in keys):
            slots.coverage = label
            break
    for label, keys in _DAMAGE_RULES:
        if any(k in blob for k in keys) and label not in slots.damages:
            slots.damages.append(label)
    slots.wants_form = any(w in blob for w in ("טופס", "טפסים", "למלא", "הורד", "PDF", "pdf"))
    slots.wants_claim = any(w in blob for w in ("תביע", "מגיש", "מגישים", "claim"))
    if slots.product == "ביטוח דירה" and not slots.damages and any(w in blob for w in ("נזק", "מזיק", "נשבר")):
        if "דוד" in blob or "שמש" in blob:
            slots.damages.append("דוד שמש")
    if not slots.product and slots.damages:
        if any(
            d in slots.damages
            for d in ("דוד שמש", "צנרת", "נזקי מים", "נזקי מזג אוויר", "נזק למבנה", "נזק לצד שלישי")
        ):
            slots.product = "ביטוח דירה"
    return slots


def _merge_thread_slots(base: ThreadSlots, inherited: ThreadSlots) -> ThreadSlots:
    out = ThreadSlots(
        product=base.product or inherited.product,
        coverage=base.coverage or inherited.coverage,
        damages=list(dict.fromkeys([*base.damages, *inherited.damages])),
        wants_form=base.wants_form or inherited.wants_form,
        wants_claim=base.wants_claim or inherited.wants_claim,
    )
    return out


def extract_thread_slots(history: list[dict[str, str]] | None, question: str = "") -> ThreadSlots:
    """Structured insurance context from the thread + current question (user turns weighted)."""
    user_blob = _user_blob(history, question)
    slots = _slots_from_blob(user_blob)
    if history and (_is_short_continuation(question) or is_followup(question, history)):
        prior = _slots_from_blob(_user_blob(history, ""))
        slots = _merge_thread_slots(slots, prior)
    if "דוד שמש" in slots.damages and "צנרת" in user_blob and "צנרת" not in slots.damages:
        slots.damages.append("צנרת")
    return slots


def thread_has_insurance_slots(history: list[dict[str, str]] | None, question: str = "") -> bool:
    s = extract_thread_slots(history, question)
    return bool(s.product or s.coverage or s.damages or s.wants_claim or s.wants_form)


def is_followup(question: str, history: list[dict[str, str]] | None = None) -> bool:
    """True for short / deictic turns that need thread context for search."""
    from .sanitize import is_unintelligible_input

    q = (question or "").strip()
    if not q or is_unintelligible_input(q):
        return False
    words = q.split()
    short = len(q) <= 90 or len(words) <= 14
    if not short:
        return False
    if any(m in q for m in _DEICTIC):
        return True
    if any(m in q for m in _FOLLOWUP_MARKERS):
        return True
    if "?" in q and len(words) <= 12:
        return True
    if history and len(words) <= 14 and thread_has_insurance_slots(history, q):
        return len(words) <= 10
    return False


def topic_anchors(history: list[dict[str, str]] | None, question: str = "") -> list[str]:
    """Specific insurance phrases for retrieval boost (no generic 'כיסוי' / 'תביעה')."""
    blob = _conversation_blob(history, question)
    found: list[str] = []
    for phrase in _TOPIC_PHRASES:
        if phrase in blob and phrase not in found:
            found.append(phrase)
    compact: list[str] = []
    for phrase in found:
        if any(phrase != other and phrase in other for other in found):
            continue
        compact.append(phrase)
    slots = extract_thread_slots(history, question)
    for d in slots.damages:
        if d not in compact:
            compact.append(d)
    if slots.coverage and slots.coverage not in compact:
        compact.append(slots.coverage)
    if slots.product and slots.product not in compact:
        compact.append(slots.product)
    return [p for p in compact if p not in _GENERIC_ANCHORS]


def thread_has_topics(history: list[dict[str, str]] | None) -> bool:
    return thread_has_insurance_slots(history)


def _damage_phrase(slots: ThreadSlots) -> str:
    if not slots.damages:
        return ""
    ordered = list(dict.fromkeys(slots.damages))
    if len(ordered) == 1:
        return ordered[0]
    return " או ".join(ordered[:3])


def _compose_from_slots(original: str, slots: ThreadSlots) -> str:
    """Build a standalone Hebrew search question from thread slots."""
    orig = (original or "").strip()
    low = orig.lower()
    product = slots.product or "ביטוח דירה"
    damage = _damage_phrase(slots)
    cov = f" עם {slots.coverage}" if slots.coverage else ""

    if is_claim_sla_question(orig):
        if damage:
            return f"{orig} — זמן החלטה / מענה על תביעת {damage} ב{product}"
        return orig

    if any(p in orig for p in ("איזה טופס", "איזו טופס", "טופס למלא", "למלא")):
        target = damage or "נזק בדירה"
        return f"איזה טופס למלא לתביעת {target} ב{product}{cov}".strip()

    if "האם זה משנה" in orig or (orig.endswith("משנה?") and "האם" in orig):
        if slots.coverage:
            return f"האם {slots.coverage} משנה לתביעת {damage or 'נזק בדירה'} ב{product}"
        return f"האם סוג הכיסוי או הנזק משנה לתביעת {damage or 'נזק בדירה'} ב{product}"

    if any(p in orig for p in ("ואיך ממשיכ", "משם", "מכאן", "הצעד הבא")):
        return f"איך ממשיכים בהגשת תביעת {damage or 'נזק'} ב{product}{cov}"

    if any(p in orig for p in ("מזג אוויר", "ממזג", "נזקי מזג")):
        if "דוד שמש" in slots.damages or "דוד" in orig:
            return (
                f"מה עושים כשהנזק לדוד השמש כתוצאה מנזקי מזג אוויר ב{product}"
                f" — תביעה והרחבות לדודים"
            )

    if damage and not any(d in orig for d in slots.damages):
        if "תביע" in orig or "מגיש" in orig or "מגישים" in orig:
            return f"איך מגישים תביעת {damage} ב{product}{cov}"
        if "מה עושים" in orig or "מה לעשות" in orig:
            return f"מה עושים במקרה של {damage} ב{product}{cov}"

    if "שכן" in low or "נזק לצד שלישי" in slots.damages:
        if "דוד" in orig or "שמש" in orig:
            dmg = "דוד שמש"
        elif "צנרת" in orig:
            dmg = "צנרת"
        else:
            dmg = damage or "נזק לצד שלישי"
        return f"תביעת נזק מ{dmg} של שכן ב{product} — כיסוי ומה לעשות"

    if damage and len(orig.split()) <= 14:
        from .boundaries import is_on_topic
        from .sanitize import is_unintelligible_input

        if not is_unintelligible_input(orig) and (
            is_on_topic(orig)
            or any(m in orig for m in _FOLLOWUP_MARKERS)
            or any(m in orig for m in _DEICTIC)
        ):
            return f"{orig} — {damage} ב{product}{cov}".strip(" —")

    if damage and ("תביע" in orig or "מגיש" in orig) and slots.wants_claim:
        return f"תביעת {damage} ב{product}{cov}"

    if damage and product and ("תביע" in orig or "מגיש" in orig):
        return f"{orig} — תביעת {damage} ב{product}{cov}"

    return orig


def _rewrite_needs_slots(original: str, rewritten: str, slots: ThreadSlots) -> bool:
    if is_claim_sla_question(original):
        return False
    if not slots.product and not slots.damages and not slots.coverage:
        return False
    orig = (original or "").strip()
    rew = (rewritten or "").strip()
    if is_followup(orig):
        return True
    if len(orig.split()) <= 14 and (slots.damages or slots.coverage):
        return True
    if slots.damages and not any(d in rew for d in slots.damages):
        return True
    if slots.product and slots.product not in rew and "ביטוח" not in rew:
        return True
    return len(rew) <= len(orig) + 15


def heuristic_rewrite(original: str, rewritten: str, history: list[dict[str, str]] | None) -> str:
    """Turn vague follow-ups into standalone search questions using thread slots."""
    from .sanitize import is_unintelligible_input

    orig = (original or "").strip()
    rew = (rewritten or "").strip() or orig
    if is_unintelligible_input(orig):
        return orig
    turns = history_window(history)
    if not turns:
        return rew[:900]

    slots = extract_thread_slots(turns, orig)
    if is_claim_sla_question(orig):
        return _compose_from_slots(orig, slots)[:900]

    if is_followup(orig, turns) and (slots.damages or slots.product or slots.coverage):
        composed = _compose_from_slots(orig, slots)
        if composed != orig:
            return composed[:900]

    if not is_followup(orig, turns) and len(orig) > 120 and not slots.damages:
        return rew[:900]

    if slots.damages:
        if not all(d in rew for d in slots.damages[:2]) or (slots.product and slots.product not in rew):
            composed = _compose_from_slots(orig, slots)
            if composed and composed != orig:
                return composed[:900]

    if _rewrite_needs_slots(orig, rew, slots):
        composed = _compose_from_slots(orig, slots)
        if composed and composed != orig:
            return composed[:900]

    if _rewrite_needs_slots(orig, rew, slots):
        extra = topic_anchors(turns, orig)
        if extra:
            return f"{rew} {' '.join(extra[:6])}".strip()[:900]
    return rew[:900]


def retrieval_query(rewritten: str, history: list[dict[str, str]] | None) -> str:
    """Search text: standalone rewrite plus only specific missing anchors."""
    from .sanitize import is_unintelligible_input

    q = (rewritten or "").strip()
    if not q or is_unintelligible_input(q):
        return q
    turns = history_window(history)
    if not turns:
        return q[:900]
    missing = [t for t in topic_anchors(turns, q) if t not in q]
    if not missing:
        return q[:900]
    return f"{q} {' '.join(missing[:6])}".strip()[:900]


def claim_path_markers(question: str, history: list[dict[str, str]] | None = None) -> list[str]:
    """URL path substrings to promote when retrieval targets apartment/car claims."""
    if is_claim_sla_question(question):
        return []
    user_blob = _user_blob(history, question)
    slots = extract_thread_slots(history, question)
    markers: list[str] = []
    plumber_topic = (
        slots.coverage in ("שרברב הסדר", "שרברב פרטי")
        or "צנרת" in slots.damages
        or any(x in user_blob for x in ("צנרת", "שרברב", "pipeline", "נזיל", "נזקי מים"))
    )
    boiler_topic = (
        "דוד שמש" in slots.damages
        or "נזקי מזג אוויר" in slots.damages
        or any(x in user_blob for x in ("דוד", "שמש", "water-heater", "מזג אוויר", "ממזג"))
    )
    if plumber_topic:
        markers.append("/claims-customer-service/claims-appartment-insurance/pipeline-damage/")
    if boiler_topic:
        markers.append("/claims-customer-service/claims-appartment-insurance/water-heater/")
    blob = f"{question} {_conversation_blob(history)}"
    if "נזק לצד שלישי" in slots.damages or any(x in blob for x in ("צד שלישי", "שכן", "השכן")):
        markers.append("/claims-customer-service/claims-appartment-insurance/")
    home_claim = (
        "בית" in blob
        and "עסק" not in blob
        and (slots.wants_claim or slots.damages or "תביע" in blob)
    )
    if slots.product == "ביטוח דירה" or "דירה" in blob or home_claim:
        if slots.wants_claim or "תביע" in blob:
            markers.append("/claims-customer-service/claims-appartment-insurance/")
    if slots.product == "ביטוח רכב" or "רכב" in blob:
        markers.append("/claims-customer-service/claims-car-insurance/")
    return list(dict.fromkeys(markers))
