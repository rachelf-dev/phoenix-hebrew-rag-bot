"""Conversation window, topic anchors, and follow-up query enrichment for RAG.

Keeps rewrite/retrieval/generation on the same 4–6 turn history so short
follow-ups ('איזה טופס למלא?', 'האם זה משנה?') inherit the thread topic.
"""

from __future__ import annotations

from typing import Any

# Default: last 8 messages (4 Q&A pairs). Config may raise this to 12.
HISTORY_WINDOW_DEFAULT = 8
HISTORY_WINDOW_MIN = 4
HISTORY_WINDOW_MAX = 12

# Longer phrases first so "שרברב פרטי" wins over "שרברב".
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
    "מקיף",
    "חובה",
    "שמאי",
    "פיצוי",
)

_FOLLOWUP_MARKERS = (
    "איזה ",
    "איזו ",
    "ואיך",
    "ומה ",
    "וגם ",
    "ואם ",
    "ואז ",
    "גם ",
    "תוך כמה",
    "למלא",
    "טופס",
    "טפסים",
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
    "פרטי",
    "הסדר",
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
    assistant_limit: int = 360,
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


def is_followup(question: str) -> bool:
    """True for short / deictic turns that are not usable as a standalone search query."""
    q = (question or "").strip()
    if not q:
        return False
    words = q.split()
    short = len(q) <= 90 or len(words) <= 14
    if not short:
        return False
    low = q.lower()
    if any(m in q or m in low for m in _FOLLOWUP_MARKERS):
        return True
    if any(m in q for m in _DEICTIC):
        return True
    if "?" in q and len(words) <= 12:
        return True
    return len(words) <= 8


def topic_anchors(history: list[dict[str, str]] | None, question: str = "") -> list[str]:
    """Insurance/claim phrases already mentioned in the recent thread (longest first, de-duped)."""
    parts = [(t.get("content") or "") for t in (history or [])]
    if question:
        parts.append(question)
    blob = " ".join(parts)
    found: list[str] = []
    for phrase in _TOPIC_PHRASES:
        if phrase in blob and phrase not in found:
            found.append(phrase)
    compact: list[str] = []
    for phrase in found:
        if any(phrase != other and phrase in other for other in found):
            continue
        compact.append(phrase)
    related_map = (
        (("שרברב", "שרברב פרטי", "שרברב הסדר", "צנרת"), ("נזקי מים", "ביטוח דירה", "תביעה")),
        (("דוד", "דוד שמש", "דוד השמש"), ("ביטוח דירה",)),
    )
    extra: list[str] = []
    for seeds, related in related_map:
        if any(s in found for s in seeds):
            extra.extend(r for r in related if r not in found and r not in extra)
    return compact + extra


def thread_has_topics(history: list[dict[str, str]] | None) -> bool:
    return bool(topic_anchors(history))


def heuristic_rewrite(original: str, rewritten: str, history: list[dict[str, str]] | None) -> str:
    """If the LLM rewrite stays vague, pin the question to thread topics (plumber, boiler, apartment…)."""
    orig = (original or "").strip()
    rew = (rewritten or "").strip() or orig
    turns = history_window(history)
    if not turns:
        return rew[:900]
    if not is_followup(orig) and len(orig) > 120:
        return rew[:900]

    topics = topic_anchors(turns, orig)
    missing = [t for t in topics if t not in rew]
    if not missing:
        return rew[:900]
    if is_followup(orig) or len(orig.split()) <= 12 or len(rew) <= len(orig) + 12:
        return f"{rew} — {', '.join(missing[:8])}"[:900]
    return rew[:900]


def retrieval_query(rewritten: str, history: list[dict[str, str]] | None) -> str:
    """Search text: standalone rewrite plus any thread topics the rewrite still omitted."""
    q = (rewritten or "").strip()
    if not q:
        return q
    turns = history_window(history)
    if not turns:
        return q[:900]
    missing = [t for t in topic_anchors(turns, q) if t not in q]
    if not missing:
        return q[:900]
    return f"{q} {' '.join(missing[:8])}".strip()[:900]
