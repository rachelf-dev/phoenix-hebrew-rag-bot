"""Strip control chars from chat input and detect prompt-injection jailbreaks.

Runs before rewrite/RAG/Gemini so insurance-flavored jailbreaks never reach the model.
"""

from __future__ import annotations

import re

DEFAULT_MAX_QUESTION_CHARS = 1000

# Phrase-level only: a lone "התעלם" / "ignore" in a real claim question must not match.
_INJECTION_RE = re.compile(
    r"("
    r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above|all)\s+(instructions|prompts?|rules)"
    r"|disregard\s+(the\s+)?(previous|prior|system)\s+(instructions?|prompts?|rules)"
    r"|forget\s+(all\s+)?(your|the)\s+(instructions|rules|prompts?)"
    r"|override\s+(your|the)\s+(instructions|rules|guardrails|system\s+prompt)"
    r"|reveal\s+(your|the)\s+(system\s+)?(prompt|instructions|hidden\s+rules)"
    r"|show\s+(me\s+)?(your|the)\s+system\s+prompt"
    r"|print\s+(your|the)\s+system\s+prompt"
    r"|new\s+system\s+prompt"
    r"|developer\s+mode"
    r"|jailbreak"
    r"|you\s+are\s+now\s+(dan|jailbreak|unrestricted)"
    r"|act\s+as\s+if\s+you\s+(have|had)\s+no\s+(rules|restrictions|limits)"
    r"|<\s*system[\s_\-]*prompt"
    r"|התעל[םמ]\s+מההוראות"
    r"|תתעל[םמ]\s+מה(הוראות|כללים)"
    r"|שכח(?:י|ו)?\s+את\s+ה(הוראות|כללים)"
    r"|חשוף\s+את\s+(ההנחיות|הפרומפט|הוראות\s+המערכת)"
    r"|תן\s+לי\s+את\s+(הפרומפט|ההנחיות)\s+(שלך|המערכת)"
    r"|התעל[םמ]\s+מהנחיות"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def sanitize_text(text: str, *, max_chars: int = DEFAULT_MAX_QUESTION_CHARS) -> str:
    """Drop control chars, collapse whitespace, and cap length."""
    raw = (text or "").replace("\x00", "")
    cleaned: list[str] = []
    for ch in raw:
        if ch in "\n\t" or ch >= " ":
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    s = "".join(cleaned)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    if max_chars > 0 and len(s) > max_chars:
        s = s[:max_chars].rstrip()
    return s


def sanitize_question(text: str, *, max_chars: int = DEFAULT_MAX_QUESTION_CHARS) -> str:
    """Normalize the current user question before boundaries, rewrite, and RAG."""
    return sanitize_text(text, max_chars=max_chars)


def looks_like_injection(text: str) -> bool:
    """True when the text tries to override the bot instructions (even with insurance words)."""
    if not (text or "").strip():
        return False
    return _INJECTION_RE.search(text) is not None


def sanitize_history(
    history: list[dict[str, str]] | None,
    *,
    max_chars: int = DEFAULT_MAX_QUESTION_CHARS,
) -> list[dict[str, str]]:
    """Normalize turns and drop user turns that look like jailbreaks."""
    out: list[dict[str, str]] = []
    for turn in history or []:
        role = (turn.get("role") or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        content = sanitize_text(turn.get("content") or "", max_chars=max_chars)
        if not content:
            continue
        if role == "user" and looks_like_injection(content):
            continue
        out.append({"role": role, "content": content})
    return out
