"""Gemini Flash grounded generation: answer only from retrieved chunks, else not_found.

On how-to questions, prefers citing a retrieved self-service action page over FAQ/join,
and will not refuse when that action page is already in the retrieved hits."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from ..config import Config
from ..indexing.retrieve import Hit
from .rerank import preferred_cite_urls

log = logging.getLogger(__name__)

SYSTEM = """אתה עוזר שעונה רק מתוך קטעי תוכן ציבוריים מאתר הפניקס (fnx.co.il).
כללים מחייבים:
1. ענה בעברית, קצר וברור (עד כ-8 משפטים).
2. השתמש רק בעובדות שמופיעות בקטעים למטה. אסור להמציא, להשלים או לנחש.
3. אם הקטעים לא מכילים תשובה מספקת לשאלה — status חייב להיות not_found. אל תנחש.
   אם השאלה דורשת מספר מדויק, מחיר אחיד, דמי השתתפות קבועים לכל המבוטחים, או נתון אישי שלא כתוב במפורש בקטעים — not_found.
4. citation_urls חייבות להיות כתובות שהופיעו בשדה "כתובת" של הקטעים, מועתקות במדויק.
5. אל תיתן ייעוץ אישי ("כדאי לך לקנות") ואל תתיימר לגשת לחשבון, לתביעה או לפוליסה אישית.
6. אם שואלים איך מגישים / לאן שולחים ויש קישורי טפסים בקטע, ציין אותם.
7. כששואלים איך מבצעים פעולה (הארכת ביטוח, עדכון פרטים אישיים, שינוי מסלול השקעה, חידוש, הגשת תביעה) — חובה לצטט את דף הפעולה / שירות עצמי (סוג "service" או כתובת עם /customer-service/) ולא FAQ, הצטרפות או מאמר כללי, אם שניהם בקטעים.
8. דפי שירות עצמי לעדכון פרטים / הארכת פוליסה / שינוי מסלול הם תוכן ציבורי באתר — זה לא גישה לחשבון אישי. אם דף כזה מופיע בקטעים, status חייב להיות answered ו-citation_urls חייבות לכלול את כתובתו.
9. כששואלים מה כולל מוצר / מה זה / למי מיועד — עדיף לצטט את דף המוצר עצמו (סוג "product") ולא מאמר הסבר או דף בית של התחום.
10. כש-status הוא answered חובה לציין לפחות כתובת אחת ב-citation_urls.
החזר JSON בלבד במבנה:
{"status":"answered"|"not_found","answer":"...","citation_urls":["https://..."]}
כש-status הוא not_found, השאר את answer ריק."""


def _client(api_key: str):
    from google import genai

    vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")
    return genai.Client(vertexai=True, api_key=api_key) if vertex else genai.Client(api_key=api_key)


def _types():
    from google.genai import types

    return types


_key_i = 0  # last Gemini key that succeeded; rotate from here on the next 429
_preferred_model: str | None = None  # last model that returned 200 — skip dead 404s on later items
_dead_models: set[str] = set()  # retired / 404-for-all-keys models (e.g. 2.5-flash-lite)


def gemini_keys() -> list[str]:
    """GEMINI_API_KEY, optional GEMINI_API_KEY_1..15, optional comma-separated GEMINI_API_KEYS."""
    raw: list[str] = [os.getenv("GEMINI_API_KEY") or ""]
    raw.extend((os.getenv("GEMINI_API_KEYS") or "").split(","))
    for n in range(1, 16):
        raw.append(os.getenv(f"GEMINI_API_KEY_{n}") or "")
    keys: list[str] = []
    seen: set[str] = set()
    for k in raw:
        k = k.strip()
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def google_client():
    keys = gemini_keys()
    if not keys:
        raise RuntimeError("GEMINI_API_KEY is missing; copy .env.example to .env")
    return _client(keys[0])


def _response_text(resp: Any) -> str:
    text = getattr(resp, "text", None)
    if text:
        return text
    try:
        parts = resp.candidates[0].content.parts
        return "".join(getattr(p, "text", "") or "" for p in parts)
    except Exception:
        return ""


def _parse_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def _chunks_block(hits: list[Hit]) -> str:
    parts: list[str] = []
    for i, h in enumerate(hits, 1):
        url = h.display_url or h.url
        forms = f"\nקישורי טפסים:\n{h.form_links}" if h.form_links else ""
        parts.append(
            f"[קטע {i}]\nכתובת: {url}\nסוג: {h.section_type}\nכותרת: {h.page_title}\nנתיב: {h.heading_path}\n"
            f"טקסט:\n{(h.text or '').strip()}{forms}"
        )
    return "\n\n".join(parts)


def rewrite_question(question: str, history: list[dict[str, str]], cfg: Config) -> str:
    """Turn a follow-up into a standalone Hebrew question. Returns `question` on failure / no history."""
    if not history:
        return question
    turns = history[-6:]
    lines = []
    for t in turns:
        role = "משתמש" if t.get("role") == "user" else "עוזר"
        lines.append(f"{role}: {t.get('content', '')}")
    prompt = (
        "כתוב מחדש את שאלת המשתמש האחרונה כשאלה עצמאית בעברית, בלי להוסיף עובדות שלא נאמרו.\n"
        "אם היא כבר עצמאית, החזר אותה כמו שהיא. החזר רק את השאלה, בלי הקדמה.\n\n"
        "שיחה:\n"
        + "\n".join(lines)
        + f"\n\nשאלה אחרונה: {question}"
    )
    try:
        return _generate_text(prompt, cfg, max_tokens=120, json_mode=False).strip() or question
    except Exception as exc:
        log.warning("rewrite failed: %s", str(exc)[:200])
        return question


def _action_fallback(hits: list[Hit], urls: list[str]) -> dict[str, Any]:
    """Grounded how-to answer from a retrieved self-service page when Gemini refuses."""
    chosen = [h for h in hits if (h.display_url or h.url) in set(urls)] or hits[:1]
    title = (chosen[0].page_title or "").strip()
    body = (chosen[0].text or "").strip()
    snippet = body.split("\n\n")[0].strip()[:700]
    answer = "\n".join(p for p in (title, snippet) if p).strip()
    if not answer:
        answer = "ניתן לבצע את הפעולה בדף השירות העצמי באתר הפניקס."
    return {"status": "answered", "answer": answer, "citation_urls": urls}


def generate_answer(question: str, hits: list[Hit], cfg: Config) -> dict[str, Any]:
    """Call Gemini; returns {status, answer, citation_urls}. Raises on missing key."""
    if not hits:
        return {"status": "not_found", "answer": "", "citation_urls": []}
    allowed = {(h.display_url or h.url) for h in hits}
    extra = preferred_cite_urls(question, hits)
    user = f"שאלה:\n{question}\n\nקטעים מהאתר:\n{_chunks_block(hits)}"
    if extra:
        user += (
            "\n\nדף הפעולה המתאים לשאלה כבר נמצא בקטעים. "
            "ענה ממנו, ציין את כתובתו ב-citation_urls, ואל תחזיר not_found. "
            "זה דף ציבורי באתר, לא גישה לחשבון אישי.\n"
            + "\n".join(extra)
        )
    raw = _generate_text(user, cfg, max_tokens=int(cfg.llm.get("max_output_tokens") or 600), json_mode=True,
                         system=SYSTEM)
    try:
        data = _parse_json(raw)
    except (json.JSONDecodeError, ValueError):
        log.warning("Gemini returned non-JSON")
        if extra:
            return _action_fallback(hits, extra)
        return {"status": "not_found", "answer": "", "citation_urls": []}
    status = str(data.get("status") or "not_found").strip()
    if status not in ("answered", "not_found"):
        status = "not_found"
    cites = []
    for u in data.get("citation_urls") or []:
        u = str(u).strip()
        if u in allowed:
            cites.append(u)
        else:
            match = next((a for a in allowed if a.rstrip("/") in u or u.rstrip("/") in a), "")
            if match:
                cites.append(match)
    if status != "answered":
        if extra:
            return _action_fallback(hits, extra)
        return {"status": status, "answer": str(data.get("answer") or "").strip(), "citation_urls": []}
    if any(p in question for p in ("לכל המבוטחים", "מחיר אחיד", "מספר מדויק", "באחוזים עם")):
        return {"status": "not_found", "answer": "", "citation_urls": []}
    cites = list(dict.fromkeys([*extra, *cites]))
    if not cites:
        cites = [h.display_url or h.url for h in hits[:2] if (h.display_url or h.url)]
    answer = str(data.get("answer") or "").strip()
    if extra and not answer:
        return _action_fallback(hits, extra)
    return {"status": status, "answer": answer, "citation_urls": list(dict.fromkeys(cites))}


def _generate_text(user: str, cfg: Config, *, max_tokens: int, json_mode: bool, system: str | None = None) -> str:
    global _key_i, _preferred_model, _dead_models
    types = _types()
    keys = gemini_keys()
    if not keys:
        raise RuntimeError("GEMINI_API_KEY is missing; copy .env.example to .env")
    models = [cfg.llm.get("model") or "gemini-3.5-flash"]
    for extra in (cfg.llm.get("fallback_model"), "gemini-3.8-flash", "gemini-3.5-flash-lite"):
        if extra and extra not in models:
            models.append(extra)
    models = [m for m in models if m not in _dead_models]
    if _preferred_model and _preferred_model in models:
        models = [_preferred_model] + [m for m in models if m != _preferred_model]
    base: dict[str, Any] = {
        "temperature": float(cfg.llm.get("temperature") or 0.1),
        "max_output_tokens": max_tokens,
    }
    if system:
        base["system_instruction"] = system
    if json_mode:
        base["response_mime_type"] = "application/json"

    configs: list[Any] = [types.GenerateContentConfig(**base)]
    try:
        configs.append(types.GenerateContentConfig(**base, thinking_config=types.ThinkingConfig(thinking_budget=0)))
    except Exception:
        pass

    start = _key_i % len(keys)
    key_order = list(range(start, len(keys))) + list(range(0, start))
    last: Exception | None = None
    for model in models:
        skip_model = False
        for ki in key_order:
            client = _client(keys[ki])
            quota_this_key = False
            for config in configs:
                delay = 2.0
                for attempt in range(1, 4):
                    try:
                        resp = client.models.generate_content(model=model, contents=user, config=config)
                        text = _response_text(resp)
                        if text.strip():
                            _key_i = ki
                            primary = cfg.llm.get("model") or "gemini-3.5-flash"
                            if model == primary:
                                _preferred_model = model
                            if len(keys) > 1:
                                log.info("Gemini ok model=%s key=%d/%d", model, ki + 1, len(keys))
                            return text
                        last = RuntimeError("empty Gemini response")
                        break
                    except Exception as exc:
                        last = exc
                        msg = str(exc)
                        if "blockByNetFree" in msg:
                            raise RuntimeError(f"NetFree blocked Gemini generate: {msg[:300]}") from exc
                        if "thinking" in msg.lower() or "INVALID_ARGUMENT" in msg or "invalid argument" in msg.lower():
                            log.warning("Gemini 400/config on %s key=%d; trying next config", model, ki + 1)
                            break
                        if any(s in msg for s in ("NOT_FOUND", "404", "not found")):
                            if "no longer available" in msg.lower():
                                _dead_models.add(model)
                                skip_model = True
                                log.warning("Gemini model %s retired; skipping for this process", model)
                            else:
                                log.warning("Gemini model %s not on key=%d; trying next key", model, ki + 1)
                            quota_this_key = True
                            break
                        if any(s in msg for s in (
                            "429", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "503",
                            "getaddrinfo", "ConnectError", "ConnectTimeout", "disconnected", "timeout", "Timeout",
                        )):
                            if len(keys) > 1:
                                log.warning("Gemini quota on %s key=%d/%d; rotating", model, ki + 1, len(keys))
                                quota_this_key = True
                                break
                            server_delay = _retry_delay(msg)
                            if attempt >= 3:
                                log.warning("Gemini quota on %s key=%d/%d; rotating", model, ki + 1, len(keys))
                                quota_this_key = True
                                break
                            wait = max(delay, min(server_delay, 20.0) if server_delay else 8.0)
                            log.warning("Gemini retry %d/3 on %s key=%d (sleep %.1fs)", attempt, model, ki + 1, wait)
                            time.sleep(wait)
                            delay = min(delay * 2, 40)
                            continue
                        raise
                if skip_model or quota_this_key:
                    break
            if skip_model:
                break
    raise RuntimeError(f"Gemini generate failed: {last}") from last


def _retry_delay(msg: str) -> float:
    """Seconds the Gemini 429 body asks us to wait, if any."""
    m = re.search(r"retryDelay'?\"?:\s*'?\"?(\d+(?:\.\d+)?)s", msg) or re.search(r"retry in (\d+(?:\.\d+)?)s", msg)
    return float(m.group(1)) if m else 0.0
