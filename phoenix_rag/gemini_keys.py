"""Load GEMINI_API_KEY* from the environment and construct google-genai clients (shared by embed + generate)."""

from __future__ import annotations

import os


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


def gemini_client(api_key: str):
    from google import genai

    vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")
    return genai.Client(vertexai=True, api_key=api_key) if vertex else genai.Client(api_key=api_key)
