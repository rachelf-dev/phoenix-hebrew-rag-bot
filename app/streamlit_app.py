# Streamlit UI: Phoenix site mock with an embedded chat widget (design from עיצוב.txt).

from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.site_mock import CHAT_WIDGET_HEADER, SITE_BACKDROP_HTML  # noqa: E402
from app.streamlit_theme import PHOENIX_CSS  # noqa: E402
from phoenix_rag.answering.pipeline import Pipeline  # noqa: E402
from phoenix_rag.config import load_config  # noqa: E402
from phoenix_rag.indexing.retrieve import Hit  # noqa: E402

SAMPLE_QUESTIONS = [
    "יש לי נזילה בבית, איך מגישים תביעה?",
    "איך מזמינים גרר ושירותי דרך ברכב?",
    "נשבר לי דוד השמש, מה עושים?",
]

GREETING = (
    "שלום! אני **עוזר השירות החכם** של הפניקס.\n\n"
    "אני עונה בדיוק מתוך עמודי **fnx.co.il** ומצרף קישורים לטפסים ולמקורות. "
    "אם משהו לא כתוב באתר — אגיד במפורש."
)


def _hits_payload(hits: list[Hit], limit: int = 5) -> list[dict]:
    out: list[dict] = []
    for h in hits[:limit]:
        url = h.display_url or h.url
        preview = (h.text or "").strip().replace("\n", " ")
        if len(preview) > 420:
            preview = preview[:417] + "…"
        out.append({
            "chunk_id": h.chunk_id,
            "url": url,
            "page_title": h.page_title or "",
            "heading_path": h.heading_path or "",
            "score": round(float(h.score or 0), 4),
            "rerank_score": round(float(h.rerank_score or 0), 4) if h.rerank_score else None,
            "preview": preview,
        })
    return out


def _source_pills(citations: list[str]) -> str:
    parts: list[str] = []
    for url in citations:
        safe = html.escape(url, quote=True)
        label = html.escape(url.replace("https://www.fnx.co.il", "fnx.co.il"), quote=False)
        parts.append(
            f'<a class="fnx-source-pill" href="{safe}" target="_blank" rel="noopener">'
            f"🔗 {label}</a>"
        )
    return "".join(parts)


@st.cache_resource(show_spinner="טוען מודל ואינדקס…")
def _pipeline() -> tuple[Pipeline, object]:
    cfg = load_config()
    if not cfg.index_npy_path.is_file() or not cfg.index_chunks_path.is_file():
        raise FileNotFoundError(
            f"חסר אינדקס NumPy ({cfg.index_npy_path}, {cfg.index_chunks_path}). "
            "הריצו: python scripts/build_index_local.py"
        )
    pipe = Pipeline(cfg)
    pipe.ensure()
    return pipe, cfg


def _reset_session() -> None:
    st.session_state.messages = [{"role": "assistant", "content": GREETING, "meta": {}}]
    st.session_state.llm_history = []


def _init_state() -> None:
    if "messages" not in st.session_state:
        _reset_session()
    if "llm_history" not in st.session_state:
        st.session_state.llm_history = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def main() -> None:
    st.set_page_config(
        page_title="הפניקס — עוזר שירות (Phoenix RAG)",
        page_icon="🔶",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(PHOENIX_CSS, unsafe_allow_html=True)
    st.markdown(SITE_BACKDROP_HTML, unsafe_allow_html=True)
    _init_state()

    # Widget chrome (matches floating chat in design mock)
    h1, h2 = st.columns([5, 1])
    with h1:
        st.markdown(CHAT_WIDGET_HEADER, unsafe_allow_html=True)
    with h2:
        if st.button("↺", key="reset_chat", help="שיחה חדשה"):
            _reset_session()
            st.rerun()

    st.markdown(
        '<div class="fnx-disclaimer">'
        "<strong>לתשומת לבך:</strong> הצ'אט מבוסס AI. מידע אינפורמטיבי בלבד. "
        "<strong>אין להזין מידע אישי.</strong></div>",
        unsafe_allow_html=True,
    )

    try:
        pipe, cfg = _pipeline()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"שגיאה בטעינת המערכת: {exc}")
        st.stop()

    chunk_count = "—"
    try:
        if cfg.chunks_path.is_file():
            with cfg.chunks_path.open("r", encoding="utf-8") as fh:
                chunk_count = f"{sum(1 for _ in fh):,}"
    except OSError:
        pass

    st.markdown(
        f'<div class="fnx-status-pill">● מאומת מול fnx.co.il • {chunk_count} מקטעים • e5 + BM25</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="fnx-chip-label">נושאים נפוצים:</div>', unsafe_allow_html=True)
    tcols = st.columns(3)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        with tcols[i % 3]:
            if st.button(q, key=f"chip_{i}"):
                st.session_state.pending_question = q
                st.rerun()

    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role, avatar="👤" if role == "user" else "🤖"):
            st.markdown(msg["content"])
            meta = msg.get("meta") or {}
            citations = meta.get("citations") or []
            if citations:
                st.markdown(_source_pills(citations), unsafe_allow_html=True)
            hits = meta.get("hits") or []
            if hits:
                with st.expander("איך מצאתי את זה"):
                    for h in hits:
                        title = h.get("page_title") or h.get("url", "")
                        path = h.get("heading_path") or ""
                        st.markdown(f"**{title}**  \n`{path}`")
                        st.caption(
                            f"`{h.get('chunk_id', '')}` • RRF {h.get('score', 0)}"
                            + (f" • rerank {h['rerank_score']}" if h.get("rerank_score") else "")
                        )
                        st.text(h.get("preview", ""))

    if st.session_state.pending_question:
        prompt = st.session_state.pending_question
        st.session_state.pending_question = None
    else:
        prompt = st.chat_input("שאלו על ביטוח, תביעה, פנסיה או שירות…")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt, "meta": {}})
        with st.spinner("מחפש באתר ומכין תשובה…"):
            try:
                ans = pipe.ask(prompt, history=list(st.session_state.llm_history))
                meta = {
                    "answer_type": ans.answer_type,
                    "citations": list(ans.citations),
                    "hits": _hits_payload(ans.hits),
                    "used_chunk_ids": list(ans.used_chunk_ids),
                }
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ans.text,
                    "meta": meta,
                })
                st.session_state.llm_history.append({"role": "user", "content": prompt})
                st.session_state.llm_history.append({"role": "assistant", "content": ans.text})
            except Exception:
                not_found = cfg.messages.get("not_found") or "לא הצלחתי לענות כרגע. נסו שוב."
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": not_found,
                    "meta": {"error": True, "status": "not_found"},
                })
        st.rerun()


if __name__ == "__main__":
    main()
