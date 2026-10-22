# מצב הפרויקט — מסמך מעבר בין שיחות

<!-- Hand-off note for the next chat session: what exists, what is blocked, and what to do next. Update when state changes. -->

עודכן: 22/10/2026

## הפרויקט

צ'אטבוט RAG בעברית לאתר fnx.co.il (הפניקס ביטוח), לפי מסמך האפיון ב-PDF שבתיקיית הפרויקט (האקתון Open Code AI 2026, גמר 28/10). README בעברית, קוד והערות באנגלית. כלל ריפו: כל קובץ קוד/קונפיג/docs נפתח ב-1–2 שורות הסבר באנגלית.

## יום 1 — הושלם (אומת 14/10)

| שלב | מצב | היכן |
|---|---|---|
| מיפוי אתר | ✅ | `config/seeds.yaml`, `docs/site_map.md` |
| קרולר אינקרמנטלי מנומס | ✅ 1,091 פעילים, 47 failed | `phoenix_rag/crawler/` |
| ניקוי ל-blocks | ✅ 1,091 JSON (887 לא-קצרים) | `phoenix_rag/cleaning/` |
| צ'אנקינג | ✅ 14,370 קטעים | `data/chunks.jsonl` |
| Embedder (e5 מקומי + Gemini עם TPM/retryDelay/מכסה יומית) | ✅ | `phoenix_rag/indexing/embed.py` |
| cache embeddings | ✅ | `data/embeddings_cache.sqlite` |
| **אינדקס מלא** | ✅ Chroma 14,370 + BM25 | `data/chroma`, `data/bm25.pkl` |
| Probe | ✅ היברידי 8/10, סמנטי 8/10 | `config/probe.yaml` |
| compare-embeddings (6.4) | ✅ e5 9/10 @5 על תת-קבוצה; Gemini חסום במכסה יומית | `phoenix_rag/indexing/compare.py` |

מודל ראשי: `models/multilingual-e5-base` (768, `provider: sentence-transformers`).

## יום 2 — שכבת תשובה

| שלב | מצב | היכן |
|---|---|---|
| היברידי → Cohere Rerank → Gemini Flash | ✅ | `phoenix_rag/answering/` |
| ציטוט URL + סירוב `not_found` | ✅ | `generate.py` + `messages` ב-`config.yaml` |
| `COHERE_API_KEY` | ✅ הוחזר ל-`.env` | rerank מדלג בנימוס אם חסום/חסר |
| סט הערכה 20+10 | ✅ ריצה 15/10 14:17: **grounded 20/20**, **unanswerable 10/10** | `config/eval.yaml` / `cli eval` |
| גבולות (אישי / ייעוץ / חשבון / מחוץ לתחום) | ✅ | `answering/boundaries.py` |
| ממשק צ'אט | ✅ CLI `ask`/`chat` + UI `serve` | http://127.0.0.1:7860/ |

## יום 3 — אחזור + תשובה מובנית (end-to-end CLI)

| שלב | מצב | היכן |
|---|---|---|
| סף + דה-דופ URL ב-`search` | ✅ | `retrieve.py` + `pipeline` מעביר מ-config |
| JSON מובנה + `used_chunk_ids` | ✅ | `Answer.to_json()` |
| ולידציית מקורות (אין מקור → not_found) | ✅ | `generate.py` |
| לוג JSONL לכל שאילתה | ✅ | `logging_utils.py` → `logs/queries.jsonl` |
| שכתוב follow-up + גבול אישי גם ב-not_found | ✅ | `pipeline.py` |
| CLI `--json` + `scenarios` A/B/C | ✅ | `python -m phoenix_rag.cli scenarios` |

## יום 4 — סט הערכה, מדדים, כיול

| שלב | מצב | היכן |
|---|---|---|
| 25 שאלות + 5 סמויות (jsonl) | ✅ | `eval/questions.jsonl` |
| `scripts/eval.py` (מדדים + תוצאות) | ✅ | `eval/results/` |
| כיול RRF (retrieval-only) | ✅ סף מוצע ~0.022 — **לא מופעל** | `relevance_threshold: 0.0` |
| ריצה מלאה `after_tuned` | ✅ **30/30**, hallucination 0 | `eval/results/20261015T180807Z_after_tuned.json` |
| כוונון | ✅ EN on-topic, how-to EN, demote business/דירה, c01 wording | `boundaries.py`, `rerank.py`, `generate.py` |

## יום 5 — ממשק FastAPI, אריזה, סיכום

| שלב | מצב | היכן |
|---|---|---|
| UI מלא `עיצוב.txt` + FastAPI | ✅ | `app/static/index.html`, `app/static/phoenix_chat.js`, `app/web.py` |
| `POST /api/ask/stream` (SSE status + done) | ✅ | `app/web.py`; לקוח: typewriter ב-`phoenix_chat.js` |
| `python -m phoenix_rag.cli run` | ✅ | port 8501 (uvicorn) |
| `docs/SUMMARY.md` | ✅ | before/after, ארכיטקטורה, DoD |
| `docs/demo_script.md` | ✅ | תסריט הדגמה + holdout |
| Streamlit legacy | — | `app/streamlit_app.py` (לא ב-`requirements.txt`; UI רשמי = FastAPI) |

## אחרי יום 5 — קשיחות קלט (22/10, לא committed)

| שלב | מצב | היכן |
|---|---|---|
| Sanitize + injection gate | ✅ | `answering/sanitize.py`, `pipeline.py` |
| גיבוב / לא מובן → `messages.unintelligible` | ✅ | `config/config.yaml`, `context.py`, `boundaries.py` |
| בדיקות regression גיבוב | ✅ | `tests/test_unintelligible.py` |
| Eval `pre_submit` 35/35 | ✅ 21/10 | `eval/results/20261021T134725Z_pre_submit.json` — **לפני** שינויי unintelligible |

### לפני הגמר (28/10)

1. אחרי commit של unintelligible: `python scripts/eval.py --tag pre_submit` (לעדכן ציון 35/35)
2. בדיקת clean checkout לפי DoD ב-`docs/SUMMARY.md`
3. חזרה על `docs/demo_script.md` מול `run`
4. **Feature freeze** — רק תיקוני באגים / feedback שופטים

## פקודות שימושיות

```powershell
python -m phoenix_rag.cli status
python -m phoenix_rag.cli ask "יש לי נזילה בבית, איך מגישים תביעה?"
python -m phoenix_rag.cli chat
python -m phoenix_rag.cli eval [--limit N] [--no-rerank]
python -m phoenix_rag.cli run [--port 8501]
python -m phoenix_rag.cli serve [--port 7860]
python -m phoenix_rag.cli scenarios
python -m phoenix_rag.cli probe [-k 12] [--no-hybrid]
```
