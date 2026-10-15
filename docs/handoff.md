# מצב הפרויקט — מסמך מעבר בין שיחות

<!-- Hand-off note for the next chat session: what exists, what is blocked, and what to do next. Update when state changes. -->

עודכן: 15/10/2026 14:20

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

לוג: `logs/eval_20261015T111048Z.log`. בלי אינדוקס מחדש.

שלושת כשלונות הציטוט מ-17/20 תוקנו (דף פעולה במקום FAQ / NONE / how-to-join):

| שאלה | ציטוט אחרי התיקון |
|---|---|
| הארכת ביטוח נסיעות מחו"ל | `/customer-service/travel-insurance/travel_policy_extension/` |
| עדכון פרטים אישיים | `/customer-service/apartment-insurance/selfservice_personal_info_update/` |
| שינוי מסלול השקעה בגמל/השתלמות | `/customer-service/common-actions/investment-track-change/` |

כוונון: `rerank.py` (דף-פעולה ל-Cohere, `drop_weak_hits`, promote לפי תביעה מול `/customer-service/` עמוק), `generate.py` (ייחוס דף פעולה + fallback אם Gemini מחזיר not_found/non-JSON), `pipeline.py`, `config.yaml` (`max_candidates: 32`, מודל `gemini-3.5-flash-lite` כי 2.5-flash פרש למפתחות חדשים). `GEMINI_API_KEY_1..15` נטענים. NetFree חוסם Vertex (`aiplatform.googleapis.com`) — רק AI Studio.

## הצעד הבא (יום 3+)

1. אופציונלי: `compare-embeddings` מול `gemini-embedding-001`.
2. אריזה להגשה (README למשפטים, דמו קצר, רשימת מגבלות). הגמר 28/10.

## פקודות שימושיות

```powershell
python -m phoenix_rag.cli status
python -m phoenix_rag.cli ask "יש לי נזילה בבית, איך מגישים תביעה?"
python -m phoenix_rag.cli chat
python -m phoenix_rag.cli eval [--limit N] [--no-rerank]
python -m phoenix_rag.cli serve [--port 7860]
python -m phoenix_rag.cli probe [-k 12] [--no-hybrid]
```
