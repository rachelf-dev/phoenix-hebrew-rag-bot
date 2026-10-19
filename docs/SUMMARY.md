# Phoenix RAG — סיכום טכני (יום 5)

מסמך זה מסכם איך נאסף התוכן, איך נבחרו קטעים, בחירות ארכיטקטורה, מגבלות, רענון אינדקס, ותוצאות ההערכה.

## איסוף תוכן

- **מקור**: דפים ציבוריים בלבד ב-`fnx.co.il`, לפי `config/seeds.yaml` ו-`docs/site_map.md`.
- **Crawler**: `phoenix_rag/crawler/` — נרמול URL, כיבוד `robots.txt`, השהיה ~1.5 שניות, concurrency 1, User-Agent מזהה, refresh אינקרמנטלי לפי hash/lastmod.
- **תוצאה**: ~1,091 עמודים פעילים ב-`data/pages.sqlite` + HTML ב-`data/raw/` (47 כשלונות — timeouts/5xx).
- **JS**: Nuxt SSR — התוכן ב-`<main id="main">`; **לא נדרש Playwright**.

## ניקוי, chunking, metadata

- **ניקוי** (`phoenix_rag/cleaning/`): הסרת boilerplate; שמירת כותרות, פסקאות, רשימות, טבלאות, FAQ; נרמול עברית.
- **Chunking** (`phoenix_rag/indexing/chunk.py`): heading-first, אחר כך גודל (~300–600 טוקנים); prefix הקשר `כותרת > נתיב`; metadata: `chunk_id`, `url`, `page_title`, `heading_path`, `section_type`, `content_hash`.
- **סטטיסטיקה**: 14,370 קטעים ב-`data/chunks.jsonl`.

## Embeddings ואינדקס

| רכיב | בחירה | נימוק |
| --- | --- | --- |
| Embeddings | `intfloat/multilingual-e5-base` (768, מקומי) | מכסת Gemini free tier (~1,000 טקסטים/יום) לא מאפשרת אינדוקס מלא בזמן האקתון |
| Vector store | Chroma (persistent) | מקומי, metadata לסינון, probe מהיר |
| Lexical | BM25 (`rank_bm25`) | שיפור recall לעברית + מונחי ביטוח |
| אחזור | RRF (semantic + BM25) | `phoenix_rag/indexing/retrieve.py`; עד 2 chunks ל-URL במאגר מועמדים |
| Rerank | Cohere `rerank-multilingual-v3.0` (אופציונלי) | דירוג מחדש לפני LLM; מדלג אם אין מפתח |
| LLM | Gemini Flash (AI Studio) | תשובה מבוססת קטעים בלבד, temperature 0 |

### השוואת מודלי embedding (מפרט 6.4)

הרצה: `python -m phoenix_rag.cli compare-embeddings`

- **e5-base מקומי**: ~9/10 hit@5 על תת-קבוצה בעברית (`config/probe.yaml`).
- **gemini-embedding-001**: נחסם במכסה יומית ב-free tier במהלך הפיתוח.

## איך נבחר הקטע לתשובה

1. שכתוב שאלה (follow-up) לפי היסטוריה — `rewrite_question`.
2. גבולות (חשבון / ייעוץ / off-topic) — `boundaries.py`.
3. אחזור היברידי + dedupe URL + סף RRF (כיום **0.0** — כיול הציע ~0.022 אך לא הפריד שאלות לא-ניתנות).
4. Cohere Rerank → `drop_weak_hits` + promote לדפי פעולה רלוונטיים.
5. Gemini עם פרומפט קשיח + **ולידציה בקוד**: אין מקור בין הקטעים → `not_found`.

## מגבלות ידועות

- לא ניגשים לאזור אישי, פוליסות ספציפיות, או מידע שלא פורסם באתר.
- דפי iframe/טפסים חיצוניים — לעיתים `too_short` או קישור בלבד.
- NetFree: חסימת Vertex; Gemini API + hf-mirror לעבודה מקומית.
- Latency: p50 ~6s, p95 ~14s (ריצת `after_tuned`) — תלוי ב-rerank + Gemini.

## רענון אינדקס

```powershell
python -m phoenix_rag.cli crawl          # אינקרמנטלי
python -m phoenix_rag.cli clean
python -m phoenix_rag.cli chunk
python -m phoenix_rag.cli index          # ממשיך מ-cache embeddings
```

## הערכה — before / after

פירוט מלא: `docs/eval_before_after.md`. סט: `eval/questions.jsonl` (25 + 5 holdout).

| מדד | לפני (eval.yaml 20+10) | after | after_tuned |
| --- | --- | --- | --- |
| Pass rate | 30/30 | 28/30 | **30/30** |
| Retrieval hit rate | — | 0.944 | **0.944** |
| Citation correctness | — | 0.889 | **1.0** |
| Correct refusal | 10/10 | 1.0 | **1.0** |
| Hallucination (heuristic) | 0 | 0 | **0** |
| Latency p50 (ms) | — | ~10,514 | **~6,276** |

Artifacts: `eval/results/20261015T180807Z_after_tuned.json`

### שאלות holdout (לבדיקה ידנית)

```powershell
python -m phoenix_rag.cli ask "מה זה גרירת משכנתא?" --json
python -m phoenix_rag.cli ask "מה כתובת המייל האישית של מנכ\"ל הפניקס?" --json
```

## ממשק (יום 5)

- **UI מלא** (Tailwind + Lucide, מקור `עיצוב.txt`): `python -m phoenix_rag.cli run` → FastAPI + `app/static/index.html`
- **UI קל (stdlib)**: `python -m phoenix_rag.cli serve`
- API: `POST /api/ask` (שאלה + history), `GET /api/meta` (ספירת עמודים/מקטעים)
- לעדכון עיצוב: לערוך `עיצוב.txt` ולהעתיק ל-`app/static/index.html`, לשמור על `phoenix_chat.js`

## Definition of Done (checklist)

- [ ] Clone + `pip install -r requirements.txt` + `.env` עם `GEMINI_API_KEY`
- [ ] אינדקס קיים (`data/chroma`) או הרצת `index`
- [ ] `python -m phoenix_rag.cli scenarios` — A/B/C
- [ ] `python scripts/eval.py --tag after_tuned` — 30/30
- [ ] 5 שאלות holdout (`h01`–`h05`) — ידנית או `--json`
- [ ] `python -m phoenix_rag.cli run` — צ'אט RTL + פאנל "איך מצאתי את זה"
