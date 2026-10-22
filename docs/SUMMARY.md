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
| LLM | Gemini 3.5 Flash Lite (AI Studio) | תשובה מבוססת קטעים בלבד, temperature 0; fallback `gemini-3.5-flash` |

### השוואת מודלי embedding (מפרט 6.4)

הרצה: `python -m phoenix_rag.cli compare-embeddings`

- **e5-base מקומי**: ~9/10 hit@5 על תת-קבוצה בעברית (`config/probe.yaml`).
- **gemini-embedding-001**: נחסם במכסה יומית ב-free tier במהלך הפיתוח.

## איך נבחר הקטע לתשובה

1. **Sanitize** — `sanitize_question` / `sanitize_history` (אורך, תווי בקרה, סינון jailbreak).
2. **יציאות מוקדמות** — injection → `out_of_scope`; גיבוב / לא מובן → `messages.unintelligible` (`sanitize.is_unintelligible_input`, ללא שכתוב מהיסטוריה).
3. **שכתוב follow-up** — `rewrite_question` + heuristics / thread slots ב-`context.py` (רק כשהקלט מובן ו-on-topic).
4. **גבולות** — חשבון / ייעוץ / off-topic — `boundaries.py` (היסטוריה רק ל-follow-up אמיתי).
5. **אחזור** — היברידי + dedupe URL + סף RRF (**0.0**); demote/filter/promote דפי פעולה.
6. **Cohere Rerank** — עד 40 מועמדים → `top_n: 8`; `drop_weak_hits` אם ציון ה-rerank הטוב ביותר מתחת ל-`min_score` (0.03) → `not_found` בלי Gemini.
7. **Gemini** — JSON + whitelist ציטוטים; claim-SLA → `not_found`; fallback לדף פעולה; אין מקור → `not_found`.

## מגבלות ידועות

- לא ניגשים לאזור אישי, פוליסות ספציפיות, או מידע שלא פורסם באתר.
- דפי iframe/טפסים חיצוניים — לעיתים `too_short` או קישור בלבד.
- NetFree: חסימת Vertex; Gemini API + hf-mirror לעבודה מקומית.
- Latency: p50 ~6.4s, p95 ~24s (ריצת `pre_submit`, 21/10) — תלוי ב-rerank + Gemini.

## רענון אינדקס

```powershell
python -m phoenix_rag.cli crawl          # אינקרמנטלי
python -m phoenix_rag.cli clean
python -m phoenix_rag.cli chunk
python -m phoenix_rag.cli index          # ממשיך מ-cache embeddings
```

## הערכה — before / after

פירוט מלא: `docs/eval_before_after.md`. סט: `eval/questions.jsonl` (35 שאלות ציבוריות + holdout; 37 שורות בקובץ).

| מדד | לפני (eval.yaml 20+10) | after_tuned (15/10) | pre_submit (21/10) |
| --- | --- | --- | --- |
| Pass rate | 30/30 | **30/30** | **35/35** |
| Retrieval hit rate | — | 0.944 | **1.0** |
| Citation correctness | — | 1.0 | **1.0** |
| Correct refusal | 10/10 | 1.0 | **1.0** |
| Hallucination (heuristic) | 0 | 0 | **0** |
| Latency p50 (ms) | — | ~6,276 | **~6,352** |

Artifact עדכני (לפני שינויי `unintelligible` האחרונים): `eval/results/20261021T134725Z_pre_submit.json`. אחרי commit — הרצת `python scripts/eval.py --tag pre_submit` מחדש.

### שאלות holdout (לבדיקה ידנית)

```powershell
python -m phoenix_rag.cli ask "מה זה גרירת משכנתא?" --json
python -m phoenix_rag.cli ask "מה כתובת המייל האישית של מנכ\"ל הפניקס?" --json
```

## ממשק (יום 5)

- **UI מלא** (Tailwind + Lucide, מקור `עיצוב.txt`): `python -m phoenix_rag.cli run` → FastAPI + `app/static/index.html`
- **UI קל (stdlib)**: `python -m phoenix_rag.cli serve`
- API: `POST /api/ask`, **`POST /api/ask/stream`** (SSE `status` + `done` — תשובה מלאה ב-`done`; typewriter ב-`phoenix_chat.js`), `GET /api/meta` (ספירת עמודים/מקטעים)
- לעדכון עיצוב: לערוך `עיצוב.txt` ולהעתיק ל-`app/static/index.html`, לשמור על `phoenix_chat.js`

## Definition of Done (checklist)

- [ ] Clone + `pip install -r requirements.txt` + `.env` עם `GEMINI_API_KEY`
- [ ] אינדקס קיים (`data/chroma`) או הרצת `index`
- [ ] `python -m phoenix_rag.cli scenarios` — A/B/C
- [ ] `python scripts/eval.py --tag pre_submit` — 35/35 (אחרי שינויי unintelligible — eval מחדש)
- [ ] `python tests/test_unintelligible.py` — גיבוב אחרי שרשור לא יורש נזילה
- [ ] 5 שאלות holdout (`h01`–`h05`) — ידנית או `--json`
- [ ] `python -m phoenix_rag.cli run` — צ'אט RTL + פאנל "איך מצאתי את זה"
