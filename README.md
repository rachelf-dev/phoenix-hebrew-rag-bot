# Phoenix RAG – צ'אטבוט עברי מבוסס תוכן האתר fnx.co.il

This README is the Hebrew project guide for Phoenix RAG: a chatbot that answers only from public fnx.co.il pages. It covers status, install, CLI commands, and how crawl / clean / index work.

פרויקט לאתגר הפניקס בהאקתון קוד פתוח AI 2026. הבוט עונה על שאלות לקוח **רק** מתוך התוכן הציבורי של אתר הפניקס, מצרף קישור למקור בכל תשובה, ואומר במפורש "זה לא כתוב באתר" כשאין מידע.

README בעברית; קוד והערות קוד באנגלית. רישיון: MIT (ראו `LICENSE`).

## מצב הפרויקט

| שלב (לפי המפרט, סעיף 10) | מצב |
|---|---|
| 1. מיפוי האתר – אזורים, seed URLs | הושלם: `docs/site_map.md`, `config/seeds.yaml` |
| 2. איסוף ואינדוקס – crawler, ניקוי, chunking, embeddings, אינדקס | **הושלם**: 1,091 עמודים → 14,370 קטעים → Chroma + BM25 (`gemini-embedding-001` @ 768, אינדוקס מקומי + cache) |
| 3. שכבת תשובה | **הושלם**: sanitize → גבולות → היברידי → Cohere Rerank → Gemini 3.5 Flash, ציטוט URL, `not_found` / `unintelligible` |
| 4. סט הערכה | **הושלם**: `eval/questions.jsonl` (יום 4; legacy `config/eval.yaml` 20+10) |
| 5. טיפול בגבולות | **הושלם**: מידע אישי / ייעוץ / חשבון / מחוץ לתחום / injection |
| 6. ממשק צ'אט ואריזה | **הושלם**: FastAPI RTL (`run`) + CLI + `docs/SUMMARY.md` |

## התקנה

דרישות: Python 3.10+.

```bash
pip install -r requirements.txt
copy .env.example .env     # GEMINI_API_KEY (תשובות + embeddings); COHERE_API_KEY ל-Rerank (אופציונלי)
```

**מפתח Gemini** (שכבת תשובה, אינדוקס ושאילתות): [aistudio.google.com/apikey](https://aistudio.google.com/apikey). מפתחות חדשים מתחילים ב-`AQ.` ועובדים מול Gemini API. אם המפתח כבול ל-"Agent Platform / Vertex AI" יש להוסיף `GOOGLE_GENAI_USE_VERTEXAI=1` ל-`.env`.

## פקודות

```bash
# 1. מיפוי: מוריד רק את ה-sitemap ומראה איך seeds.yaml מסווג כל אזור (ללא סריקת עמודים)
python -m phoenix_rag.cli map [--write]

# 2. סריקה (מנומסת: בקשה אחת בכל פעם, 1.5 שניות השהיה, robots.txt, User-Agent מזהה)
python -m phoenix_rag.cli crawl                              # סריקה מלאה / רענון אינקרמנטלי
python -m phoenix_rag.cli crawl --prefix /claims-customer-service/ --limit 20   # ריצת בדיקה
python -m phoenix_rag.cli crawl --force                      # התעלמות מ-lastmod/ETag, הורדה מחדש של הכול

# 3. ניקוי: HTML גולמי -> JSON מובנה (כותרות, פסקאות, רשימות, טבלאות, זוגות שאלה-תשובה, קישורים לטפסים)
python -m phoenix_rag.cli clean [--limit N] [--url URL] [--verbose 800]

# 4. chunking: עמודים נקיים -> data/chunks.jsonl (קטעים עם קידומת הקשר, מטא-דאטה מלא)
python -m phoenix_rag.cli chunk [--verbose 300]

# 5. אינדוקס: Gemini embeddings + Chroma + BM25, ואז probe של 10 שאלות בעברית
python -m phoenix_rag.cli index                      # ריצה מלאה דרך API (מכסה free tier; cache ממשיך אם נעצר)
python -m phoenix_rag.cli index --limit 120 --no-probe   # בדיקת עשן
python -m phoenix_rag.cli index --rechunk            # לבנות מחדש גם את chunks.jsonl

# 6. בדיקות אחזור
python -m phoenix_rag.cli probe [-k 12] [--no-hybrid]          # 10 שאלות smoke מ-config/probe.yaml
python -m phoenix_rag.cli compare-embeddings [--models a b]    # השוואת מודלי embedding על תת-קבוצה בעברית (מפרט 6.4)

# שכבת תשובה (יום 2): היברידי → Cohere Rerank → Gemini Flash
python -m phoenix_rag.cli ask "יש לי נזילה בבית, איך מגישים תביעה?"
python -m phoenix_rag.cli chat
python -m phoenix_rag.cli eval [--limit N] [--no-rerank]
python -m phoenix_rag.cli run [--port 8501]     # UI מלא לפי עיצוב.txt (FastAPI)
python -m phoenix_rag.cli serve [--port 7860]   # UI קל (stdlib), http://127.0.0.1:7860/

# מצב המאגר וריצות אחרונות
python -m phoenix_rag.cli status
```

## מה נבנה ביום 1

### מיפוי (`docs/site_map.md`)

* האתר מוגש ב-Nuxt עם SSR – כל התוכן נמצא ב-HTML בתוך `<main id="main">`. **אין צורך ב-Playwright.**
* ה-sitemap מכיל 1,645 כתובות; אחרי סינון לפי `config/seeds.yaml` נשארות **1,101 כתובות בתחום** (תביעות, שירות, מוצרי ביטוח, פנסיה/גמל/חיסכון). מוחרגים: אתר באנגלית, קשרי משקיעים, ממשל תאגידי, תגיות, קריירה, בית השקעות.
* לכל אזור מוגדר `section_type` ברירת מחדל (claim / service / product / guide / faq / form / terms) ועדיפות סריקה.

### Crawler (`phoenix_rag/crawler/`)

* מונחה sitemap + גילוי קישורים מתוך `<main>` (אותו דומיין, אזורים מותרים בלבד).
* נרמול כתובות: unescape של HTML entities (כתובות בעברית), https + `www`, הסרת query ו-fragment, percent-encoding אחיד, slash בסוף.
* `robots.txt` נאכף עם Protego (תומך ב-wildcards כמו `/*?page=`).
* רענון אינקרמנטלי: עמוד נטען מחדש רק אם `<lastmod>` ב-sitemap השתנה (או דרך ETag / If-Modified-Since כשאין lastmod). אחרי טעינה, hash של הטקסט ב-`<main>` קובע אם התוכן באמת השתנה. בריצה מלאה, עמודים שנעלמו מה-sitemap מסומנים `removed` (כדי שהאינדקס יוכל למחוק את ה-chunks שלהם).
* אחסון: `data/pages.sqlite` (URL, canonical, כותרת, סטטוס HTTP, זמן איסוף, hash, lastmod, ETag, section_type, מצב) + HTML גולמי דחוס ב-`data/raw/<sha1>.html.gz`. כל אירוע (fetched / unchanged / blocked / failed / excluded / removed) נרשם בטבלת `crawl_log` וב-`logs/`.

### ניקוי (`phoenix_rag/cleaning/`)

* עובד ברמת בלוקים של האתר: שומר `intro-area`, בלוקי טקסט, `faq-section` (אקורדיון `collapse-group-item` → זוג שאלה/תשובה שנשמר יחד), טבלאות (→ markdown), רשימות ושלבים ממוספרים. זורק פירורי לחם (נשמרים כמטא-דאטה), בלוק "כל הדרכים לדבר איתנו" שחוזר בכל עמוד, טפסי לידים, גרפיקה, כפילויות mobile/desktop.
* `<p><span class="rte-subtitle-N">` – הדרך של האתר לכתוב כותרות משנה – מזוהה ככותרת, כך שלכל בלוק יש `heading_path` (למשל `נזקי צנרת > מה לעשות אם הנזק מהצנרת שלך?`) שישמש כקידומת הקשר ב-chunking.
* נרמול עברית: NFC, איחוד גרש/גרשיים/מירכאות, הסרת תווי bidi ו-zero-width, ניקוי רווחים.
* הדיסקליימר המשפטי החוזר ("מידע זה הינו כללי…") מוסר מהגוף ונשמר פעם אחת כשדה `disclaimer`.
* קישורים ל-PDF ולטפסים דיגיטליים (`homeclaim.fnx.co.il`, `forms.fnx.co.il`, `natav.net`…) נשמרים כ-`links` עם סוג (pdf / form / internal / external) – תשובות "איפה מגישים" תלויות בהם.
* עמודים עם פחות מ-200 תווים אחרי ניקוי מסומנים `too_short` (למעט FAQ וטפסים).

פלט: `data/clean/<url_id>.json` עם `url`, `display_url`, `title`, `h1`, `breadcrumbs`, `section_type`, `fetched_at`, `content_hash`, `language`, `blocks[]`, `links[]`, `disclaimer`.

### Chunking (`phoenix_rag/indexing/chunk.py`)

* חלוקה לפי מבנה הכותרות קודם (`heading_path` מהניקוי), ואז לפי גודל: יעד 450 טוקנים, תקרה 600, חפיפה 60 (cl100k; ניתן לכוונון ב-`config.yaml → chunking`).
* זוגות שאלה-תשובה, טבלאות ושלבים ממוספרים נשארים אטומיים כשהם נכנסים בתקרה; שאלה ארוכה מתפצלת כך שהשאלה חוזרת בכל חלק.
* כל קטע נפתח בקידומת הקשר `כותרת: … \ נתיב: …` (הטקסט ל-embedding), ובנפרד נשמר `original_text` בדיוק כפי שמופיע באתר (לציטוט).
* מטא-דאטה לכל קטע (מפרט 6.3): `chunk_id`, `url`, `display_url`, `page_title`, `heading_path`, `fetch_date`, `content_hash`, `section_type`, `language`, `token_count`, `block_types`, `form_links` (קישורי טפסים/PDF מהעמוד – חשובים ל"איפה מגישים").
* תוצאה: 887 עמודים לא-קצרים → **14,370 קטעים**, ממוצע 380 טוקנים. אינדקס מאוכלס: Chroma 14,370 + BM25.

### Embeddings ואינדקס (`phoenix_rag/indexing/`)

* **מודל ראשי**: `gemini-embedding-001` (`provider: google`, 768 ממדים MRL, נרמול L2) עם `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`. השרת לא טוען Torch או משקולות מקומיות — רק קריאת embedding לשאילתה בזמן ריצה; האינדקס (`data/chroma`) נבנה מראש (מקומית או עם billing).
* **קצב ומכסות**: `embed.py` שומר על תקציב טוקנים לדקה, מגביל גודל בקשה, מכבד `retryDelay` ב-429, ומזהה מכסה יומית (free tier ~**1,000 טקסטים ליום למודל**) כדי לעצור בלי לאבד את ה-cache.
* **Chroma** (`data/chroma`, cosine) – מקומי, ללא שרת, עם מטא-דאטה לסינון. שם המודל נשמר באוסף; החלפת מודל מחייבת אינדוקס מחדש.
* **BM25** (`data/bm25.pkl`) על טוקניזציה עברית קלה (`hebrew_lex.py`): לכל מילה עם קידומת ה/ו/ב/ל/מ/כ/ש נוסף גם הגזע. מיזוג עם האחזור הסמנטי ב-**RRF** (`retrieve.py`).
* **המשכיות**: כל embedding נשמר ב-`data/embeddings_cache.sqlite` (מפתח: מודל + ממד + hash של הטקסט). ריצה שנעצרה ממשיכה מאותה נקודה.
* **רענון**: `crawl` מסמן עמודים שנעלמו כ-`removed`; `clean` מוחק JSON של עמודים שאינם פעילים; `index --rechunk` בונה את האוסף מחדש. סדר: `crawl → clean → index --rechunk`.
* **Probe** (`config/probe.yaml`, 10 שאלות, hit = כתובת-היעד מופיעה ב-top-12): היברידי **8/10**, סמנטי בלבד **8/10**. שתי ה-MISS: (1) "נזילה בבית" לא מדרגת את עמוד *נזקי צנרת* (האתר כותב צנרת/מים; Cohere Rerank ביום 2 אמור לעזור); (2) "תביעה בביטוח מחלות קשות" מחזירה את עמוד התביעה `claims-serious-illness` במקום ה-FAQ שסומן כ-expect — תוצאה נכונה יותר ללקוח.

### השוואת מודלי embedding (מפרט 6.4)

| מועמד | סביבה | hit@5 על תת-קבוצה | הערה |
|---|---|---|---|
| `multilingual-e5-base` (מקומי, היסטורי) | CPU | **9/10**, MRR 0.850 | הוחלף ב-Gemini לפריסה רזה |
| `gemini-embedding-001` | Gemini API | ראו `compare-embeddings` | מודל ראשי נוכחי |

ההשוואה: `python -m phoenix_rag.cli compare-embeddings --models gemini-embedding-001 --distractors 400`. תת-הקבוצה כוללת את כל הקטעים של עמודי ה-expect ועוד 400 מסיחים.

## שכבת התשובה וההערכה (יום 2)

לפי ארגז הכלים של ההאקתון (חלק הפניקס), לא לפי APIs בתשלום:

* **קלט**: `sanitize_question` / `sanitize_history` (אורך מקסימלי, תווי בקרה, סינון jailbreak); **injection** → `out_of_scope`; **גיבוב / לא מובן** → `messages.unintelligible` (בלי שכתוב follow-up מהיסטוריה).
* **אחזור**: היברידי (וקטור + BM25), ואז **Cohere Rerank** (`rerank-multilingual-v3.0`) על עד 40 מועמדים — מצמצם ל-`top_n: 8` לפני Gemini. אם ציון ה-rerank הטוב ביותר מתחת ל-`min_score` (0.03) → `not_found` בלי LLM. בלי מפתח / חסימת NetFree: ממשיכים עם דירוג היברידי.
* **LLM**: **Gemini 3.5 Flash Lite** (`gemini-3.5-flash-lite`; נפילה ל-`gemini-3.5-flash`). מפתח AI Studio: `GEMINI_API_KEY`, אופציונלי `GEMINI_API_KEY_1`…`_15` או `GEMINI_API_KEYS` (מופרד בפסיקים) — ב-429 מעבר למפתח הבא. לענות רק מהקטעים שאוחזרו; אחרת `not_found`. ציון `URL` בכל תשובה.
* **גבולות**: שאלת חשבון/תביעה אישית → `no_account_access`; "הפוליסה שלי" / ייעוץ אישי → תשובה כללית מהאתר + הודעת גבול; נושא לא קשור לאתר → `out_of_scope`.
* **סט הערכה**: מדד ההצלחה הוא "לא ממציא". סט ראשי: **`eval/questions.jsonl`** (35 שאלות ציבוריות + holdout; ראו `docs/eval_before_after.md`). ריצה מלאה: `python scripts/eval.py --tag pre_submit` — **35/35** ב-21/10/2026 (`eval/results/20261021T134725Z_pre_submit.json`) *לפני* שינויי `unintelligible` האחרונים; אחרי commit מומלץ eval מחדש. Legacy: `config/eval.yaml` (20+10), `python -m phoenix_rag.cli eval`.
* **ממשק**: `run` — FastAPI + `app/static/index.html` + `phoenix_chat.js` (`POST /api/ask`, **`POST /api/ask/stream`** — SSE `status` + `done`; typewriter בצד לקוח); `serve` — UI קל; `ask` / `chat` בטרמינל.
* **סיכום והדגמה**: `docs/SUMMARY.md`, `docs/demo_script.md`, `docs/eval_before_after.md`.

## הערות תפעול

* **קצב הורדה**: עמוד באתר שוקל ~450KB (רובו תפריטים) ומוגש לא דחוס. בחיבור שנבדק כל עמוד ארך 9–30 שניות, כך שסריקה מלאה של ~1,100 עמודים עשויה לקחת כמה שעות. הריצה ניתנת לעצירה והמשך – ריצה חוזרת מדלגת על כל מה שכבר נאסף ולא השתנה.
* `data/` ו-`logs/` לא נכנסים ל-git.
* **סינון NetFree (סביבת הפיתוח)**: `aiplatform.googleapis.com` (Vertex) חסום; `generativelanguage.googleapis.com` (Gemini API) **פתוח**.
* **Quota של Gemini**: חוץ מ-~30K טוקנים/דקה, ה-free tier מגביל ל-1,000 בקשות embedding ביום למודל (כל טקסט ב-batch = בקשה). האינדוקס המלא רץ מחוץ לשרת הפרוד; `index` חוזר ממשיך מה-cache אחרי מכסה יומית.
* **פערי תוכן ידועים**: ~47 עמודים נכשלו בסריקה (שגיאות שרת/timeout, ראו `status`), וכמה עשרות עמודים הם "iframe בלבד" (טפסים/מחשבונים חיצוניים) ומסומנים `too_short`.

## מבנה

```
config/config.yaml      הגדרות: סריקה, ניקוי, chunking, embeddings, אחזור (כולל Cohere Rerank), Gemini Flash, הודעות קבועות
eval/questions.jsonl    סט הערכה ראשי (יום 4); config/eval.yaml — legacy 20+10
config/seeds.yaml       אזורים נכללים/מוחרגים + section_type
config/probe.yaml       10 שאלות smoke לאחזור (עמוד צפוי לכל שאלה)
docs/site_map.md        מפת האתר ונימוקים (שלב 1)
docs/SUMMARY.md         סיכום טכני, הערכה, מגבלות (יום 5)
docs/demo_script.md     תסריט הדגמה לגמר
docs/handoff.md         מצב הפרויקט ומסמך מעבר בין סשנים
app/static/index.html   UI מלא (מקור: עיצוב.txt)
app/web.py              FastAPI + /api/ask + /api/ask/stream
phoenix_rag/cli.py      map | crawl | clean | chunk | index | probe | compare-embeddings | ask | chat | eval | run | serve | inspect | status
phoenix_rag/crawler/    urls.py (נרמול+סיווג), robots.py, store.py (SQLite), crawl.py
phoenix_rag/cleaning/   hebrew.py (נרמול טקסט), clean.py (HTML -> בלוקים)
phoenix_rag/indexing/   chunk.py, embed.py (Gemini + Chroma EF), embed_cache.py, store.py, retrieve.py, compare.py
phoenix_rag/answering/  pipeline.py, sanitize.py, context.py, rerank.py, generate.py, boundaries.py, eval_run.py, ui.py
scripts/                check_deploy_ready.py — בדיקות לפני Docker/Render
data/                   pages.sqlite, raw/, clean/, chunks.jsonl, embeddings_cache.sqlite, chroma/, bm25.pkl  (לא ב-git)
```
