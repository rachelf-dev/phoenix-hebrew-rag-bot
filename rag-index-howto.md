# אינדקס RAG: בונים פעם אחת, השרת רק טוען - הנחיות עם קוד

## מה הבעיה היום

1. **בכל עלייה ב-Render הקוד בונה את האינדקס מאפס**, כלומר שולח לגוגל 11,590 בקשות embed. זה מה ששורף את המכסה היומית החינמית.
2. **הקובץ שוקל 236MB** כי `gemini-embedding-001` מחזיר 3,072 מספרים לכל chunk. הוא יודע להחזיר 768 (פרמטר `output_dimensionality`), האיכות כמעט זהה, והגודל יורד לכ-35MB.

הפתרון: בניית האינדקס היא הכנה חד-פעמית על המחשב שלך. התוצר מועלה לענן. השרת ב-Render מוריד אותו בעלייה ועונה משם. ההגשה היא הלינק בענן, והמערכת שם עצמאית לגמרי.

---

## שלב 1 - בניית האינדקס פעם אחת, על המחשב שלך

התקנה:

```bash
pip install google-genai numpy
```

קובץ `build_index.py`:

```python
import json, time, numpy as np
from google import genai

KEY = "המפתח שלך"
client = genai.Client(api_key=KEY)

# chunks = רשימת מחרוזות, בדיוק כמו שהסקריפט שלך טוען היום מקובץ המקור
chunks = json.load(open("chunks_source.json", encoding="utf-8"))

vecs = []
BATCH = 50
for i in range(0, len(chunks), BATCH):
    batch = chunks[i:i + BATCH]
    for attempt in range(5):
        try:
            r = client.models.embed_content(
                model="gemini-embedding-001",
                contents=batch,
                config={"task_type": "RETRIEVAL_DOCUMENT", "output_dimensionality": 768},
            )
            vecs.extend(e.values for e in r.embeddings)
            break
        except Exception as err:          # חריגה ממכסה לדקה - מחכים וממשיכים
            print("retry", i, err)
            time.sleep(60)
    print(f"{len(vecs)}/{len(chunks)}")
    time.sleep(15)                        # שומר על המכסה החינמית לדקה

M = np.array(vecs, dtype="float32")
M /= np.linalg.norm(M, axis=1, keepdims=True)   # נרמול - דמיון קוסינוס הופך למכפלה
np.save("index.npy", M)
json.dump(chunks, open("chunks.json", "w", encoding="utf-8"), ensure_ascii=False)
print("done", M.shape)
```

הרצה:

```bash
python build_index.py
```

- במכסה החינמית זה רץ בערך שעה-שעתיים (232 בקשות במקום 11,590). להשאיר ברקע.
- אם המכסה של היום כבר נגמרה, היא מתאפסת בחצות שעון קליפורניה (10 בבוקר שעון ישראל).
- בסוף יש שני קבצים: `index.npy` (כ-35MB) ו-`chunks.json`.
- זה קורה פעם אחת. אם קובץ המקור ישתנה, מריצים שוב.

הוסיפי ל-`.gitignore`:

```
index.npy
chunks.json
```

---

## שלב 2 - להעלות את שני הקבצים לענן, לא ל-GitHub

הדרך הפשוטה: Hugging Face Hub, חינם ובלי מגבלת 100MB.

1. נרשמות ב-huggingface.co.
2. Settings → Access Tokens → יוצרות טוקן עם הרשאת Write.
3. מריצות:

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import HfApi

REPO = "השם-שלך/rag-index"
api = HfApi(token="hf_...")
api.create_repo(REPO, repo_type="dataset", exist_ok=True)
api.upload_file(path_or_fileobj="index.npy", path_in_repo="index.npy", repo_id=REPO, repo_type="dataset")
api.upload_file(path_or_fileobj="chunks.json", path_in_repo="chunks.json", repo_id=REPO, repo_type="dataset")
print("uploaded")
```

תשאירי את הריפו ציבורי, אז ההורדה בשרת לא צריכה טוקן.

חלופה: Google Drive עם קישור ישיר והורדה דרך `gdown`. עובד, אבל פחות יציב.

---

## שלב 3 - בשרת: להוריד ולטעון בעלייה, בלי לבנות כלום

`requirements.txt`:

```
google-genai
numpy
huggingface_hub
```

אפשר למחוק את `chromadb`.

קוד הטעינה (רץ פעם אחת כשהשרת עולה):

```python
import json, numpy as np
from huggingface_hub import hf_hub_download
from google import genai

REPO = "השם-שלך/rag-index"
idx_path = hf_hub_download(REPO, "index.npy", repo_type="dataset")
chunks_path = hf_hub_download(REPO, "chunks.json", repo_type="dataset")

M = np.load(idx_path)                                   # (11590, 768), כ-35MB בזיכרון
chunks = json.load(open(chunks_path, encoding="utf-8"))
client = genai.Client()                                 # קורא GEMINI_API_KEY ממשתני הסביבה
print("index loaded", M.shape)
```

חיפוש בזמן שאלה:

```python
def retrieve(question, k=5):
    r = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config={"task_type": "RETRIEVAL_QUERY", "output_dimensionality": 768},
    )
    q = np.array(r.embeddings[0].values, dtype="float32")
    q /= np.linalg.norm(q)
    sims = M @ q                                        # מילישניות
    top = np.argsort(-sims)[:k]
    return [chunks[i] for i in top]
```

`retrieve` מחזירה את הקטעים הרלוונטיים, ואותם מכניסים לפרומפט של המודל שעונה, בדיוק כמו היום. כל שאלה = בקשת embed אחת, רחוק מכל מכסה.

ב-Render: Environment → להוסיף `GEMINI_API_KEY`. לא צריך דיסק, לא צריך Chroma, ו-512MB מספיקים.

---

## שלב 4 - בדיקה

1. מקומית: `python app.py`, לשאול שאלה, לראות תשובה.
2. push ל-GitHub, Render בונה. בלוג של העלייה צריך להופיע `index loaded (11590, 768)`.
3. לשאול דרך הלינק הציבורי. זה הלינק שמגישים.

---

## שני דגשים לערב הגמר

- Render חינמי נרדם אחרי 15 דקות בלי תנועה, והעלייה הראשונה כוללת הורדה של 35MB. חצי שעה לפני ההצגה לפתוח את הלינק ולשאול שאלה, כדי שיהיה ער.
- להחזיק גם הרצה על הלפטופ כגיבוי, למקרה שהאינטרנט באולם ייפול. ההגשה והשיפוט הם על הלינק בענן.

---

## אם מעדיפים להישאר עם Chroma

אפשר, רק לא לבנות ממנו embeddings. בעלייה טוענים את `index.npy` ו-`chunks.json` ומכניסים ל-collection עם embeddings מוכנים:

```python
import chromadb
coll = chromadb.Client().create_collection("docs")
for i in range(0, len(chunks), 5000):
    coll.add(
        ids=[str(j) for j in range(i, min(i + 5000, len(chunks)))],
        documents=chunks[i:i + 5000],
        embeddings=M[i:i + 5000].tolist(),
    )
```

זה מהיר ובלי API, אבל numpy פשוט יותר ויש פחות מה שייפול.

---

## אם המכסה של גוגל ממשיכה להציק

בונים את האינדקס מקומית עם `intfloat/multilingual-e5-base` על המחשב (CPU, כ-15 דקות, בלי מכסות), ובזמן שאלה מבקשים את ה-embedding של השאלה מ-Hugging Face Inference API לאותו מודל. השרת לא מריץ מודל אף פעם. חשוב: אותו מודל לבנייה ולשאלה, ו-E5 מצפה לקידומת `passage: ` לפני כל קטע ו-`query: ` לפני השאלה.
