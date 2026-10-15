# מפת האתר fnx.co.il – אזורים רלוונטיים ללקוח (שלב 1)

This page maps fnx.co.il for the crawler: which URL prefixes are in scope, which are excluded, and why. Counts come from the sitemap of 2026-10-11.

נבדק ב-11/10/2026 מתוך `https://www.fnx.co.il/sitemap.xml` (1,645 כתובות) ומתוך `robots.txt`.

## ממצאים טכניים

| נושא | ממצא | השלכה |
|---|---|---|
| רינדור | Nuxt עם SSR (`data-n-head="ssr"`): כל התוכן נמצא ב-HTML שמוחזר מהשרת, בתוך `<main id="main">` | אין צורך ב-Playwright. מספיק HTTP client רגיל |
| מבנה עמוד | `nav.breadcrumbs` ואחריו בלוקים `div.section-item` לפי סוג: `intro-area`, `typography-section` (טקסט חופשי), `faq-section` (אקורדיון `collapse-group-item` עם `.question` / `.answer`), `contacts-section` (חוזר בכל עמוד), `locating-service-providers`, `service-narrow`, `category-card` | הניקוי עובד ברמת בלוק: שומרים תוכן, זורקים כרום |
| גודל עמוד | ~450KB HTML, מתוכם ~20KB ב-`<main>`. השאר תפריטים ו-JS | hash לזיהוי שינויים מחושב על `<main>` בלבד |
| robots.txt | חוסם `?page=`, `?tag=`, `?domain=`, `?staff=`, `/more-information/`, מספר דפי מכירה. מצביע על ה-sitemap | מסירים query strings לגמרי; מכבדים את כל ה-Disallow דרך parser שתומך ב-wildcards (protego) |
| כתובות בעברית | חלק מהכתובות ב-sitemap מקודדות כ-HTML entities (`&#x5E7;…`) | `html.unescape` ואז percent-encoding אחיד בנרמול |
| canonical | קיים בכל עמוד (`link[rel=canonical]`) | נשמר ומשמש לדה-דופליקציה |
| PDF | קישורים לפוליסות/טפסים ב-`digital-content.fnx.co.il` וב-`/sites/docs/` | נשמרים כמטא-דאטה של העמוד (לא נסרקים ביום 1) |

## אזורים שנכללים (לפי עדיפות)

| עדיפות | אזור (prefix) | עמודים | section_type | למה זה חשוב ללקוח |
|---|---|---|---|---|
| 1 | `/claims-customer-service/` | 130 | claim | **הכוונה המרכזית של לקוחות**: איך מגישים תביעה לפי אירוע (נזקי צנרת, תאונה, ניתוח, מזוודה שאבדה…), מה לצרף, לאן שולחים, מה קורה אחר כך. כולל `claims_settlement` (הסדרי תביעות) ו-`pension-benefits` |
| 1 | `/customer-service/` | 81 (−16 תנאי שימוש) | service | פעולות שירות: חידוש, ביטול, העתק פוליסה, תשלומים, שינוי פרטים, הארכת ביטוח נסיעות, דרכי התקשרות, אמנת שירות, אפליקציה |
| 1 | `/apartment-insurance/` | 66 | product / guide | ביטוח דירה: מבנה, תכולה, צד ג׳, מדריכים (נזקי מים, רעידת אדמה, דירה שכורה…) – תרחיש A בהנחיות |
| 1 | `/car-insurance/` | 167 | product / guide | מקיף, חובה, צד ג׳, נהג צעיר, רכב חשמלי, מדריכים |
| 1 | `/health-insurance/` | 57 | product / faq | ניתוחים, מחלות קשות, אמבולטורי, סיעודי, **FAQ** |
| 1 | `/travel-insurance/` | 98 | product / guide | כיסויים, הרחבות, יעדים, מדריכים |
| 1 | `/life-insurance/` | 13 | product | ריסק, הכנסה למשפחה, משיכת כספים |
| 2 | `/mortgage-insurance/` | 23 | product / guide | ביטוח חיים ומבנה למשכנתא |
| 2 | `/disability-and-incapacity-to-work/` | 8 | product | אובדן כושר עבודה, מטריה ביטוחית |
| 2 | `/pets-insurance/` | 21 | product / guide | כלב/חתול + מדריכים |
| 2 | `/pension-savings/` | 180 | product | פנסיה, ביטוח מנהלים, מסלולי השקעה, הלוואות כנגד חיסכון |
| 2 | `/gemel/` | 131 | product | קופות גמל, תיקון 190 |
| 2 | `/ishtalmutfund/` | 27 | product | קרנות השתלמות |
| 2 | `/savings/` | 31 | product | פוליסות חיסכון/השקעה |
| 2 | `/pension_retirement/` | 8 | product | קצבה מיידית, משכנתא הפוכה |
| 2 | `/takanottashlum/` | 13 | terms / form | תקנות תשלומים – שלבים וטפסים |
| 2 | `/ituran/` | 11 | service | איתור נותני שירות (שרברבים, מוסכים, שמאים, רופאים), טפסים ופוליסות |
| 3 | `/business-insurance/`, `/other-insurances/`, `/truck-insurance/`, `/marine_aviation_insurance/`, `/loans/` | ~45 | product | מוצרים לעסקים / נישה – נכללים כדי שהבוט יוכל לענות "מה כתוב באתר" ולא "לא קיים" |
| 3 | `/calculators/`, `/articles/`, `/` | 6 | guide / general | הסברים טקסטואליים; דף הבית לגילוי קישורים |

סה"כ משוער לסריקה: **~1,050 עמודים**. בקצב 1.5 שניות לבקשה: כ-30 דקות לסריקה מלאה.

## אזורים שמוחרגים

| אזור | עמודים | סיבה |
|---|---|---|
| `/en/` | 185 | אתר באנגלית; הבוט עונה בעברית מתוכן עברי |
| `/investor-relations/` | 241 | קשרי משקיעים – לא רלוונטי ללקוח |
| `/fnx-insurance-ltd/` | 38 | ממשל תאגידי, דוחות, דירקטוריון |
| `/tags/` | 33 | עמודי תגיות – רשימות כפולות ללא תוכן עצמאי |
| `/career/`, `/esop/`, `/agents-promotions/` | 13 | לא פונה ללקוח פרטי |
| `/alternativeinvestments/`, `/haskaot-mitkadmot/`, `/trading/`, `/portfoliomanagement/`, `/fnxw/` | 11 | בית השקעות / ניהול הון – מחוץ לגבולות האתגר |
| `/business-financing/`, `/projectvalidation/`, `/guarantee-validation/` | 5 | מימון עסקי B2B |
| `/customer-service/terms-of-use/` | 16 | תנאי שימוש משפטיים לאפליקציות; ניתן להחזיר בהמשך |
| `/more-information/` ועוד | – | חסום ב-robots.txt |
| `/404/`, `/sitemap/`, `/ksmc-1/` | 3 | טכני / נחיתה שיווקית |

## כללי החרגה בתוך האזורים הנכללים

* כל כתובת עם query string (robots חוסם `?page=` וכו׳; האתר לא משתמש ב-query לתוכן).
* עמודים שאחרי הניקוי נשארים עם פחות מ-200 תווים (למעט FAQ וטפסים).
* עמודים עם `canonical` שמצביע לכתובת אחרת שכבר נסרקה (דה-דופליקציה).
* קישורים חיצוניים (`homeclaim.fnx.co.il`, `digital-content.fnx.co.il`, `natav.net`…) לא נסרקים אלא נשמרים כמטא-דאטה "קישורים לטפסים/PDF" של העמוד.

## דוגמאות לעמודי יעד לתרחישי הקבלה

| תרחיש | עמוד |
|---|---|
| A – "יש לי נזילה בבית, איך מגישים תביעה?" | `/claims-customer-service/claims-appartment-insurance/pipeline-damage/` (+ `damage-to-the-apartment/`, `apartment-information/water-plumbing-damages/`) |
| B – "תוך כמה זמן מקבלים תשובה?" | אותו עמוד; מכיל "תוך יום עסקים יחזור נציג…" אבל **לא** זמן תשובה לתביעה → הבוט צריך להגיד שזה לא כתוב |
| C – "האם הפוליסה שלי מכסה את זה?" | גבול אישי; מידע כללי מ-`/apartment-insurance/structure/` וכו׳ |
