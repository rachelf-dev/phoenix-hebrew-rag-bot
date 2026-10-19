# Static fnx.co.il marketing shell (HTML) shown behind the embedded chat widget.

SITE_BACKDROP_HTML = """
<div class="fnx-site-backdrop" aria-hidden="true">
  <header class="fnx-site-header">
    <div class="fnx-topbar">
      <span>קריירה</span><span>פורטל סוכנים</span><span>מוקד: 5040*</span>
    </div>
    <div class="fnx-navbar">
      <div class="fnx-brand">
        <span class="fnx-logo" aria-hidden="true"></span>
        <span class="fnx-brand-name">הפניקס</span>
      </div>
      <nav class="fnx-nav-links">
        <span>ביטוחים</span><span>פנסיה, גמל וחסכון</span><span>השקעות</span>
        <span>אשראי</span><span>תביעות</span><span>שירות ופעולות</span>
      </nav>
      <span class="fnx-personal-pill">לאזור האישי</span>
    </div>
  </header>

  <section class="fnx-hero">
    <div class="fnx-hero-inner">
      <div class="fnx-hero-copy">
        <span class="fnx-hero-tag">האפליקציה החדשה של הפניקס לשליטה מלאה</span>
        <h2>אפליקציית הפניקס<br><em>כי מגיע לכם להיות בשליטה</em></h2>
        <p>בענייני הביטוח, ההשקעות והאשראי שלכם</p>
        <div class="fnx-hero-cta">
          <span class="fnx-btn-primary">להורדת האפליקציה</span>
          <span class="fnx-btn-ghost">✦ שאלות? שאל את ה-AI</span>
        </div>
      </div>
      <div class="fnx-hero-visual">
        <div class="fnx-qr">QR</div>
        <div class="fnx-phone"></div>
      </div>
    </div>
  </section>

  <section class="fnx-cards">
    <div class="fnx-card"><strong>ביטוח רכב smart</strong><p>פוליסה דיגיטלית, שירותי דרך וגרירה.</p></div>
    <div class="fnx-card"><strong>ביטוח דירה ומבנה</strong><p>תביעות נזקי צנרת בדיגיטל וליווי מקצועי.</p></div>
    <div class="fnx-card"><strong>פנסיה והשתלמות</strong><p>חיסכון פנסיוני עם שקיפות מלאה.</p></div>
    <div class="fnx-card"><strong>ביטוח בריאות וחיים</strong><p>כיסוי רפואי רחב והחזרים דיגיטליים.</p></div>
  </section>
</div>
"""

CHAT_WIDGET_HEADER = """
<div class="fnx-widget-header">
  <div class="fnx-widget-id">
    <div class="fnx-widget-icon">✦</div>
    <div>
      <div class="fnx-widget-title">עוזר השירות הפניקס <span class="fnx-badge">RAG AI</span></div>
      <div class="fnx-widget-sub">עונה על בסיס מידע רשמי מ-fnx.co.il</div>
    </div>
  </div>
</div>
"""
