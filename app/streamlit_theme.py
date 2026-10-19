# Phoenix RAG Streamlit theme: fnx.co.il site mock + floating chat widget (from עיצוב.txt).

PHOENIX_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Assistant:wght@400;500;600;700;800&family=Rubik:wght@500;600;700;800&display=swap');

html, body, [class*="css"] {
  direction: rtl;
  text-align: right;
  font-family: 'Assistant', sans-serif;
}

header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"],
section[data-testid="stSidebar"] {
  display: none !important;
}

.stApp {
  background: #f1f5f9 !important;
}

/* ---- Site backdrop (mock fnx.co.il) ---- */
.fnx-site-backdrop {
  position: fixed;
  inset: 0;
  overflow-y: auto;
  z-index: 0;
  pointer-events: none;
  background: #f8fafc;
}
.fnx-site-header {
  background: #0c172d;
  color: #fff;
  border-bottom: 1px solid #18284b;
  box-shadow: 0 4px 14px rgba(0,0,0,.15);
}
.fnx-topbar {
  display: flex;
  gap: 1.5rem;
  padding: 0.35rem 2rem;
  font-size: 0.7rem;
  color: #94a3b8;
  border-bottom: 1px solid rgba(255,255,255,.08);
}
.fnx-navbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.85rem 2rem;
  gap: 1rem;
  flex-wrap: wrap;
}
.fnx-brand { display: flex; align-items: center; gap: 0.5rem; }
.fnx-logo {
  width: 32px; height: 32px; border-radius: 50%;
  background: linear-gradient(135deg, #ff7b39, #ff551f 60%, #e13b06);
  display: inline-block;
}
.fnx-brand-name {
  font-family: 'Rubik', sans-serif;
  font-size: 1.75rem;
  font-weight: 800;
}
.fnx-nav-links {
  display: flex;
  gap: 1.25rem;
  font-weight: 600;
  font-size: 0.9rem;
  color: rgba(255,255,255,.92);
  flex-wrap: wrap;
}
.fnx-personal-pill {
  border: 1px solid rgba(255,255,255,.4);
  border-radius: 999px;
  padding: 0.45rem 1.25rem;
  font-size: 0.85rem;
  font-weight: 600;
}

.fnx-hero {
  background: linear-gradient(180deg, #0c172d 0%, #0f1d38 45%, #0c172d 100%);
  color: #fff;
  padding: 2.5rem 2rem 4rem;
  position: relative;
  overflow: hidden;
}
.fnx-hero::before {
  content: "";
  position: absolute;
  right: -80px; top: 20px;
  width: 420px; height: 420px;
  border: 2px dashed rgba(255,85,31,.35);
  transform: rotate(12deg);
  border-radius: 8px;
}
.fnx-hero-inner {
  max-width: 1100px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: 1.2fr 0.8fr;
  gap: 2rem;
  align-items: center;
  position: relative;
  z-index: 1;
}
.fnx-hero-tag {
  display: inline-block;
  background: rgba(255,255,255,.1);
  border: 1px solid rgba(255,255,255,.12);
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  font-size: 0.75rem;
  color: #fdba74;
  margin-bottom: 0.75rem;
}
.fnx-hero h2 {
  font-family: 'Rubik', sans-serif;
  font-size: clamp(1.75rem, 3vw, 2.75rem);
  line-height: 1.15;
  margin: 0 0 0.75rem;
  font-weight: 800;
}
.fnx-hero h2 em {
  font-style: normal;
  color: #e2e8f0;
  font-weight: 700;
}
.fnx-hero-copy p { color: #cbd5e1; font-size: 1.15rem; margin: 0 0 1.25rem; }
.fnx-hero-cta { display: flex; flex-wrap: wrap; gap: 0.75rem; }
.fnx-btn-primary {
  background: #ff551f;
  color: #fff;
  font-weight: 700;
  padding: 0.75rem 1.75rem;
  border-radius: 999px;
  box-shadow: 0 10px 25px -5px rgba(255,85,31,.45);
}
.fnx-btn-ghost {
  border: 1px solid rgba(255,255,255,.25);
  background: rgba(255,255,255,.08);
  padding: 0.75rem 1.25rem;
  border-radius: 999px;
  font-weight: 600;
}
.fnx-hero-visual {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
}
.fnx-qr {
  width: 88px; height: 88px;
  background: rgba(13,27,51,.9);
  border: 2px solid rgba(255,85,31,.7);
  border-radius: 16px;
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 0.75rem;
}
.fnx-phone {
  width: 160px; height: 300px;
  border-radius: 28px;
  border: 4px solid #334155;
  background: linear-gradient(180deg, #f4f6fa 0%, #e2e8f0 100%);
  box-shadow: 0 20px 40px rgba(0,0,0,.35);
}

.fnx-cards {
  max-width: 1100px;
  margin: -2rem auto 3rem;
  padding: 0 2rem;
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  position: relative;
  z-index: 2;
}
.fnx-card {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  padding: 1.25rem;
  box-shadow: 0 8px 24px rgba(15,23,42,.08);
}
.fnx-card strong {
  font-family: 'Rubik', sans-serif;
  display: block;
  margin-bottom: 0.35rem;
  color: #0f172a;
}
.fnx-card p { margin: 0; font-size: 0.75rem; color: #64748b; line-height: 1.45; }

@media (max-width: 900px) {
  .fnx-hero-inner { grid-template-columns: 1fr; }
  .fnx-hero-visual { display: none; }
  .fnx-cards { grid-template-columns: 1fr 1fr; }
  .fnx-nav-links { display: none; }
}

/* ---- Floating chat widget (Streamlit main panel) ---- */
section.main {
  position: fixed !important;
  left: 24px !important;
  bottom: 24px !important;
  top: auto !important;
  right: auto !important;
  width: min(425px, calc(100vw - 32px)) !important;
  max-height: min(640px, calc(100vh - 40px)) !important;
  z-index: 200 !important;
  background: #fff !important;
  border-radius: 18px !important;
  border: 1px solid rgba(226,232,240,.95) !important;
  box-shadow: 0 24px 50px -12px rgba(11,22,44,.45), 0 0 25px rgba(255,85,31,.15) !important;
  overflow: hidden !important;
}

section.main > div.block-container {
  padding: 0 !important;
  max-width: none !important;
  height: 100%;
  max-height: min(640px, calc(100vh - 40px));
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

section.main [data-testid="stVerticalBlock"]:first-child {
  flex: 1;
  overflow-y: auto;
  padding: 0 0.65rem 0.25rem;
}

.fnx-widget-header {
  background: #0c172d;
  color: #fff;
  padding: 0.85rem 1rem;
  border-bottom: 1px solid #1d325c;
  margin: 0 -0.65rem 0.5rem;
  padding-left: 1rem;
  padding-right: 1rem;
}
.fnx-widget-id { display: flex; align-items: center; gap: 0.65rem; }
.fnx-widget-icon {
  width: 40px; height: 40px; border-radius: 12px;
  background: linear-gradient(135deg, #ff551f, #ff7b39);
  display: flex; align-items: center; justify-content: center;
  font-weight: 800; color: #fff;
  box-shadow: 0 4px 10px rgba(255,85,31,.35);
}
.fnx-widget-title {
  font-family: 'Rubik', sans-serif;
  font-weight: 700;
  font-size: 0.95rem;
}
.fnx-widget-sub { font-size: 0.72rem; color: #94a3b8; margin-top: 2px; }
.fnx-badge {
  display: inline-block;
  background: rgba(255,85,31,.2);
  color: #ffb899;
  font-size: 0.58rem;
  font-weight: 800;
  padding: 1px 6px;
  border-radius: 4px;
  border: 1px solid rgba(255,85,31,.4);
  vertical-align: middle;
  margin-right: 4px;
}

.fnx-disclaimer {
  background: #fffbeb;
  border-bottom: 1px solid #fde68a;
  color: #78350f;
  padding: 0.55rem 0.85rem;
  font-size: 0.68rem;
  line-height: 1.4;
  margin: 0 -0.65rem 0.5rem;
}

.fnx-status-pill {
  text-align: center;
  font-size: 0.62rem;
  color: #64748b;
  background: #f1f5f9;
  border-radius: 999px;
  padding: 0.25rem 0.65rem;
  margin: 0.25rem 0 0.5rem;
  display: inline-block;
  width: 100%;
  box-sizing: border-box;
}

.fnx-toolbar {
  display: flex;
  gap: 0.35rem;
  flex-wrap: wrap;
  margin-bottom: 0.35rem;
}
.fnx-toolbar .stButton > button {
  font-size: 0.68rem !important;
  padding: 0.35rem 0.5rem !important;
  min-height: 0 !important;
  height: auto !important;
  background: #f1f5f9 !important;
  color: #334155 !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 8px !important;
}
.fnx-toolbar .stButton > button:hover {
  background: #fff3ed !important;
  color: #ff551f !important;
  border-color: rgba(255,85,31,.35) !important;
}
section.main div[data-testid="column"]:nth-of-type(2) div[data-testid="stButton"] > button {
  margin-top: 0.55rem;
  background: rgba(12,23,45,.06) !important;
  color: #64748b !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 10px !important;
  min-width: 2.25rem !important;
  padding: 0.35rem !important;
}
section.main div[data-testid="column"]:nth-of-type(1) .fnx-widget-header {
  margin-right: -0.5rem;
}
.fnx-chip-label {
  font-size: 0.68rem;
  font-weight: 600;
  color: #64748b;
  margin: 0.15rem 0 0.25rem;
}

div[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 0.15rem 0 !important;
  box-shadow: none !important;
}
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  border-top-right-radius: 4px;
  padding: 0.65rem 0.85rem;
  box-shadow: 0 4px 14px rgba(0,0,0,.05);
  font-size: 0.82rem;
  line-height: 1.5;
}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
  background: #0c172d !important;
  color: #fff !important;
  border: none;
  border-top-left-radius: 4px;
  border-top-right-radius: 16px;
}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) li {
  color: #fff !important;
}

.stChatInputContainer {
  border-top: 1px solid #e2e8f0 !important;
  background: #fff !important;
  padding: 0.5rem 0.65rem 0.65rem !important;
  margin: 0 -0.65rem !important;
}
.stChatInputContainer textarea {
  font-size: 0.8rem !important;
  border-radius: 12px !important;
}
.stChatInputContainer button {
  background: #ff551f !important;
  border-color: #ff551f !important;
}

.fnx-source-link a {
  color: #ff551f !important;
  font-weight: 600;
  font-size: 0.75rem;
  text-decoration: none;
}
.fnx-source-link a:hover { text-decoration: underline; }

.fnx-source-pill {
  display: block;
  margin-top: 0.5rem;
  padding: 0.45rem 0.65rem;
  background: #fff7ed;
  border: 1px solid #fed7aa;
  border-radius: 12px;
  font-size: 0.72rem;
}

@media (max-width: 640px) {
  section.main {
    left: 0 !important;
    bottom: 0 !important;
    width: 100% !important;
    max-height: 92vh !important;
    border-radius: 18px 18px 0 0 !important;
  }
  .fnx-cards { grid-template-columns: 1fr; }
}
</style>
"""
