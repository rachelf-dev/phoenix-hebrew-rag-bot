"""Tiny local Hebrew chat UI (stdlib http.server). Opens at http://127.0.0.1:7860/."""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from ..config import Config
from .pipeline import Pipeline

log = logging.getLogger(__name__)

HTML = """<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>עוזר הפניקס — תוכן האתר</title>
<style>
  :root { --red:#c8102e; --ink:#1b1b1b; --muted:#5c5c5c; --bg:#f6f4f1; --card:#fff; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: "Segoe UI", Arial, sans-serif; background:var(--bg); color:var(--ink); }
  header { background:var(--red); color:#fff; padding:16px 20px; }
  header h1 { margin:0; font-size:1.2rem; font-weight:600; }
  header p { margin:6px 0 0; font-size:.85rem; opacity:.92; }
  main { max-width:760px; margin:0 auto; padding:16px; display:flex; flex-direction:column; height:calc(100vh - 72px); }
  #log { flex:1; overflow:auto; display:flex; flex-direction:column; gap:12px; padding-bottom:12px; }
  .msg { background:var(--card); border-radius:12px; padding:12px 14px; box-shadow:0 1px 3px rgba(0,0,0,.06); max-width:92%; white-space:pre-wrap; line-height:1.5; }
  .msg.user { align-self:flex-start; background:#fde8ec; }
  .msg.bot { align-self:flex-end; }
  .src { margin-top:8px; font-size:.85rem; }
  .src a { color:var(--red); }
  .disc { color:var(--muted); font-size:.75rem; margin-top:8px; }
  form { display:flex; gap:8px; }
  input[type=text] { flex:1; padding:12px 14px; border:1px solid #ddd; border-radius:10px; font-size:1rem; }
  button { background:var(--red); color:#fff; border:0; border-radius:10px; padding:12px 18px; font-size:1rem; cursor:pointer; }
  button:disabled { opacity:.6; cursor:wait; }
</style>
</head>
<body>
<header>
  <h1>עוזר הפניקס</h1>
  <p>עונה רק מתוך המידע הציבורי באתר fnx.co.il. לא מחליף את תנאי הפוליסה או את שירות הלקוחות.</p>
</header>
<main>
  <div id="log"></div>
  <form id="f">
    <input id="q" type="text" autocomplete="off" placeholder="שאלה על ביטוח, תביעה או שירות…"/>
    <button type="submit">שלח</button>
  </form>
</main>
<script>
const logEl = document.getElementById("log");
const form = document.getElementById("f");
const input = document.getElementById("q");
const history = [];
function add(role, html) {
  const d = document.createElement("div");
  d.className = "msg " + role;
  d.innerHTML = html;
  logEl.appendChild(d);
  logEl.scrollTop = logEl.scrollHeight;
}
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  input.value = "";
  add("user", q.replace(/</g,"&lt;"));
  form.querySelector("button").disabled = true;
  try {
    const r = await fetch("/api/ask", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({question: q, history: history.slice(-8)})
    });
    const data = await r.json();
    let html = (data.answer || "").replace(/</g,"&lt;");
    if (data.citations && data.citations.length) {
      html += '<div class="src">מקורות:<br>' + data.citations.map(u =>
        '<a href="'+u+'" target="_blank" rel="noopener">'+u+'</a>').join("<br>") + "</div>";
    }
    if (data.disclaimer) html += '<div class="disc">'+data.disclaimer.replace(/</g,"&lt;")+"</div>";
    add("bot", html);
    history.push({role:"user", content:q});
    history.push({role:"assistant", content: data.answer || ""});
  } catch (err) {
    add("bot", "שגיאה: " + err);
  }
  form.querySelector("button").disabled = false;
  input.focus();
});
</script>
</body>
</html>
"""


def serve(cfg: Config, host: str = "127.0.0.1", port: int = 7860) -> None:
    pipe = Pipeline(cfg)
    log.info("loading embedder + index for chat UI…")
    pipe.ensure()
    lock = threading.Lock()
    disclaimer = cfg.messages.get("disclaimer") or ""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            log.info("%s - " + fmt, self.address_string(), *args)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if urlparse(self.path).path in ("/", "/index.html"):
                self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/ask":
                self._send(404, b"not found", "text/plain")
                return
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            question = str(payload.get("question") or "")
            history = list(payload.get("history") or [])
            with lock:
                ans = pipe.ask(question, history=history)
            body = json.dumps({
                "status": ans.status,
                "answer": ans.text,
                "citations": ans.citations,
                "disclaimer": disclaimer,
                "answer_type": ans.answer_type,
                "sources": ans.citations,
                "used_chunk_ids": ans.used_chunk_ids,
            }, ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")

    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"chat UI: {url}")
    log.info("serving chat UI at %s", url)
    httpd.serve_forever()
