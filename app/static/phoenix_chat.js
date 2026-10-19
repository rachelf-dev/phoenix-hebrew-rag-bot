// Phoenix RAG chat client: preserves עיצוב.txt UX; calls POST /api/ask on the FastAPI backend.

(function () {
  "use strict";

  let conversationHistory = [];
  let isChatOpen = true;
  let metaPages = "—";
  let metaChunks = "—";

  const chatWindow = document.getElementById("chat-window");
  const messagesContainer = document.getElementById("messages-container");
  const scrollToBottomBtn = document.getElementById("scroll-to-bottom-btn");
  const scrollBottomAnchor = document.getElementById("scroll-bottom-anchor");
  const userInput = document.getElementById("user-input");
  const mobileMenuBtn = document.getElementById("mobile-menu-btn");
  const mobileMenu = document.getElementById("mobile-menu");

  function escapeHtml(text) {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return String(text).replace(/[&<>"']/g, (m) => map[m]);
  }

  function formatAnswerHtml(text) {
    return escapeHtml(text).replace(/\n/g, "<br>");
  }

  function scrollToBottomSmooth() {
    messagesContainer.scrollTo({ top: messagesContainer.scrollHeight, behavior: "smooth" });
  }

  function removeDynamicMessages() {
    messagesContainer.querySelectorAll(".chat-dynamic").forEach((el) => el.remove());
  }

  function typewriterInto(el, text, speed, onDone) {
    el.classList.add("cursor-blink");
    el.innerHTML = "";
    let i = 0;
    function tick() {
      if (i >= text.length) {
        el.classList.remove("cursor-blink");
        if (onDone) onDone();
        return;
      }
      const ch = text.charAt(i);
      el.innerHTML += ch === "\n" ? "<br>" : escapeHtml(ch);
      i += 1;
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
      setTimeout(tick, speed);
    }
    tick();
  }

  function buildSourceBadge(citations, hits) {
    if (!citations || !citations.length) return "";

    let pills = citations
      .map((url) => {
        const safe = escapeHtml(url);
        const label = escapeHtml(url.replace(/^https?:\/\/(www\.)?fnx\.co\.il/i, "fnx.co.il"));
        return `
          <a class="flex items-center justify-between p-2 rounded-xl bg-orange-50/70 hover:bg-orange-100/80 border border-orange-200/60 transition-colors group" href="${safe}" target="_blank" rel="noopener">
            <div class="flex items-center gap-2 overflow-hidden">
              <i class="w-3.5 h-3.5 text-[#ff551f] shrink-0" data-lucide="external-link"></i>
              <span class="text-xs font-semibold text-slate-800 truncate">${label}</span>
            </div>
            <i class="w-3.5 h-3.5 text-[#ff551f] transform group-hover:-translate-x-0.5 transition-transform" data-lucide="arrow-left"></i>
          </a>`;
      })
      .join("");

    let retrieval = "";
    if (hits && hits.length) {
      const rows = hits
        .slice(0, 4)
        .map((h) => {
          const title = escapeHtml(h.page_title || h.url || "");
          const prev = escapeHtml((h.preview || "").slice(0, 180));
          return `<li class="text-[10px] text-slate-600 mt-1"><strong>${title}</strong><br>${prev}</li>`;
        })
        .join("");
      retrieval = `
        <details class="mt-2 text-[11px] text-slate-600">
          <summary class="cursor-pointer text-[#ff551f] font-semibold">איך מצאתי את זה</summary>
          <ul class="list-disc pr-4 mt-1">${rows}</ul>
        </details>`;
    }

    return `
      <div class="pt-2 mt-2 border-t border-slate-100 flex flex-col gap-2 source-badge-block">
        <div class="flex items-center justify-between text-[11px] text-slate-500">
          <span class="flex items-center gap-1 text-emerald-600 font-semibold">
            <i class="w-3.5 h-3.5" data-lucide="check-circle-2"></i>
            מקור מאומת מתוך אתר הפניקס
          </span>
        </div>
        ${pills}
        ${retrieval}
      </div>`;
  }

  function appendUserBubble(text) {
    const userBubble = document.createElement("div");
    userBubble.className = "flex items-start justify-end gap-2.5 chat-dynamic";
    userBubble.innerHTML = `
      <div class="max-w-[85%] bg-[#0c172d] text-white rounded-2xl rounded-tl-none p-3.5 shadow-sm text-sm leading-relaxed whitespace-pre-wrap">${escapeHtml(text)}</div>
      <div class="w-7 h-7 rounded-lg bg-slate-300 text-slate-700 flex items-center justify-center shrink-0 shadow-sm mt-0.5">
        <i data-lucide="user" class="w-4 h-4"></i>
      </div>`;
    messagesContainer.insertBefore(userBubble, scrollBottomAnchor);
    lucide.createIcons();
  }

  function showTypingIndicator() {
    const typingIndicator = document.createElement("div");
    typingIndicator.id = "temp-indicator";
    typingIndicator.className = "flex items-start gap-2.5 chat-dynamic";
    typingIndicator.innerHTML = `
      <div class="w-7 h-7 rounded-lg bg-[#ff551f] text-white flex items-center justify-center shrink-0 shadow-sm mt-0.5">
        <i data-lucide="bot" class="w-4 h-4"></i>
      </div>
      <div class="bg-white border border-slate-200 rounded-2xl rounded-tr-none px-4 py-3 shadow-bubble flex items-center gap-1.5 text-xs text-slate-500">
        <span class="w-2 h-2 rounded-full bg-[#ff551f] animate-pulse"></span>
        <span>מחפש ב-${metaPages} עמודי האתר ומפעיל RAG היברידי...</span>
      </div>`;
    messagesContainer.insertBefore(typingIndicator, scrollBottomAnchor);
    lucide.createIcons();
  }

  function appendBotResponse(answer, citations, hits) {
    const ind = document.getElementById("temp-indicator");
    if (ind) ind.remove();

    const uniqueId = "stream-" + Date.now();
    const badgeId = "badge-" + uniqueId;
    const wrap = document.createElement("div");
    wrap.className = "flex items-start gap-2.5 chat-dynamic";
    wrap.innerHTML = `
      <div class="w-7 h-7 rounded-lg bg-[#ff551f] text-white flex items-center justify-center shrink-0 shadow-sm mt-0.5">
        <i data-lucide="bot" class="w-4 h-4"></i>
      </div>
      <div class="max-w-[88%] bg-white border border-slate-200 rounded-2xl rounded-tr-none p-3.5 shadow-bubble text-slate-800 text-sm leading-relaxed relative">
        <div id="${uniqueId}" class="cursor-blink text-slate-800 text-xs leading-relaxed space-y-2 mb-2"></div>
        <div id="${badgeId}"></div>
      </div>`;
    messagesContainer.insertBefore(wrap, scrollBottomAnchor);
    lucide.createIcons();

    const targetDiv = document.getElementById(uniqueId);
    typewriterInto(targetDiv, answer, 14, () => {
      const badgeHost = document.getElementById(badgeId);
      if (badgeHost) {
        badgeHost.innerHTML = buildSourceBadge(citations, hits);
        lucide.createIcons();
      }
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    });
  }

  async function submitUserMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    appendUserBubble(text);
    userInput.value = "";
    scrollToBottomSmooth();
    showTypingIndicator();
    scrollToBottomSmooth();

    const submitBtn = document.querySelector("#chat-form button[type='submit']");
    if (submitBtn) submitBtn.disabled = true;

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, history: conversationHistory }),
      });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      const answer = data.answer || data.answer_text || "";
      appendBotResponse(answer, data.citations || data.sources || [], data.hits || []);
      conversationHistory.push({ role: "user", content: text });
      conversationHistory.push({ role: "assistant", content: answer });
    } catch (err) {
      const ind = document.getElementById("temp-indicator");
      if (ind) ind.remove();
      appendBotResponse("לא הצלחתי לענות כרגע. נסו שוב בעוד רגע.", [], []);
      console.error(err);
    } finally {
      if (submitBtn) submitBtn.disabled = false;
      userInput.focus();
    }
  }

  function resetConversation() {
    conversationHistory = [];
    removeDynamicMessages();
    scrollToBottomSmooth();
  }

  function toggleChatWindow() {
    isChatOpen = !isChatOpen;
    const chevron = document.getElementById("launcher-chevron");
    if (isChatOpen) {
      chatWindow.classList.remove("hidden", "scale-90", "opacity-0");
      chatWindow.classList.add("scale-100", "opacity-100");
      if (chevron) chevron.setAttribute("data-lucide", "chevron-down");
    } else {
      chatWindow.classList.add("scale-90", "opacity-0");
      setTimeout(() => chatWindow.classList.add("hidden"), 200);
      if (chevron) chevron.setAttribute("data-lucide", "chevron-up");
    }
    lucide.createIcons();
  }

  window.handleSampleQuery = function (queryText) {
    userInput.value = queryText;
    submitUserMessage();
  };
  window.submitUserMessage = submitUserMessage;
  window.toggleChatWindow = toggleChatWindow;
  window.scrollToBottomSmooth = scrollToBottomSmooth;

  async function loadMeta() {
    try {
      const res = await fetch("/api/meta");
      if (!res.ok) return;
      const data = await res.json();
      metaPages = data.pages != null ? String(data.pages) : metaPages;
      metaChunks = data.chunks != null ? data.chunks.toLocaleString("he-IL") : metaChunks;
      const statusLine = document.getElementById("index-status-line");
      if (statusLine) {
        statusLine.textContent = `מאומת מול ${metaPages} עמודי fnx.co.il • מודל e5-hybrid`;
      }
      const footer = document.getElementById("footer-index-line");
      if (footer) {
        footer.textContent = `אינדקס פעיל: ${metaChunks} מקטעים • e5 + BM25`;
      }
    } catch (_) {
      /* keep defaults */
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    lucide.createIcons();
    loadMeta();

    if (mobileMenuBtn && mobileMenu) {
      mobileMenuBtn.addEventListener("click", () => mobileMenu.classList.toggle("hidden"));
    }

    messagesContainer.addEventListener("scroll", () => {
      const scrollPos = messagesContainer.scrollTop + messagesContainer.clientHeight;
      const distanceFromBottom = messagesContainer.scrollHeight - scrollPos;
      if (distanceFromBottom > 70) {
        scrollToBottomBtn.classList.remove("opacity-0", "pointer-events-none", "translate-y-2");
        scrollToBottomBtn.classList.add("opacity-100", "translate-y-0");
      } else {
        scrollToBottomBtn.classList.add("opacity-0", "pointer-events-none", "translate-y-2");
        scrollToBottomBtn.classList.remove("opacity-100", "translate-y-0");
      }
    });

    document.getElementById("btn-minimize").addEventListener("click", toggleChatWindow);
    document.getElementById("btn-reset").addEventListener("click", resetConversation);
  });
})();
