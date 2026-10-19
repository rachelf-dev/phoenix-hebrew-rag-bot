// Phoenix RAG chat client: preserves עיצוב.txt UX; calls POST /api/ask on the FastAPI backend.

(function () {
  "use strict";

  let conversationHistory = [];
  let isChatOpen = false;
  let chatEverOpened = false;
  let chatCloseTimer = null;

  const chatWindow = document.getElementById("chat-window");
  const messagesContainer = document.getElementById("messages-container");
  const scrollToBottomBtn = document.getElementById("scroll-to-bottom-btn");
  const scrollBottomAnchor = document.getElementById("scroll-bottom-anchor");
  const userInput = document.getElementById("user-input");
  const mobileMenuBtn = document.getElementById("mobile-menu-btn");
  const mobileMenu = document.getElementById("mobile-menu");
  const launcherWrap = document.getElementById("launcher-wrap");
  const widgetContainer = document.getElementById("widget-container");

  function normUrlKey(url) {
    return String(url || "")
      .trim()
      .replace(/^https?:\/\/(www\.)?/i, "")
      .replace(/\/$/, "")
      .toLowerCase();
  }

  function escapeHtml(text) {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return String(text).replace(/[&<>"']/g, (m) => map[m]);
  }

  /** Official page title (h1) — not chunk heading_path text. */
  function pageLabel(hit) {
    return (hit.page_h1 || hit.page_title || "").trim();
  }

  function isGenericSourceTitle(title) {
    if (!title) return true;
    const t = String(title).trim();
    if (!t || t === "מקור באתר הפניקס" || t === "עמוד באתר הפניקס") return true;
    if (/^https?:\/\//i.test(t) || t.includes("fnx.co.il/")) return true;
    return false;
  }

  function buildUrlTitleMap(hits, sources) {
    const map = new Map();
    (sources || []).forEach((s) => {
      if (s && s.url && s.title && !isGenericSourceTitle(s.title)) {
        map.set(normUrlKey(s.url), s.title);
      }
    });
    (hits || []).forEach((h) => {
      const url = h.url || h.display_url;
      const label = pageLabel(h);
      if (url && label && !isGenericSourceTitle(label)) map.set(normUrlKey(url), label);
    });
    return map;
  }

  function titleForUrl(url, urlTitles) {
    const key = normUrlKey(url);
    if (urlTitles.has(key)) return urlTitles.get(key);
    return "";
  }

  function formatAnswerHtml(text, hits, citations, sources) {
    const urlTitles = buildUrlTitleMap(hits, sources);
    let plain = String(text || "")
      .replace(/\(\s*fnx\.co\.il\s*\)/gi, "")
      .replace(/https?:\/\/(www\.)?fnx\.co\.il[^\s)\]]*/gi, "");
    let out = escapeHtml(plain);

    out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (_, label, url) => {
      const safeLabel = escapeHtml(label.trim() || titleForUrl(url, urlTitles));
      if (!safeLabel) return safeLabel;
      const safeUrl = escapeHtml(url);
      return `<a class="text-[#ff551f] font-semibold underline decoration-[#ff551f]/40 hover:decoration-[#ff551f]" href="${safeUrl}" target="_blank" rel="noopener">${safeLabel}</a>`;
    });

    out = out.replace(/(^|[\s(>])(https?:\/\/[^\s<)"']+)/g, (match, prefix, url) => {
      const label = titleForUrl(url, urlTitles);
      if (!label) return match;
      const safeUrl = escapeHtml(url);
      return `${prefix}<a class="text-[#ff551f] font-semibold underline decoration-[#ff551f]/40 hover:decoration-[#ff551f]" href="${safeUrl}" target="_blank" rel="noopener">${escapeHtml(label)}</a>`;
    });

    return out.replace(/\n/g, "<br>");
  }

  /** Plain text for typewriter: no bare URLs, markdown labels only. */
  function plainAnswerForTyping(text) {
    let out = String(text || "")
      .replace(/\(\s*fnx\.co\.il\s*\)/gi, "")
      .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, "$1");
    out = out.replace(/https?:\/\/[^\s)\]]+/g, "");
    return out.replace(/\s{2,}/g, " ").trim();
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

  function normalizeSources(sources) {
    if (!sources || !sources.length || typeof sources[0] !== "object") return [];
    return sources.filter(
      (s) => s && s.url && s.title && !isGenericSourceTitle(s.title)
    );
  }

  function hitTitleForCitationUrl(url, hits) {
    const key = normUrlKey(url);
    const slug = key.split("/").filter(Boolean).pop() || "";
    for (const h of hits || []) {
      const hu = normUrlKey(h.url || h.display_url);
      if (hu === key) return pageLabel(h);
    }
    if (slug) {
      for (const h of hits || []) {
        const hu = normUrlKey(h.url || h.display_url);
        if (hu.endsWith("/" + slug) || hu.endsWith(slug)) return pageLabel(h);
      }
    }
    return "";
  }

  /** Merge API source cards with citations + hits (legacy API / missing titles). */
  function coalesceSourceCards(citations, hits, sources) {
    let cards = normalizeSources(sources);
    if (cards.length) return cards;

    let citeList = Array.isArray(citations) ? citations.slice() : [];
    if (!citeList.length && sources && sources.length && typeof sources[0] === "string") {
      citeList = sources.slice();
    }

    const seen = new Set();
    cards = [];
    citeList.forEach((url) => {
      if (!url) return;
      const key = normUrlKey(url);
      if (seen.has(key)) return;
      const title = hitTitleForCitationUrl(url, hits);
      if (!title || isGenericSourceTitle(title)) return;
      seen.add(key);
      cards.push({ url: String(url).trim(), title });
    });

    if (cards.length) return cards;

    (hits || []).slice(0, 3).forEach((h) => {
      const url = (h.url || h.display_url || "").trim();
      if (!url) return;
      const key = normUrlKey(url);
      if (seen.has(key)) return;
      const title = pageLabel(h);
      if (!title || isGenericSourceTitle(title)) return;
      seen.add(key);
      cards.push({ url, title });
    });
    return cards;
  }

  function buildSourceBadge(hits, cards) {
    if (!cards || !cards.length) return "";

    let pills = cards
      .map(({ url, title }) => {
        const safe = escapeHtml(url);
        const label = escapeHtml(title);
        return `
          <a class="flex items-center justify-between p-2.5 rounded-xl bg-orange-50/70 hover:bg-orange-100/80 border border-orange-200/60 transition-colors group" href="${safe}" target="_blank" rel="noopener noreferrer" title="${label}">
            <div class="flex items-center gap-2 min-w-0 flex-1">
              <i class="w-3.5 h-3.5 text-[#ff551f] shrink-0" data-lucide="external-link"></i>
              <span class="text-xs font-bold text-slate-900 leading-snug">${label}</span>
            </div>
            <i class="w-3.5 h-3.5 text-[#ff551f] shrink-0 transform group-hover:-translate-x-0.5 transition-transform" data-lucide="arrow-left"></i>
          </a>`;
      })
      .join("");

    let retrieval = "";
    if (hits && hits.length) {
      const rows = hits
        .slice(0, 4)
        .map((h) => {
          const title = escapeHtml(pageLabel(h));
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
        <span>מחפש את המידע בשבילך...</span>
      </div>`;
    messagesContainer.insertBefore(typingIndicator, scrollBottomAnchor);
    lucide.createIcons();
  }

  function appendBotResponse(answer, citations, hits, sources) {
    const ind = document.getElementById("temp-indicator");
    if (ind) ind.remove();

    const sourceCards = coalesceSourceCards(citations, hits, sources);
    const badgeHtml = buildSourceBadge(hits, sourceCards);

    const uniqueId = "stream-" + Date.now();
    const wrap = document.createElement("div");
    wrap.className = "flex items-start gap-2.5 chat-dynamic";
    wrap.innerHTML = `
      <div class="w-7 h-7 rounded-lg bg-[#ff551f] text-white flex items-center justify-center shrink-0 shadow-sm mt-0.5">
        <i data-lucide="bot" class="w-4 h-4"></i>
      </div>
      <div class="max-w-[88%] bg-white border border-slate-200 rounded-2xl rounded-tr-none p-3.5 shadow-bubble text-slate-800 text-sm leading-relaxed relative">
        <div id="${uniqueId}" class="cursor-blink text-slate-800 text-xs leading-relaxed space-y-2 mb-2"></div>
        ${badgeHtml}
      </div>`;
    messagesContainer.insertBefore(wrap, scrollBottomAnchor);
    lucide.createIcons();

    const targetDiv = document.getElementById(uniqueId);
    const typingText = plainAnswerForTyping(answer);
    typewriterInto(targetDiv, typingText, 14, () => {
      targetDiv.innerHTML = formatAnswerHtml(answer, hits, citations, sourceCards);
      targetDiv.classList.remove("mb-2");
      lucide.createIcons();
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
      appendBotResponse(answer, data.citations || [], data.hits || [], data.sources || []);
      conversationHistory.push({ role: "user", content: text });
      conversationHistory.push({ role: "assistant", content: answer });
    } catch (err) {
      const ind = document.getElementById("temp-indicator");
      if (ind) ind.remove();
      appendBotResponse("לא הצלחתי לענות כרגע. נסו שוב בעוד רגע.", [], [], []);
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

  function hideLauncherBeacon() {
    const beacon = document.getElementById("launcher-beacon");
    if (beacon) beacon.remove();
  }

  function applyChatState() {
    if (!widgetContainer || !chatWindow || !launcherWrap) return;

    if (chatCloseTimer) {
      clearTimeout(chatCloseTimer);
      chatCloseTimer = null;
    }

    widgetContainer.dataset.chat = isChatOpen ? "open" : "closed";
    const chevron = document.getElementById("launcher-chevron");

    if (isChatOpen) {
      chatWindow.classList.remove("chat-panel-closed", "pointer-events-none");
      chatWindow.setAttribute("aria-hidden", "false");
      launcherWrap.setAttribute("aria-hidden", "true");
      requestAnimationFrame(() => {
        chatWindow.classList.remove("scale-90", "opacity-0");
        chatWindow.classList.add("scale-100", "opacity-100");
      });
      if (chevron) chevron.setAttribute("data-lucide", "chevron-down");
    } else {
      chatWindow.classList.remove("scale-100", "opacity-100");
      chatWindow.classList.add("scale-90", "opacity-0", "pointer-events-none");
      chatWindow.setAttribute("aria-hidden", "true");
      launcherWrap.setAttribute("aria-hidden", "false");
      chatCloseTimer = window.setTimeout(() => {
        if (!isChatOpen) chatWindow.classList.add("chat-panel-closed");
        chatCloseTimer = null;
      }, 300);
      if (chevron) chevron.setAttribute("data-lucide", "chevron-up");
    }
  }

  function openChatWindow() {
    if (isChatOpen) return;
    if (!chatEverOpened) {
      chatEverOpened = true;
      hideLauncherBeacon();
    }
    isChatOpen = true;
    applyChatState();
    lucide.createIcons();
  }

  function closeChatWindow() {
    if (!isChatOpen) return;
    isChatOpen = false;
    applyChatState();
    lucide.createIcons();
  }

  function toggleChatWindow() {
    if (isChatOpen) closeChatWindow();
    else openChatWindow();
  }

  window.handleSampleQuery = function (queryText) {
    if (!isChatOpen) openChatWindow();
    userInput.value = queryText;
    submitUserMessage();
  };
  window.submitUserMessage = submitUserMessage;
  window.toggleChatWindow = toggleChatWindow;
  window.scrollToBottomSmooth = scrollToBottomSmooth;

  document.addEventListener("DOMContentLoaded", () => {
    lucide.createIcons();
    applyChatState();

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
