// Phoenix RAG chat client: preserves עיצוב.txt UX; calls POST /api/ask on the FastAPI backend.

(function () {
  "use strict";

  let conversationHistory = [];
  let chatSessionId = 0;
  let activeAskAbort = null;
  let activeTypewriterCancel = null;
  let isChatOpen = false;
  let chatEverOpened = false;
  let chatCloseTimer = null;
  let isSubmitting = false;
  let pendingQuoteText = "";
  let selectionQuotePayload = "";
  const CHAT_ANIM_MS = 350;

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

  function abortActiveChatWork() {
    if (activeAskAbort) {
      try {
        activeAskAbort.abort();
      } catch (_) {
        /* ignore */
      }
      activeAskAbort = null;
    }
    if (activeTypewriterCancel) {
      activeTypewriterCancel();
      activeTypewriterCancel = null;
    }
    const ind = document.getElementById("temp-indicator");
    if (ind) ind.remove();
  }

  let speechRecognition = null;
  let isVoiceListening = false;
  let voiceSessionPrefix = "";
  let voiceSessionFinal = "";

  const SpeechRecognitionCtor =
    window.SpeechRecognition || window.webkitSpeechRecognition;

  function setVoiceStatus(message, isError) {
    const el = document.getElementById("voice-status");
    if (!el) return;
    el.textContent = message || "";
    el.classList.remove("text-red-600", "text-slate-500");
    if (!message) return;
    el.classList.add(isError ? "text-red-600" : "text-slate-500");
  }

  function setVoiceListeningUI(listening) {
    isVoiceListening = listening;
    const btn = document.getElementById("voice-input-btn");
    if (!btn) return;
    btn.setAttribute("aria-pressed", listening ? "true" : "false");
    btn.setAttribute(
      "aria-label",
      listening ? "מקליט — לחצו לעצירה" : "הקלטת שאלה"
    );
    btn.setAttribute("title", listening ? "מקליט… לחצו לעצירה" : "הקלטת שאלה");
    btn.classList.toggle("voice-listening", listening);
  }

  function resetVoiceSessionState() {
    voiceSessionPrefix = "";
    voiceSessionFinal = "";
  }

  function clearUserInput() {
    resetVoiceSessionState();
    if (userInput) userInput.value = "";
  }

  function clearPendingQuote() {
    pendingQuoteText = "";
    const block = document.getElementById("input-quote-block");
    const textEl = document.getElementById("input-quote-text");
    if (textEl) textEl.textContent = "";
    if (block) block.classList.add("hidden");
  }

  function setPendingQuote(text) {
    const snippet = String(text || "").trim().replace(/\s+/g, " ");
    if (!snippet) return;
    pendingQuoteText = snippet.slice(0, 600);
    const block = document.getElementById("input-quote-block");
    const textEl = document.getElementById("input-quote-text");
    if (textEl) textEl.textContent = pendingQuoteText;
    if (block) block.classList.remove("hidden");
    hideSelectionQuoteBtn();
    window.getSelection()?.removeAllRanges();
    if (userInput) {
      userInput.focus();
      userInput.setSelectionRange(userInput.value.length, userInput.value.length);
    }
  }

  function buildApiQuestion(userText) {
    const q = (userText || "").trim();
    if (!pendingQuoteText) return q;
    if (!q) return `בהתייחס לציטוט הבא מהתשובה: «${pendingQuoteText}»`;
    return `בהתייחס לציטוט הבא מהתשובה: «${pendingQuoteText}»\n\n${q}`;
  }

  function formatUserBubbleContent(userText, quoteAtSend) {
    const q = escapeHtml((userText || "").trim());
    if (!quoteAtSend) return q;
    const quote = escapeHtml(quoteAtSend);
    const body = q
      ? `<p class="mt-2 font-medium">${q}</p>`
      : "";
    return (
      `<div class="border-r-4 border-orange-300/90 pr-2.5 text-xs font-medium text-white/95 leading-[1.5]">«${quote}»</div>${body}`
    );
  }

  function hideSelectionQuoteBtn() {
    const btn = document.getElementById("selection-quote-btn");
    if (btn) btn.classList.add("hidden");
    selectionQuotePayload = "";
  }

  function positionSelectionQuoteBtn(clientRect) {
    const btn = document.getElementById("selection-quote-btn");
    if (!btn || !messagesContainer) return;
    const host = messagesContainer.getBoundingClientRect();
    const top = clientRect.top - host.top - 8;
    const left = clientRect.left - host.left + clientRect.width / 2;
    btn.style.top = `${Math.max(8, top)}px`;
    btn.style.left = `${Math.min(host.width - 8, Math.max(8, left))}px`;
    btn.classList.remove("hidden");
    lucide.createIcons();
  }

  function initSelectionQuote() {
    const btn = document.getElementById("selection-quote-btn");
    const clearBtn = document.getElementById("clear-quote-btn");
    if (clearBtn) {
      clearBtn.addEventListener("click", () => clearPendingQuote());
    }
    if (btn) {
      btn.addEventListener("mousedown", (e) => e.preventDefault());
      btn.addEventListener("click", () => {
        if (selectionQuotePayload) setPendingQuote(selectionQuotePayload);
      });
    }
    if (!messagesContainer) return;

    function handleSelection() {
      window.setTimeout(() => {
        const sel = window.getSelection();
        if (!sel || sel.rangeCount === 0 || sel.isCollapsed) {
          hideSelectionQuoteBtn();
          return;
        }
        const text = sel.toString().trim().replace(/\s+/g, " ");
        if (text.length < 2) {
          hideSelectionQuoteBtn();
          return;
        }
        const node = sel.anchorNode;
        const el = node && node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
        if (!el || !el.closest(".chat-selectable-answer")) {
          hideSelectionQuoteBtn();
          return;
        }
        const range = sel.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        if (!rect.width && !rect.height) {
          hideSelectionQuoteBtn();
          return;
        }
        selectionQuotePayload = text;
        positionSelectionQuoteBtn(rect);
      }, 12);
    }

    messagesContainer.addEventListener("mouseup", handleSelection);
    messagesContainer.addEventListener("touchend", handleSelection);
    document.addEventListener("mousedown", (e) => {
      const btn = document.getElementById("selection-quote-btn");
      if (btn && !btn.classList.contains("hidden") && !btn.contains(e.target)) {
        window.setTimeout(() => {
          const sel = window.getSelection();
          if (!sel || sel.isCollapsed) hideSelectionQuoteBtn();
        }, 0);
      }
    });
    messagesContainer.addEventListener("scroll", hideSelectionQuoteBtn, { passive: true });
  }

  function stopVoiceInput() {
    if (speechRecognition && isVoiceListening) {
      try {
        speechRecognition.stop();
      } catch (_) {
        /* already stopped */
      }
    }
    setVoiceListeningUI(false);
  }

  function initVoiceInput() {
    const btn = document.getElementById("voice-input-btn");
    if (!btn || !userInput) return;

    if (!SpeechRecognitionCtor) {
      btn.hidden = true;
      return;
    }

    speechRecognition = new SpeechRecognitionCtor();
    speechRecognition.lang = "he-IL";
    speechRecognition.interimResults = true;
    speechRecognition.continuous = true;

    speechRecognition.onresult = (event) => {
      if (isSubmitting || !isVoiceListening) return;
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const piece = event.results[i][0].transcript;
        if (event.results[i].isFinal) voiceSessionFinal += piece;
        else interim += piece;
      }
      userInput.value = voiceSessionPrefix + voiceSessionFinal + interim;
    };

    speechRecognition.onerror = (event) => {
      const code = event.error || "";
      if (code === "aborted") return;
      if (code === "not-allowed" || code === "service-not-allowed") {
        setVoiceStatus("אין גישה למיקרופון. אפשרו הרשאה בהגדרות הדפדפן.", true);
      } else if (code === "no-speech") {
        setVoiceStatus("לא זוהה דיבור. נסו שוב.", true);
      } else if (code === "network") {
        setVoiceStatus("שגיאת רשת בזיהוי הקול. נסו שוב.", true);
      } else if (code) {
        setVoiceStatus("ההקלטה לא הצליחה. נסו שוב.", true);
      }
      setVoiceListeningUI(false);
    };

    speechRecognition.onend = () => {
      setVoiceListeningUI(false);
    };

    btn.addEventListener("click", () => {
      if (isSubmitting) return;
      if (isVoiceListening) {
        stopVoiceInput();
        return;
      }
      setVoiceStatus("");
      const existing = userInput.value.trim();
      voiceSessionPrefix = existing ? existing + " " : "";
      voiceSessionFinal = "";
      try {
        speechRecognition.start();
        setVoiceListeningUI(true);
      } catch (err) {
        const msg = String(err && err.message ? err.message : err);
        if (/already/i.test(msg)) {
          setVoiceListeningUI(true);
          return;
        }
        setVoiceStatus("לא הצלחנו להתחיל הקלטה.", true);
        setVoiceListeningUI(false);
      }
    });
  }

  function setSubmittingLocked(locked) {
    isSubmitting = locked;
    if (userInput) userInput.disabled = locked;
    const submitBtn = document.querySelector("#chat-form button[type='submit']");
    if (submitBtn) submitBtn.disabled = locked;
    const voiceBtn = document.getElementById("voice-input-btn");
    if (voiceBtn) voiceBtn.disabled = locked;
    document.querySelectorAll("#suggested-queries button").forEach((btn) => {
      btn.disabled = locked;
    });
    if (locked) stopVoiceInput();
  }

  function typewriterInto(el, text, speed, onDone) {
    el.classList.add("cursor-blink");
    el.innerHTML = "";
    let i = 0;
    let cancelled = false;
    let timerId = null;

    function cancel() {
      cancelled = true;
      if (timerId != null) clearTimeout(timerId);
      el.classList.remove("cursor-blink");
    }

    function tick() {
      if (cancelled || !el.isConnected) return;
      if (i >= text.length) {
        el.classList.remove("cursor-blink");
        if (onDone && !cancelled) onDone();
        return;
      }
      const ch = text.charAt(i);
      el.innerHTML += ch === "\n" ? "<br>" : escapeHtml(ch);
      i += 1;
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
      timerId = setTimeout(tick, speed);
    }
    tick();
    return cancel;
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

  function appendUserBubble(userText, quoteAtSend) {
    const userBubble = document.createElement("div");
    userBubble.className = "flex items-start justify-end gap-2.5 chat-dynamic";
    const body = formatUserBubbleContent(userText, quoteAtSend);
    userBubble.innerHTML = `
      <div class="max-w-[85%] bg-[#0c172d] text-white rounded-2xl rounded-tl-none p-3.5 shadow-sm chat-message-text">${body}</div>
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

  function appendBotResponse(answer, citations, hits, sources, sessionId) {
    if (sessionId !== chatSessionId) {
      return Promise.resolve();
    }

    const ind = document.getElementById("temp-indicator");
    if (ind) ind.remove();

    const sourceCards = coalesceSourceCards(citations, hits, sources);

    const uniqueId = "stream-" + Date.now();
    const badgeId = "badge-" + uniqueId;
    const wrap = document.createElement("div");
    wrap.className = "flex items-start gap-2.5 chat-dynamic";
    wrap.innerHTML = `
      <div class="w-7 h-7 rounded-lg bg-[#ff551f] text-white flex items-center justify-center shrink-0 shadow-sm mt-0.5">
        <i data-lucide="bot" class="w-4 h-4"></i>
      </div>
      <div class="max-w-[88%] bg-white border border-slate-200 rounded-2xl rounded-tr-none p-3.5 shadow-bubble text-slate-800 chat-message-text relative">
        <div id="${uniqueId}" class="chat-selectable-answer chat-bot-body cursor-blink text-slate-800 space-y-2 mb-2"></div>
        <div id="${badgeId}"></div>
      </div>`;
    messagesContainer.insertBefore(wrap, scrollBottomAnchor);
    lucide.createIcons();

    const targetDiv = document.getElementById(uniqueId);
    const typingText = plainAnswerForTyping(answer);
    return new Promise((resolve) => {
      if (sessionId !== chatSessionId) {
        wrap.remove();
        resolve();
        return;
      }
      if (activeTypewriterCancel) {
        activeTypewriterCancel();
        activeTypewriterCancel = null;
      }
      activeTypewriterCancel = typewriterInto(targetDiv, typingText, 14, () => {
        if (sessionId !== chatSessionId) {
          wrap.remove();
          activeTypewriterCancel = null;
          resolve();
          return;
        }
        targetDiv.innerHTML = formatAnswerHtml(answer, hits, citations, sourceCards);
        targetDiv.classList.remove("mb-2");
        const badgeHost = document.getElementById(badgeId);
        if (badgeHost) {
          badgeHost.innerHTML = buildSourceBadge(hits, sourceCards);
        }
        lucide.createIcons();
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        activeTypewriterCancel = null;
        resolve();
      });
    });
  }

  async function submitUserMessage() {
    if (isSubmitting) return;
    const userText = userInput.value.trim();
    const text = buildApiQuestion(userText);
    if (!text) return;

    const quoteAtSend = pendingQuoteText;
    const sessionAtSend = chatSessionId;
    const controller = new AbortController();
    activeAskAbort = controller;

    stopVoiceInput();
    clearUserInput();
    setSubmittingLocked(true);

    appendUserBubble(userText, quoteAtSend);
    scrollToBottomSmooth();
    showTypingIndicator();
    scrollToBottomSmooth();

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, history: conversationHistory }),
        signal: controller.signal,
      });
      if (sessionAtSend !== chatSessionId) return;
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      if (sessionAtSend !== chatSessionId) return;
      const answer = data.answer || data.answer_text || "";
      await appendBotResponse(
        answer,
        data.citations || [],
        data.hits || [],
        data.sources || [],
        sessionAtSend
      );
      if (sessionAtSend !== chatSessionId) return;
      conversationHistory.push({ role: "user", content: text });
      conversationHistory.push({ role: "assistant", content: answer });
      clearPendingQuote();
    } catch (err) {
      if (err && err.name === "AbortError") return;
      if (sessionAtSend !== chatSessionId) return;
      const ind = document.getElementById("temp-indicator");
      if (ind) ind.remove();
      await appendBotResponse(
        "לא הצלחתי לענות כרגע. נסו שוב בעוד רגע.",
        [],
        [],
        [],
        sessionAtSend
      );
      console.error(err);
    } finally {
      if (activeAskAbort === controller) activeAskAbort = null;
      if (sessionAtSend === chatSessionId) {
        setSubmittingLocked(false);
        if (userInput) userInput.focus();
      }
    }
  }

  function resetConversation() {
    chatSessionId += 1;
    abortActiveChatWork();
    conversationHistory = [];
    removeDynamicMessages();
    clearUserInput();
    clearPendingQuote();
    hideSelectionQuoteBtn();
    stopVoiceInput();
    setSubmittingLocked(false);
    scrollToBottomSmooth();
  }

  function hideLauncherBeacon() {
    const beacon = document.getElementById("launcher-beacon");
    if (beacon) beacon.remove();
  }

  function setChevron(open) {
    const chevron = document.getElementById("launcher-chevron");
    if (chevron) chevron.setAttribute("data-lucide", open ? "chevron-down" : "chevron-up");
  }

  function initChatClosed() {
    if (!widgetContainer || !chatWindow || !launcherWrap) return;
    if (chatCloseTimer) {
      clearTimeout(chatCloseTimer);
      chatCloseTimer = null;
    }
    isChatOpen = false;
    widgetContainer.dataset.chat = "closed";
    chatWindow.classList.add("chat-panel-closed");
    chatWindow.classList.remove("chat-panel-open");
    chatWindow.setAttribute("aria-hidden", "true");
    launcherWrap.classList.remove("launcher-hidden", "launcher-suppressed");
    launcherWrap.setAttribute("aria-hidden", "false");
    setChevron(false);
  }

  function hideLauncherInstant() {
    launcherWrap.classList.remove("launcher-hidden");
    launcherWrap.classList.add("launcher-suppressed");
    launcherWrap.setAttribute("aria-hidden", "true");
  }

  function openChatWindow() {
    if (isChatOpen || !widgetContainer || !chatWindow || !launcherWrap) return;
    if (!chatEverOpened) {
      chatEverOpened = true;
      hideLauncherBeacon();
    }
    if (chatCloseTimer) {
      clearTimeout(chatCloseTimer);
      chatCloseTimer = null;
    }

    isChatOpen = true;
    hideLauncherInstant();
    void launcherWrap.offsetHeight;
    widgetContainer.dataset.chat = "open";

    chatWindow.classList.remove("chat-panel-closed", "chat-panel-open");
    chatWindow.setAttribute("aria-hidden", "false");

    void chatWindow.offsetHeight;
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        chatWindow.classList.add("chat-panel-open");
      });
    });

    setChevron(true);
    lucide.createIcons();
  }

  function closeChatWindow() {
    if (!isChatOpen || !widgetContainer || !chatWindow || !launcherWrap) return;
    stopVoiceInput();
    isChatOpen = false;

    chatWindow.classList.remove("chat-panel-open");
    chatWindow.setAttribute("aria-hidden", "true");
    setChevron(false);

    if (chatCloseTimer) clearTimeout(chatCloseTimer);
    chatCloseTimer = window.setTimeout(() => {
      if (isChatOpen) return;
      chatWindow.classList.add("chat-panel-closed");
      widgetContainer.dataset.chat = "closed";
      launcherWrap.classList.remove("launcher-suppressed");
      launcherWrap.classList.add("launcher-hidden");
      launcherWrap.setAttribute("aria-hidden", "false");
      void launcherWrap.offsetHeight;
      requestAnimationFrame(() => {
        launcherWrap.classList.remove("launcher-hidden");
      });
      chatCloseTimer = null;
      lucide.createIcons();
    }, CHAT_ANIM_MS);

    lucide.createIcons();
  }

  function toggleChatWindow() {
    if (isChatOpen) closeChatWindow();
    else openChatWindow();
  }

  window.handleSampleQuery = function (queryText) {
    if (isSubmitting) return;
    if (!isChatOpen) openChatWindow();
    userInput.value = queryText;
    void submitUserMessage();
  };
  window.submitUserMessage = submitUserMessage;
  window.toggleChatWindow = toggleChatWindow;
  window.scrollToBottomSmooth = scrollToBottomSmooth;

  document.addEventListener("DOMContentLoaded", () => {
    lucide.createIcons();
    initChatClosed();

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
    initVoiceInput();
    initSelectionQuote();
  });
})();
