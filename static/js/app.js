(() => {
  const chatListEl = document.getElementById("chatList");
  const messagesEl = document.getElementById("messages");
  const emptyStateEl = document.getElementById("emptyState");
  const composerEl = document.getElementById("composer");
  const inputEl = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");
  const newChatBtn = document.getElementById("newChatBtn");
  const sidebar = document.getElementById("sidebar");
  const sidebarToggle = document.getElementById("sidebarToggle");
  const sidebarScrim = document.getElementById("sidebarScrim");

  let currentChatId = null;
  let chats = [];
  const pollTimers = new Map();

  // ---------- API ----------

  async function api(path, opts = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...opts,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error || `Request failed: ${res.status}`);
    }
    return res.json();
  }

  const getChats = () => api("/api/chats");
  const createChat = () => api("/api/chats", { method: "POST" });
  const getChat = (id) => api(`/api/chats/${id}`);
  const deleteChat = (id) => api(`/api/chats/${id}`, { method: "DELETE" });
  const postMessage = (id, text) =>
    api(`/api/chats/${id}/messages`, { method: "POST", body: JSON.stringify({ text }) });
  const getMessage = (id) => api(`/api/messages/${id}`);

  // ---------- Time formatting ----------

  function relativeTime(iso) {
    const then = new Date(iso).getTime();
    const diffSec = Math.max(0, (Date.now() - then) / 1000);
    if (diffSec < 60) return "just now";
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
    if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
    if (diffSec < 172800) return "yesterday";
    if (diffSec < 604800) return `${Math.floor(diffSec / 86400)}d ago`;
    return new Date(iso).toLocaleDateString();
  }

  function clockTime(iso) {
    return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  // ---------- Sidebar ----------

  function renderChatList() {
    chatListEl.innerHTML = "";
    if (chats.length === 0) {
      const empty = document.createElement("div");
      empty.className = "sidebar-empty";
      empty.textContent = "No chats yet — start one below.";
      chatListEl.appendChild(empty);
      return;
    }
    for (const chat of chats) {
      const item = document.createElement("div");
      item.className = "chat-item" + (chat.id === currentChatId ? " active" : "");
      item.dataset.id = chat.id;

      const title = document.createElement("div");
      title.className = "chat-item-title";
      title.textContent = chat.title || "New chat";

      const meta = document.createElement("div");
      meta.className = "chat-item-meta";
      meta.textContent = chat.preview
        ? `${relativeTime(chat.created_at)} · ${chat.preview}`
        : relativeTime(chat.created_at);

      const del = document.createElement("button");
      del.className = "chat-delete";
      del.setAttribute("aria-label", "Delete chat");
      del.innerHTML =
        '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M2 4h10M5.5 4V2.5h3V4M3.5 4l.6 8h5.8l.6-8" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>';
      del.addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!confirm(`Delete "${chat.title}"? This can't be undone.`)) return;
        await deleteChat(chat.id);
        if (currentChatId === chat.id) {
          currentChatId = null;
          showEmptyState();
        }
        await refreshChats();
      });

      item.appendChild(title);
      item.appendChild(meta);
      item.appendChild(del);
      item.addEventListener("click", () => selectChat(chat.id));
      chatListEl.appendChild(item);
    }
  }

  async function refreshChats() {
    chats = await getChats();
    renderChatList();
  }

  // ---------- Chat view ----------

  function showEmptyState() {
    emptyStateEl.style.display = "flex";
    messagesEl.style.display = "none";
    messagesEl.innerHTML = "";
    renderChatList();
  }

  function showMessages() {
    emptyStateEl.style.display = "none";
    messagesEl.style.display = "flex";
  }

  async function selectChat(id) {
    currentChatId = id;
    closeSidebarOnMobile();
    const chat = await getChat(id);
    showMessages();
    messagesEl.innerHTML = "";
    for (const msg of chat.messages) {
      renderMessage(msg);
    }
    scrollToBottom();
    renderChatList();

    for (const msg of chat.messages) {
      if (msg.role === "assistant" && msg.status === "generating") {
        pollMessage(msg.id);
      }
    }
  }

  function scrollToBottom() {
    const view = document.getElementById("chatView");
    view.scrollTop = view.scrollHeight;
  }

  function renderMessage(msg) {
    const wrap = document.createElement("div");
    wrap.className = `msg ${msg.role}`;
    wrap.dataset.msgId = msg.id;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = msg.text;
    wrap.appendChild(bubble);

    if (msg.role === "assistant") {
      const attach = document.createElement("div");
      attach.className = "attach-slot";
      wrap.appendChild(attach);
      updateAttachment(attach, msg);
    }

    const time = document.createElement("div");
    time.className = "msg-time";
    time.textContent = clockTime(msg.created_at);
    wrap.appendChild(time);

    messagesEl.appendChild(wrap);
    return wrap;
  }

  function updateAttachment(slotEl, msg) {
    slotEl.innerHTML = "";
    if (msg.status === "generating") {
      slotEl.appendChild(buildGeneratingIndicator());
    } else if (msg.status === "error") {
      const err = document.createElement("div");
      err.className = "generating-error";
      err.textContent = "Couldn't generate audio for this one.";
      slotEl.appendChild(err);
    } else if (msg.audio_path) {
      slotEl.appendChild(buildAudioPlayer(msg.audio_path));
    }
  }

  function buildGeneratingIndicator() {
    const el = document.createElement("div");
    el.className = "generating";
    el.innerHTML = `
      <div class="eq-bars"><span></span><span></span><span></span><span></span><span></span></div>
      <div class="generating-label">Composing your track…</div>
    `;
    return el;
  }

  function buildAudioPlayer(src) {
    const card = document.createElement("div");
    card.className = "audio-card";

    const audio = new Audio(src);
    audio.preload = "metadata";

    const playBtn = document.createElement("button");
    playBtn.className = "play-btn";
    playBtn.setAttribute("aria-label", "Play");
    const playIcon = '<svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor"><path d="M3 1.5l10 5.5-10 5.5V1.5z"/></svg>';
    const pauseIcon = '<svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor"><rect x="3" y="2" width="3" height="10" rx="0.5"/><rect x="8" y="2" width="3" height="10" rx="0.5"/></svg>';
    playBtn.innerHTML = playIcon;

    const body = document.createElement("div");
    body.className = "audio-body";

    const track = document.createElement("div");
    track.className = "audio-track";
    const fill = document.createElement("div");
    fill.className = "audio-track-fill";
    track.appendChild(fill);

    const timeLabel = document.createElement("div");
    timeLabel.className = "audio-time";
    timeLabel.textContent = "0:00 / 0:30";

    body.appendChild(track);
    body.appendChild(timeLabel);

    function fmt(sec) {
      if (!isFinite(sec)) return "0:00";
      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60).toString().padStart(2, "0");
      return `${m}:${s}`;
    }

    playBtn.addEventListener("click", () => {
      document.querySelectorAll("audio").forEach((a) => {
        if (a !== audio) a.pause();
      });
      if (audio.paused) {
        audio.play();
      } else {
        audio.pause();
      }
    });

    audio.addEventListener("play", () => { playBtn.innerHTML = pauseIcon; });
    audio.addEventListener("pause", () => { playBtn.innerHTML = playIcon; });
    audio.addEventListener("timeupdate", () => {
      const pct = audio.duration ? (audio.currentTime / audio.duration) * 100 : 0;
      fill.style.width = `${pct}%`;
      timeLabel.textContent = `${fmt(audio.currentTime)} / ${fmt(audio.duration || 30)}`;
    });
    audio.addEventListener("loadedmetadata", () => {
      timeLabel.textContent = `0:00 / ${fmt(audio.duration)}`;
    });
    audio.addEventListener("ended", () => {
      playBtn.innerHTML = playIcon;
      fill.style.width = "0%";
    });

    track.addEventListener("click", (e) => {
      const rect = track.getBoundingClientRect();
      const pct = (e.clientX - rect.left) / rect.width;
      if (audio.duration) audio.currentTime = pct * audio.duration;
    });

    card.appendChild(playBtn);
    card.appendChild(body);
    return card;
  }

  // ---------- Polling for in-flight generation ----------

  function pollMessage(messageId) {
    if (pollTimers.has(messageId)) return;
    const timer = setInterval(async () => {
      try {
        const msg = await getMessage(messageId);
        if (msg.status !== "generating") {
          clearInterval(timer);
          pollTimers.delete(messageId);
          const wrap = messagesEl.querySelector(`[data-msg-id="${messageId}"]`);
          if (wrap) {
            const slot = wrap.querySelector(".attach-slot");
            if (slot) updateAttachment(slot, msg);
          }
        }
      } catch {
        clearInterval(timer);
        pollTimers.delete(messageId);
      }
    }, 2500);
    pollTimers.set(messageId, timer);
  }

  // ---------- Sending ----------

  async function handleSend(e) {
    e.preventDefault();
    const text = inputEl.value.trim();
    if (!text) return;

    inputEl.value = "";
    autoResize();
    sendBtn.disabled = true;

    try {
      if (currentChatId === null) {
        const chat = await createChat();
        currentChatId = chat.id;
        await refreshChats();
      }

      showMessages();
      const optimisticUser = {
        id: `tmp-${Date.now()}`,
        role: "user",
        text,
        created_at: new Date().toISOString(),
      };
      renderMessage(optimisticUser);
      scrollToBottom();

      const result = await postMessage(currentChatId, text);

      // Replace optimistic bubble with the real one (has a real id).
      const tmpEl = messagesEl.querySelector(`[data-msg-id="${optimisticUser.id}"]`);
      if (tmpEl) tmpEl.remove();
      renderMessage(result.user_message);
      renderMessage(result.assistant_message);
      scrollToBottom();

      if (result.assistant_message.status === "generating") {
        pollMessage(result.assistant_message.id);
      }

      await refreshChats();
    } catch (err) {
      alert(`Something went wrong: ${err.message}`);
    } finally {
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }

  function autoResize() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
  }

  // ---------- Sidebar mobile toggle ----------

  function closeSidebarOnMobile() {
    sidebar.classList.remove("open");
    sidebarScrim.classList.remove("visible");
    sidebarToggle.classList.remove("hidden");
  }

  sidebarToggle.addEventListener("click", () => {
    const opening = !sidebar.classList.contains("open");
    sidebar.classList.toggle("open", opening);
    sidebarScrim.classList.toggle("visible", opening);
    sidebarToggle.classList.toggle("hidden", opening);
  });
  sidebarScrim.addEventListener("click", closeSidebarOnMobile);

  // ---------- Wiring ----------

  newChatBtn.addEventListener("click", () => {
    currentChatId = null;
    showEmptyState();
    closeSidebarOnMobile();
    inputEl.focus();
  });

  composerEl.addEventListener("submit", handleSend);
  inputEl.addEventListener("input", autoResize);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend(e);
    }
  });

  // ---------- Init ----------

  (async function init() {
    await refreshChats();
    if (chats.length > 0) {
      await selectChat(chats[0].id);
    } else {
      showEmptyState();
    }
  })();
})();
