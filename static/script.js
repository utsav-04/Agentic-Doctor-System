// api/static/script.js
// Talks to POST /session and POST /chat on the same origin.

const API_BASE = "";

let sessionId = null;
let currentTool = "";     // "" = Auto, else "medicine" | "lab" | "doctor"
let awaiting = null;      // null | "location" | "follow_up" — mirrors the backend's session state
let sending = false;

const el = {
  hero: document.getElementById("hero"),
  messageList: document.getElementById("messageList"),
  chatScroll: document.getElementById("chatScroll"),
  messageInput: document.getElementById("messageInput"),
  sendBtn: document.getElementById("sendBtn"),
  modeRow: document.getElementById("modeRow"),
  composerHint: document.getElementById("composerHint"),
  statusPill: document.getElementById("statusPill"),
  newChatBtn: document.getElementById("newChatBtn"),
  sidebar: document.getElementById("sidebar"),
  sidebarToggle: document.getElementById("sidebarToggle"),
};

const MODE_HINTS = {
  "": "Auto mode — I'll decide which tool(s) to use",
  medicine: "Medicine mode — you'll get medicine info only",
  lab: "Lab mode — you'll get lab test suggestions only",
  doctor: "Doctor mode — you'll get doctor/hospital results only",
};

init();

async function init() {
  if (window.innerWidth <= 720) el.sidebar.classList.add("hidden");
  await startNewSession();

  el.sendBtn.addEventListener("click", sendMessage);
  el.messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  el.messageInput.addEventListener("input", autoGrow);

  el.modeRow.addEventListener("click", (e) => {
    const btn = e.target.closest(".mode-pill");
    if (!btn) return;
    selectMode(btn);
  });

  document.querySelectorAll(".suggestion-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      el.messageInput.value = chip.dataset.fill;
      autoGrow();
      el.messageInput.focus();
    });
  });

  el.newChatBtn.addEventListener("click", startNewSession);
  el.sidebarToggle.addEventListener("click", () => el.sidebar.classList.toggle("hidden"));
}

async function startNewSession() {
  setStatus("connecting");
  try {
    const res = await fetch(`${API_BASE}/session`, { method: "POST" });
    const data = await res.json();
    sessionId = data.session_id;
  } catch (err) {
    setStatus("offline");
    console.error("Could not create session:", err);
    return;
  }

  awaiting = null;
  el.messageList.innerHTML = "";
  el.messageList.classList.remove("visible");
  el.hero.style.display = "";
  resetModeToAuto();
  setStatus("ready");
  el.messageInput.value = "";
  autoGrow();
}

function selectMode(btn) {
  document.querySelectorAll(".mode-pill").forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  currentTool = btn.dataset.tool;
  el.composerHint.textContent = MODE_HINTS[currentTool] ?? "";
  el.composerHint.classList.remove("waiting");
}

function resetModeToAuto() {
  const autoBtn = el.modeRow.querySelector('[data-tool=""]');
  if (autoBtn) selectMode(autoBtn);
}

function autoGrow() {
  el.messageInput.style.height = "auto";
  el.messageInput.style.height = Math.min(el.messageInput.scrollHeight, 160) + "px";
}

async function sendMessage() {
  const text = el.messageInput.value.trim();
  if (!text || sending || !sessionId) return;

  if (el.hero.style.display !== "none") {
    el.hero.style.display = "none";
    el.messageList.classList.add("visible");
  }

  addBubble("user", text);
  el.messageInput.value = "";
  autoGrow();
  sending = true;
  el.sendBtn.disabled = true;
  setStatus("busy");

  const typingId = addTyping();

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message: text,
        tool: currentTool || null,
      }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    removeTyping(typingId);
    addBubble("assistant", data.reply, data.criticality);

    awaiting = data.awaiting || null;
    applyAwaitingHint();
  } catch (err) {
    removeTyping(typingId);
    addBubble("assistant", "Sorry, something went wrong reaching the server. Please try again.");
    console.error(err);
  } finally {
    sending = false;
    el.sendBtn.disabled = false;
    setStatus("ready");
    el.messageInput.focus();
  }
}

function applyAwaitingHint() {
  if (awaiting === "location") {
    el.composerHint.textContent = "Waiting for your city and state (e.g. 'Mumbai, Maharashtra')";
    el.composerHint.classList.add("waiting");
    setStatus("waiting");
  } else if (awaiting === "follow_up") {
    el.composerHint.textContent = "Reply with 'lab', 'doctor', 'both', or 'no'";
    el.composerHint.classList.add("waiting");
    setStatus("waiting");
  } else {
    el.composerHint.textContent = MODE_HINTS[currentTool] ?? "";
    el.composerHint.classList.remove("waiting");
    setStatus("ready");
  }
}

function addBubble(role, text, criticality) {
  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";

  if (role === "assistant" && criticality) {
    const badge = document.createElement("span");
    badge.className = `badge ${criticality}`;
    badge.textContent = criticality;
    bubble.appendChild(badge);
    bubble.appendChild(document.createTextNode(text));
  } else {
    bubble.textContent = text;
  }

  row.appendChild(bubble);
  el.messageList.appendChild(row);
  scrollToBottom();
}

let typingCounter = 0;
function addTyping() {
  const id = `typing-${++typingCounter}`;
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  row.id = id;
  row.innerHTML = `<div class="msg-bubble"><span class="typing-dots"><span></span><span></span><span></span></span></div>`;
  el.messageList.appendChild(row);
  scrollToBottom();
  return id;
}

function removeTyping(id) {
  const node = document.getElementById(id);
  if (node) node.remove();
}

function scrollToBottom() {
  el.chatScroll.scrollTop = el.chatScroll.scrollHeight;
}

function setStatus(state) {
  el.statusPill.textContent = state;
  el.statusPill.classList.remove("busy", "waiting");
  if (state === "busy") el.statusPill.classList.add("busy");
  if (state === "waiting") el.statusPill.classList.add("waiting");
}