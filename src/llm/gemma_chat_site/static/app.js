// Wires the board, eval bar, move list, and chat to the backend.
const boardEl = document.getElementById("board");
const fill = document.getElementById("evalfill");
const evalText = document.getElementById("evaltext");
const turnPill = document.getElementById("turnpill");
const moveList = document.getElementById("movelist");
const messages = document.getElementById("messages");
const messagesBase = document.getElementById("messages-base");
const basecol = document.getElementById("basecol");
const cmptoggle = document.getElementById("cmptoggle");
const form = document.getElementById("chatform");
const input = document.getElementById("chatinput");
const sendBtn = form.querySelector(".send");
let chatBusy = false;
let compareMode = false;

async function api(path, body) {
  const opt = body ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }
                    : {};
  const r = await fetch(path, opt);
  return r.json();
}

function renderState(state) {
  Board.render(state);
  const ev = state.eval || { bar: 50, text: "0.00" };
  fill.style.height = ev.bar + "%";
  evalText.textContent = ev.text;
  if (state.game_over) turnPill.textContent = ev.text.includes("draw") || ev.kind === "over"
    ? "Game over" : "Checkmate";
  else turnPill.textContent = (state.turn === "white" ? "White" : "Black") + " to move"
    + (state.in_check ? " · check!" : "");
  moveList.textContent = pairMoves(state.history);
}

function pairMoves(sans) {
  let out = "";
  for (let i = 0; i < sans.length; i += 2)
    out += `${i / 2 + 1}. ${sans[i]} ${sans[i + 1] || ""} `;
  return out.trim();
}

function addMsgTo(container, text, cls) {
  const d = document.createElement("div");
  d.className = "msg " + cls; d.textContent = text;
  container.appendChild(d); container.scrollTop = container.scrollHeight;
  return d;
}

function addLoadingTo(container) {
  const d = document.createElement("div");
  d.className = "msg bot loading";
  d.innerHTML = `<div class="skeleton">
    <span class="skel-line"></span><span class="skel-line"></span><span class="skel-line short"></span>
  </div>`;
  container.appendChild(d); container.scrollTop = container.scrollHeight;
  return d;
}

const addMsg = (text, cls) => addMsgTo(messages, text, cls);

function renderReply(container, payload, loadingEl) {
  if (loadingEl) loadingEl.remove();
  const calls = payload.tool_calls || (payload.tool_call ? [payload.tool_call] : []);
  const results = payload.tool_results || (payload.tool_result ? [payload.tool_result] : []);
  calls.forEach((c, i) => addMsgTo(container, `🔧 ${c}  →  ${results[i] || ""}`, "tool"));
  addMsgTo(container, payload.reply || "(no reply)", "bot");
}

function setBusy(busy) {
  chatBusy = busy;
  input.disabled = busy;
  sendBtn.disabled = busy;
  form.classList.toggle("busy", busy);
}

async function refresh() { renderState(await api("/api/state")); }

async function onMove(uci) {
  const res = await api("/api/move", { uci });
  if (res.state) renderState(res.state);
}

async function sendChat(message) {
  if (chatBusy) return;
  setBusy(true);
  addMsgTo(messages, message, "user");
  const loadSft = addLoadingTo(messages);
  let loadBase = null;
  if (compareMode) { addMsgTo(messagesBase, message, "user"); loadBase = addLoadingTo(messagesBase); }
  try {
    const res = await api("/api/chat", { message, variant: compareMode ? "both" : "sft" });
    if (compareMode && res.sft) {
      renderReply(messages, res.sft, loadSft);
      renderReply(messagesBase, res.base, loadBase);
    } else {
      renderReply(messages, res, loadSft);
    }
    if (res.state) renderState(res.state);
  } catch (e) {
    if (loadSft) loadSft.remove();
    if (loadBase) loadBase.remove();
    addMsgTo(messages, "Sorry, something went wrong reaching the coach.", "bot");
  } finally {
    setBusy(false);
    input.focus();
  }
}

Board.init(boardEl, onMove);

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const msg = input.value.trim();
  if (!msg || chatBusy) return;
  input.value = "";
  sendChat(msg);
});

document.getElementById("reset").addEventListener("click", async () => {
  const res = await api("/api/reset", {});
  messages.innerHTML = "";
  messagesBase.innerHTML = "";
  addMsg("New game started. Make a move on the board or ask me anything!", "bot");
  if (compareMode) addMsgTo(messagesBase, "Base model ready (no training).", "bot");
  renderState(res.state);
});

cmptoggle.addEventListener("change", () => {
  compareMode = cmptoggle.checked;
  basecol.hidden = !compareMode;
  if (compareMode && !messagesBase.childElementCount)
    addMsgTo(messagesBase, "Base Gemma — same harness + skills, no SFT. Ask anything to compare.", "bot");
});

// ---- Skills & plugins panel ------------------------------------------------
const skItems = document.getElementById("sk-items");
const skCount = document.getElementById("sk-count");
const skMsg = document.getElementById("sk-msg");

function renderCatalog(payload) {
  const skills = payload.skills || [];
  skCount.textContent = `(${skills.length})`;
  skItems.innerHTML = "";
  skills.forEach((s) => {
    const li = document.createElement("li");
    const del = s.runtime ? `<button class="del" data-name="${s.name}" title="remove">✕</button>` : "";
    li.innerHTML = `${del}<span class="nm">${s.name}</span>${s.runtime ? '<span class="rt">runtime</span>' : ""}<br>${s.description}`;
    skItems.appendChild(li);
  });
  skItems.querySelectorAll(".del").forEach((b) =>
    b.addEventListener("click", async () => {
      await api("/api/skill/delete", { name: b.dataset.name });
      loadCatalog();
    }));
  const pc = payload.plugin_context || {};
  document.getElementById("pl-installed").value = (pc.installed || []).join(", ");
  document.getElementById("pl-enabled").value = (pc.enabled || []).join(", ");
  document.getElementById("pl-market").value = (pc.marketplace || []).join(", ");
}

async function loadCatalog() { renderCatalog(await api("/api/skills")); }

document.getElementById("sk-add").addEventListener("click", async () => {
  const name = document.getElementById("sk-name").value.trim();
  const description = document.getElementById("sk-desc").value.trim();
  const body = document.getElementById("sk-body").value;
  if (!name || !description) { skMsg.textContent = "name + description required"; skMsg.style.color = "#e07a7a"; return; }
  const res = await api("/api/skill", { name, description, body });
  if (res.ok) {
    skMsg.textContent = `added — now ask something it should route to, and watch for load_skill`;
    skMsg.style.color = "#7fd1a0";
    document.getElementById("sk-name").value = "";
    document.getElementById("sk-desc").value = "";
    document.getElementById("sk-body").value = "";
    renderCatalog(res);
  } else { skMsg.textContent = res.error || "failed"; skMsg.style.color = "#e07a7a"; }
});

document.getElementById("pl-apply").addEventListener("click", async () => {
  const res = await api("/api/plugin", {
    installed: document.getElementById("pl-installed").value,
    enabled: document.getElementById("pl-enabled").value,
    marketplace: document.getElementById("pl-market").value,
  });
  if (res.ok) { skMsg.textContent = "plugin context updated"; skMsg.style.color = "#7fd1a0"; renderCatalog(res); }
});

addMsg("Hi! I'm your chess coach. Drag pieces to play, or ask me things like \"how's my position?\" or \"what should I play?\"", "bot");
refresh();
loadCatalog();
