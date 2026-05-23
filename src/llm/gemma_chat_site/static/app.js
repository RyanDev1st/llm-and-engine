// Wires the board, eval bar, move list, and chat to the backend.
const boardEl = document.getElementById("board");
const fill = document.getElementById("evalfill");
const evalText = document.getElementById("evaltext");
const turnPill = document.getElementById("turnpill");
const moveList = document.getElementById("movelist");
const messages = document.getElementById("messages");
const typing = document.getElementById("typing");

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

function addMsg(text, cls) {
  const d = document.createElement("div");
  d.className = "msg " + cls; d.textContent = text;
  messages.appendChild(d); messages.scrollTop = messages.scrollHeight;
  return d;
}

async function refresh() { renderState(await api("/api/state")); }

async function onMove(uci) {
  const res = await api("/api/move", { uci });
  if (res.state) renderState(res.state);
}

async function sendChat(message) {
  addMsg(message, "user");
  typing.classList.remove("hidden");
  try {
    const res = await api("/api/chat", { message });
    if (res.tool_call) addMsg(`🔧 ${res.tool_call}  →  ${res.tool_result}`, "tool");
    addMsg(res.reply || "(no reply)", "bot");
    if (res.state) renderState(res.state);
  } catch (e) {
    addMsg("Sorry, something went wrong reaching the coach.", "bot");
  } finally {
    typing.classList.add("hidden");
  }
}

Board.init(boardEl, onMove);

document.getElementById("chatform").addEventListener("submit", (e) => {
  e.preventDefault();
  const input = document.getElementById("chatinput");
  const msg = input.value.trim();
  if (!msg) return;
  input.value = "";
  sendChat(msg);
});

document.getElementById("reset").addEventListener("click", async () => {
  const res = await api("/api/reset", {});
  messages.innerHTML = "";
  addMsg("New game started. Make a move on the board or ask me anything!", "bot");
  renderState(res.state);
});

addMsg("Hi! I'm your chess coach. Drag pieces to play, or ask me things like \"how's my position?\" or \"what should I play?\"", "bot");
refresh();
