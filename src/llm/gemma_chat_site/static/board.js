// Interactive board: renders from FEN, click-to-move with legal-target hints.
// Board is display-only state; the backend stays authoritative.
const GLYPH = { p:"♟", n:"♞", b:"♝", r:"♜", q:"♛", k:"♚",
                P:"♙", N:"♘", B:"♗", R:"♖", Q:"♕", K:"♔" };
const FILES = ["a","b","c","d","e","f","g","h"];

function fenBoard(fen) {
  const rows = fen.split(" ")[0].split("/");
  const grid = [];
  for (const row of rows) {
    const r = [];
    for (const ch of row) {
      if (/\d/.test(ch)) { for (let i = 0; i < +ch; i++) r.push(""); }
      else r.push(ch);
    }
    grid.push(r);
  }
  return grid; // grid[0] = rank 8
}

const Board = {
  el: null, selected: null, legal: [], turn: "white", lastMove: null, onMove: null,
  init(el, onMove) { this.el = el; this.onMove = onMove; },

  render(state) {
    this.legal = state.legal || [];
    this.turn = state.turn;
    this.lastMove = state.last_move;
    const grid = fenBoard(state.fen);
    this.el.innerHTML = "";
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const sqName = FILES[col] + (8 - row);
        const piece = grid[row][col];
        const sq = document.createElement("div");
        sq.className = "sq " + ((row + col) % 2 === 0 ? "light" : "dark");
        sq.dataset.sq = sqName;
        if (this.lastMove && (this.lastMove.slice(0,2) === sqName || this.lastMove.slice(2,4) === sqName))
          sq.classList.add("last");
        if (this.selected === sqName) sq.classList.add("sel");
        if (piece) {
          const span = document.createElement("span");
          span.className = "piece " + (piece === piece.toUpperCase() ? "white-piece" : "black-piece");
          span.textContent = GLYPH[piece];
          sq.appendChild(span);
        }
        if (this.selected) {
          const mv = this._legalFrom(this.selected).find(m => m.slice(2,4) === sqName);
          if (mv) { if (piece) sq.classList.add("cap");
            const dot = document.createElement("span"); dot.className = "dot"; sq.appendChild(dot); }
        }
        if (col === 0) this._coord(sq, 8 - row);
        if (row === 7) this._coord(sq, FILES[col]);
        sq.addEventListener("click", () => this._click(sqName, piece));
        this.el.appendChild(sq);
      }
    }
  },

  _coord(sq, label) {
    const c = document.createElement("span"); c.className = "coord"; c.textContent = label;
    sq.appendChild(c);
  },

  _legalFrom(sqName) { return this.legal.filter(m => m.slice(0, 2) === sqName); },

  _click(sqName, piece) {
    if (this.selected) {
      const moves = this._legalFrom(this.selected);
      let mv = moves.find(m => m.slice(2, 4) === sqName);
      if (mv) {
        if (mv.length === 4 && this._isPromotion(this.selected, sqName)) mv += "q";
        this.selected = null;
        this.onMove(mv);
        return;
      }
    }
    // select a square that has legal moves
    if (this._legalFrom(sqName).length) { this.selected = sqName; }
    else { this.selected = null; }
    this._refreshSelection();
  },

  _isPromotion(from, to) {
    return (to[1] === "8" || to[1] === "1");
  },

  _refreshSelection() {
    // cheap re-render of selection by re-rendering current grid
    const sqEls = this.el.querySelectorAll(".sq");
    sqEls.forEach(s => { s.classList.remove("sel"); s.querySelectorAll(".dot").forEach(d => d.remove()); s.classList.remove("cap"); });
    if (!this.selected) return;
    this.el.querySelector(`[data-sq="${this.selected}"]`)?.classList.add("sel");
    for (const mv of this._legalFrom(this.selected)) {
      const t = this.el.querySelector(`[data-sq="${mv.slice(2,4)}"]`);
      if (!t) continue;
      if (t.querySelector(".piece")) t.classList.add("cap");
      const dot = document.createElement("span"); dot.className = "dot"; t.appendChild(dot);
    }
  },
};
