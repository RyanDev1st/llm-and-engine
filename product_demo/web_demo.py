from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from chess_tool_demo import Board, run_tool_turn

BOARD = Board.from_fen()

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Engine-backed chess coach demo</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; background: #101418; color: #eef2f7; }
    main { max-width: 1100px; margin: 0 auto; display: grid; gap: 18px; grid-template-columns: 360px 1fr; }
    h1 { grid-column: 1 / -1; margin: 0; }
    .board { display: grid; grid-template-columns: repeat(8, 42px); grid-template-rows: repeat(8, 42px); border: 2px solid #3c4654; width: 336px; }
    .sq { display: grid; place-items: center; font-size: 25px; font-weight: 700; }
    .light { background: #e3d6bd; color: #1c232b; }
    .dark { background: #789065; color: #101418; }
    .panel { background: #18202a; border: 1px solid #2b3542; border-radius: 12px; padding: 16px; }
    .controls { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; }
    input, button, select { font: inherit; border-radius: 8px; border: 1px solid #394656; padding: 8px 10px; }
    input, select { background: #0e1319; color: #eef2f7; }
    button { background: #4f7cff; color: white; cursor: pointer; }
    button.secondary { background: #2c3644; }
    pre { white-space: pre-wrap; background: #0e1319; border-radius: 8px; padding: 12px; min-height: 100px; }
    .muted { color: #9aa8ba; }
  </style>
</head>
<body>
  <main>
    <h1>Chess engine + SFT tool-call demo</h1>
    <section class="panel">
      <div id="board" class="board"></div>
      <p id="status" class="muted"></p>
      <div class="controls">
        <input id="uci" value="e2e4" aria-label="UCI move">
        <button onclick="callTool('move')">Move</button>
        <button onclick="callTool('review_move')">Review</button>
        <button onclick="callTool('undo')" class="secondary">Undo</button>
        <button onclick="resetBoard()" class="secondary">Reset</button>
      </div>
    </section>
    <section class="panel">
      <h2>Router → Tool → Narrator</h2>
      <div class="controls">
        <button onclick="callTool('legal_moves')" class="secondary">Legal moves</button>
        <button onclick="callTool('best_move')" class="secondary">Best move</button>
        <button onclick="callTool('eval')" class="secondary">Eval</button>
      </div>
      <pre id="trace">Ready. Try e2e4, e7e5, then review g1f3.</pre>
      <h2>SFT shape</h2>
      <pre>{"role":"assistant","tool_call":{"tool_name":"move","arguments":{"uci":"e2e4"}}}
{"role":"tool","name":"move","content":{"status":"ok","move":"e2e4"}}
{"role":"assistant","content":"Move accepted. It is Black to move."}</pre>
    </section>
  </main>
  <script>
    const pieces = {K:'♔',Q:'♕',R:'♖',B:'♗',N:'♘',P:'♙',k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟'};

    async function request(path, body) {
      const response = await fetch(path, {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(body || {})});
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    async function refresh() {
      const response = await fetch('/state');
      render(await response.json());
    }

    function render(state) {
      const board = document.getElementById('board');
      board.innerHTML = '';
      state.board.flat().forEach((piece, index) => {
        const rank = Math.floor(index / 8);
        const file = index % 8;
        const square = document.createElement('div');
        square.className = `sq ${(rank + file) % 2 === 0 ? 'light' : 'dark'}`;
        square.textContent = pieces[piece] || '';
        board.appendChild(square);
      });
      document.getElementById('status').textContent = `Side: ${state.turn}. Eval: ${state.evaluation.bucket} (${state.evaluation.score_cp_white} cp). Legal moves: ${state.legal_moves.length}.`;
    }

    async function callTool(toolName) {
      const uci = document.getElementById('uci').value.trim();
      const args = ['move', 'review_move'].includes(toolName) ? {uci} : {};
      const result = await request('/tool', {tool_name: toolName, arguments: args});
      document.getElementById('trace').textContent = `router -> ${JSON.stringify({tool_name: toolName, arguments: args})}\ntool <- ${JSON.stringify(result.tool_result, null, 2)}\nnarrator -> ${result.narration}`;
      render(result.state);
    }

    async function resetBoard() {
      const result = await request('/reset');
      document.getElementById('trace').textContent = 'Board reset.';
      render(result);
    }

    refresh();
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))
            return
        if self.path == "/state":
            self.write_json(BOARD.snapshot())
            return
        self.send_error(404)

    def do_POST(self) -> None:
        global BOARD
        if self.path == "/reset":
            BOARD = Board.from_fen()
            self.write_json(BOARD.snapshot())
            return
        if self.path == "/tool":
            body = self.read_json()
            name = str(body.get("tool_name", ""))
            args = body.get("arguments") if isinstance(body.get("arguments"), dict) else {}
            self.write_json(run_tool_turn(BOARD, name, args))
            return
        self.send_error(404)

    def read_json(self) -> dict:
        length = int(self.headers.get("content-length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def write_json(self, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Open http://127.0.0.1:8765")
    server.serve_forever()


if __name__ == "__main__":
    main()
