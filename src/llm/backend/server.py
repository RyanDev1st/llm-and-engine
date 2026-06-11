"""Stdlib HTTP server for the chess-coach web app (no extra deps).

Single shared board; drag-moves (/api/move) and chat tool calls mutate the same
game, the frontend re-syncs from /api/state. The App (board + model loops) lives
in web_app.py. /api/chat takes an optional `variant` ("sft"|"base"|"both") so the
demo can compare our SFT adapter against the untrained base side by side.

Run from repo root:  python -m backend.server [adapter_dir]
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import skill_admin
from .web_app import App

WEB = Path(__file__).resolve().parents[1] / "gemma_chat_site" / "static"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8"}


def bind_address() -> tuple[str, int]:
    return os.environ.get("CHESS_HOST", "127.0.0.1"), int(os.environ.get("CHESS_PORT", "7860"))


APP: App


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            return self._json(APP.state())
        if path == "/api/skills":
            return self._json(APP.skills_payload())
        if path in ("/", "/index.html"):
            return self._file(WEB / "index.html")
        if path.startswith("/static/"):
            return self._file(WEB / path.removeprefix("/static/"))
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._read()
            if path == "/api/move":
                ok = APP.game.move_uci(str(body.get("uci", "")))
                return self._json({"ok": ok, "state": APP.state()}, 200 if ok else 400)
            if path == "/api/sync":
                # The client board is authoritative (smooth, no per-move round-trip);
                # mirror it here before a chat turn. Normal play sends a move list
                # (replayed -> preserves history for review_move/undo); a FEN-loaded
                # position (puzzle/paste) sends fen (history starts fresh).
                fen = str(body.get("fen", "")).strip()
                moves = body.get("moves", [])
                if fen:
                    ok = APP.game.load_fen(fen)
                else:
                    ok = APP.game.load_uci_moves([str(m) for m in moves] if isinstance(moves, list) else [])
                APP.executor.game = APP.game
                return self._json({"ok": ok, "state": APP.state()}, 200 if ok else 400)
            if path == "/api/reset":
                return self._json({"ok": True, "state": APP.reset()})
            if path == "/api/chat":
                msg = str(body.get("message", "")).strip()
                if not msg:
                    raise ValueError("empty message")
                return self._json({"ok": True, **APP.chat(msg, str(body.get("variant", "sft")))})
            if path == "/api/skill":
                skill_admin.add_skill(str(body.get("name", "")), str(body.get("description", "")),
                                      str(body.get("body", "")))
                return self._json({"ok": True, **APP.skills_payload()})
            if path == "/api/skill/delete":
                skill_admin.delete_skill(str(body.get("name", "")))
                return self._json({"ok": True, **APP.skills_payload()})
            if path == "/api/plugin":
                skill_admin.apply_plugin(APP.plugin_context, body)
                return self._json({"ok": True, **APP.skills_payload()})
            self.send_error(404)
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)}, 500)

    def _read(self) -> dict:
        n = int(self.headers.get("content-length", "0"))
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Path) -> None:
        p = path.resolve()
        if not (p.exists() and p.is_file()) or WEB not in p.parents and p != WEB / "index.html":
            return self.send_error(404)
        data = p.read_bytes()
        self.send_response(200)
        self.send_header("content-type", TYPES.get(p.suffix, "application/octet-stream"))
        self.send_header("content-length", str(len(data)))
        self.send_header("Cache-Control", "no-store, must-revalidate")  # dev: always fresh
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        return


def main() -> None:
    global APP
    skill_admin.register()  # runtime skills dir on CHESS_SKILLS_DIRS, wiped fresh
    adapter = sys.argv[1] if len(sys.argv) > 1 else None
    APP = App(adapter)
    print(f"loading model (adapter={adapter}) ...", flush=True)
    APP.load_model()
    host, port = bind_address()
    print(f"open http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
