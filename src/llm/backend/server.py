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
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


def chunk_reply(text: str) -> list[str]:
    """Split a finished reply into sentence/clause chunks for paced streaming reveal.
    Boundary = . ! ? FOLLOWED BY whitespace (so decimals like +0.34 stay intact); a long
    sentence is further broken on , ; : — not letter-by-letter, not all-at-once."""
    text = (text or "").strip()
    # boundary = .!? + whitespace, but NOT after a digit-period ("1." "2." list markers,
    # and decimals already lack the trailing space) so numbered move lists stay intact.
    sents = [s for s in re.split(r"(?<=[.!?])(?<![0-9]\.)\s+", text) if s.strip()]
    chunks: list[str] = []
    for s in sents:
        if len(s) <= 90:
            chunks.append(s)
        else:
            chunks += [c for c in re.split(r"(?<=[,;:])\s+", s) if c.strip()]
    return chunks

from . import client_registry, skill_admin
from .web_app import load_shared_model

WEB = Path(__file__).resolve().parents[1] / "gemma_chat_site" / "static"
BACKEND_DIR = Path(__file__).resolve().parent
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8"}


def _dev_reload_token() -> float:
    """Newest mtime across the served frontend + backend code. The page polls this in
    dev; when it changes (you saved a file — dev_serve also restarts the app on backend
    saves) the browser reloads itself. Cheap; harmless in prod (just a number)."""
    files = list(WEB.glob("*.html")) + list(WEB.glob("*.js")) + list(WEB.glob("*.css")) \
        + list(BACKEND_DIR.glob("*.py"))
    return max((p.stat().st_mtime for p in files), default=0.0)


def bind_address() -> tuple[str, int]:
    return os.environ.get("CHESS_HOST", "127.0.0.1"), int(os.environ.get("CHESS_PORT", "7860"))


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.1 so the SSE stream uses chunked transfer encoding — under the default
    # HTTP/1.0 (no Content-Length) the browser buffers the whole body until the
    # connection closes, which made token streaming "snap in" as one block at the end.
    protocol_version = "HTTP/1.1"

    def _client(self):
        """Resolve THIS browser's per-client App from the cookie cid (minting one on first visit).
        Every request operates on its own board/history/sessions — no cross-user leakage."""
        cid = self._cookie("chess_cid")
        if not client_registry.valid_cid(cid):
            cid = client_registry.new_cid()
            self._new_cid = cid                  # _json/_file/_chat_stream set the Set-Cookie header
        self._cid = cid
        return client_registry.get_client(cid)

    def _cookie(self, name: str) -> str:
        for part in (self.headers.get("Cookie", "") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == name:
                return v
        return ""

    def _maybe_set_cookie(self) -> None:
        cid = getattr(self, "_new_cid", None)
        if cid:                                  # HttpOnly: the page doesn't read it; sent automatically
            self.send_header("Set-Cookie",
                             f"chess_cid={cid}; Path=/; Max-Age=31536000; SameSite=Lax; HttpOnly")
            self._new_cid = None

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/dev/reload-token":  # live-reload: newest mtime of static+backend (no client)
            return self._json({"token": _dev_reload_token()})
        if path in ("/", "/index.html"):
            self._client()                     # mint the per-browser cid cookie on first page load
            return self._file(WEB / "index.html")
        if path.startswith("/static/"):
            return self._file(WEB / path.removeprefix("/static/"))
        app = self._client()
        if path == "/api/state":
            return self._json(app.state())
        if path == "/api/skills":
            return self._json(app.skills_payload())
        if path == "/api/sessions":            # the sidebar's session list (+ the active id)
            return self._json(app.list_sessions())
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._read()
            app = self._client()
            if path == "/api/move":
                ok = app.game.move_uci(str(body.get("uci", "")))
                return self._json({"ok": ok, "state": app.state()}, 200 if ok else 400)
            if path == "/api/sync":
                # The client board is authoritative (smooth, no per-move round-trip);
                # mirror it here AND persist to the active session (so a reload restores
                # the exact position). Normal play sends a move list (replayed -> preserves
                # history for review_move/undo); a FEN-loaded position sends fen.
                out = app.sync(fen=str(body.get("fen", "")), moves=body.get("moves", []))
                return self._json(out, 200 if out.get("ok") else 400)
            if path == "/api/reset":
                return self._json({"ok": True, "state": app.reset()})
            if path == "/api/session/new":     # create + switch to a fresh empty game
                return self._json({"ok": True, **app.new_session()})
            if path == "/api/session/switch":  # restore a session: board + chat history
                sid = str(body.get("id", "")).strip()
                return self._json({"ok": True, **app.use_session(sid or None)})
            if path == "/api/session/delete":
                return self._json(app.delete_session(str(body.get("id", "")).strip()))
            if path == "/api/opponent":    # neural engine plays the side to move (vs random)
                from . import opponent
                return self._json(opponent.choose(str(body.get("fen", ""))))
            if path == "/api/engine":      # toggle the eval-bar engine (stockfish | custom)
                from . import eval_engines
                eval_engines.set_engine(str(body.get("engine", "")).strip())
                return self._json({"ok": True, "engine": eval_engines.current(),
                                   "available": eval_engines.available(), "state": app.state()})
            if path == "/api/base/load":   # demo dual: bring up the untrained HF base on demand
                return self._json(app.load_base())
            if path == "/api/base/unload":  # free it immediately when dual is turned off
                return self._json(app.unload_base())
            if path == "/api/chat":
                msg = str(body.get("message", "")).strip()
                if not msg:
                    raise ValueError("empty message")
                coverage = bool(body.get("coverage", True))
                variant = str(body.get("variant", "sft"))
                mode = str(body.get("mode", ""))   # reasoning mode: fast | think | auto | plan
                if variant == "base":  # the untrained side of the dual — independent request
                    return self._json({"ok": True, **app.chat_base(msg, mode)})
                if body.get("stream"):  # SSE: emit each tool step live, then the final reply
                    return self._chat_stream(app, msg, coverage, mode)
                return self._json({"ok": True, **app.chat(msg, variant, coverage, mode=mode)})
            if path == "/api/skill":
                skill_admin.add_skill(str(body.get("name", "")), str(body.get("description", "")),
                                      str(body.get("body", "")))
                return self._json({"ok": True, **app.skills_payload()})
            if path == "/api/skill/delete":
                skill_admin.delete_skill(str(body.get("name", "")))
                return self._json({"ok": True, **app.skills_payload()})
            if path == "/api/plugin":
                skill_admin.apply_plugin(app.plugin_context, body)
                return self._json({"ok": True, **app.skills_payload()})
            self.send_error(404)
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)}, 500)

    def _chat_stream(self, app, msg: str, coverage: bool, mode: str = "") -> None:
        """Server-Sent Events. Two live streams during generation: `tool` events (THINKING
        — tool steps as they run) and `token` events (CHAT — reply tokens as the model
        produces them, true streaming). A final `done` carries the authoritative reply +
        board. If the backend can't stream tokens, fall back to a post-hoc sentence reveal."""
        self.send_response(200)
        self._maybe_set_cookie()
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("x-accel-buffering", "no")          # disable proxy buffering
        # nosniff: stop Chromium holding the first ~1KB to sniff the content-type — that
        # buffer was stalling the stream ~7s before any chunk reached the page's reader.
        self.send_header("x-content-type-options", "nosniff")
        self.send_header("transfer-encoding", "chunked")     # incremental framing for the browser
        self.end_headers()
        saw_token = [False]

        def _write_chunk(payload: bytes) -> None:
            # one HTTP chunk: <hex length>CRLF <payload> CRLF — the browser renders each as
            # it arrives instead of buffering until connection close.
            self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
            self.wfile.flush()

        # Padding comment FIRST: 2KB of SSE comment (lines starting with ':') pushes past
        # the browser's sniff buffer so subsequent events are delivered to JS immediately.
        try:
            _write_chunk(b": " + (b" " * 2048) + b"\n\n")
        except Exception:
            pass

        def emit(ev: dict) -> None:
            if ev.get("type") == "token":
                saw_token[0] = True
            payload = b"data: " + json.dumps(ev).encode("utf-8") + b"\n\n"
            try:
                _write_chunk(payload)
            except Exception:
                pass  # client disconnected; let the turn finish server-side

        try:
            result = app.chat(msg, "sft", coverage, on_event=emit, mode=mode)
            if not saw_token[0]:
                # Backend didn't stream tokens (non-streaming model) — fall back to a
                # paced sentence/clause reveal of the finished, grounded reply.
                for chunk in chunk_reply(result.get("reply") or ""):
                    emit({"type": "reply_chunk", "text": chunk})
                    time.sleep(0.05)
            # `done` carries the authoritative full reply + board state (frontend swaps it in).
            emit({"type": "done", "ok": True, **result})
        except Exception as exc:  # already streaming -> report via an event, not a 500
            emit({"type": "error", "error": str(exc)})
        try:
            self.wfile.write(b"0\r\n\r\n")                    # terminating chunk
            self.wfile.flush()
        except Exception:
            pass

    def _read(self) -> dict:
        n = int(self.headers.get("content-length", "0"))
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._maybe_set_cookie()
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
        self._maybe_set_cookie()
        self.send_header("content-type", TYPES.get(p.suffix, "application/octet-stream"))
        self.send_header("content-length", str(len(data)))
        self.send_header("Cache-Control", "no-store, must-revalidate")  # dev: always fresh
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        return


def main() -> None:
    skill_admin.register()  # runtime skills dir on CHESS_SKILLS_DIRS, wiped fresh
    adapter = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"loading model (adapter={adapter}) ...", flush=True)
    # Load the weights ONCE and register them; every browser gets its own App (board/history/
    # sessions) bound to this one shared model — multi-user isolated without duplicating weights.
    model, has_adapter, error = load_shared_model(adapter)
    client_registry.set_shared_model(model, has_adapter, error)
    host, port = bind_address()
    print(f"open http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
