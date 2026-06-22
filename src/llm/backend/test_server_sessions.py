"""HTTP-level smoke for the session endpoints wired into server.py: GET /api/sessions and
POST /api/session/new|switch|delete, plus /api/sync now routing through App.sync() (persisting).
Runs the real Handler on an ephemeral port against an App with NO model (board + session routes
don't need weights). Isolated via CHESS_SESSIONS_DIR. Complements the App-level test_web_app_sessions."""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import backend.server as server
from backend.web_app import App


def _serve(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESS_SESSIONS_DIR", str(tmp_path))
    server.APP = App(adapter=None)                       # no weights — these routes are board/disk only
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]

    def call(method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read().decode())

    return httpd, call


def test_session_endpoints_roundtrip(tmp_path, monkeypatch):
    httpd, call = _serve(tmp_path, monkeypatch)
    try:
        new = call("POST", "/api/session/new", {})
        assert new["ok"] and new["id"]
        sid = new["id"]

        synced = call("POST", "/api/sync", {"moves": ["e2e4", "e7e5"]})   # persists to the active session
        assert synced["ok"] and synced["state"]["fen"].split()[0] == "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR"

        lst = call("GET", "/api/sessions")
        assert lst["current"] == sid and any(s["id"] == sid for s in lst["sessions"])

        call("POST", "/api/session/new", {})                              # a second game (fresh)
        sw = call("POST", "/api/session/switch", {"id": sid})             # back to the first
        assert sw["ok"] and sw["moves"] == ["e2e4", "e7e5"]               # board replayable
        assert sw["state"]["fen"].split()[0] == "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR"

        d = call("POST", "/api/session/delete", {"id": sid})
        assert d["ok"] and sid not in [s["id"] for s in d["sessions"]]
    finally:
        httpd.shutdown()
