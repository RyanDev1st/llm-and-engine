"""HTTP-level smoke for the session endpoints + per-client (cookie) isolation. Runs the real Handler
on an ephemeral port against the registry with NO model (board + session routes don't need weights).
Each `browser()` is a urllib opener with its OWN cookie jar = its own cid = its own App, so this also
proves two browsers don't see each other's board/sessions. Isolated via CHESS_SESSIONS_DIR."""
import http.cookiejar
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import backend.server as server
from backend import client_registry


def _serve(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESS_SESSIONS_DIR", str(tmp_path))
    client_registry.reset()                              # no model registered -> board/disk routes only
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]

    def browser():
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

        def call(method, path, body=None):
            data = json.dumps(body).encode() if body is not None else None
            req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                         headers={"content-type": "application/json"})
            with opener.open(req) as r:
                return json.loads(r.read().decode())
        return call

    return httpd, browser


def test_session_endpoints_roundtrip(tmp_path, monkeypatch):
    httpd, browser = _serve(tmp_path, monkeypatch)
    call = browser()
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


def test_two_browsers_are_isolated(tmp_path, monkeypatch):
    # THE multi-user fix: two cookie jars (two browsers) must not see each other's board or sessions.
    httpd, browser = _serve(tmp_path, monkeypatch)
    alice, bob = browser(), browser()
    try:
        alice("POST", "/api/session/new", {}); alice("POST", "/api/sync", {"moves": ["e2e4"]})
        bob("POST", "/api/session/new", {});   bob("POST", "/api/sync", {"moves": ["d2d4"]})

        a_state = alice("GET", "/api/state"); b_state = bob("GET", "/api/state")
        assert a_state["fen"].split()[0].endswith("/4P3/8/PPPP1PPP/RNBQKBNR")   # Alice = 1.e4
        assert b_state["fen"].startswith("rnbqkbnr/pppppppp/8/8/3P4")           # Bob = 1.d4

        a_list = alice("GET", "/api/sessions")["sessions"]
        b_list = bob("GET", "/api/sessions")["sessions"]
        assert len(a_list) == 1 and len(b_list) == 1                            # each sees only its own
        assert a_list[0]["id"] != b_list[0]["id"]
    finally:
        httpd.shutdown()
