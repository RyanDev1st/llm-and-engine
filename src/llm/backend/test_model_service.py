"""The model-service split: a persistent process holds the weights; the app talks to
it over localhost so it can restart without reloading them. Tested with a STUB model
in-thread (no real weights) — proves the client/server round-trip and that App can run
a full chat through the remote."""
import threading
from http.server import ThreadingHTTPServer

import chess

from backend import model_server
from backend.model_remote import RemoteModel, server_has_adapter
from backend.web_app import App


class StubModel:
    def __init__(self, steps):
        self.steps = list(steps)
        self.i = 0

    def generate(self, messages, max_new_tokens, stop, use_adapter=True):
        out = self.steps[min(self.i, len(self.steps) - 1)]
        self.i += 1
        return out

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def context_limit(self):
        return 4096


def _serve(model, has_adapter):
    model_server.MODEL = model
    model_server.HAS_ADAPTER = has_adapter
    srv = ThreadingHTTPServer(("127.0.0.1", 0), model_server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_remote_model_roundtrip(monkeypatch):
    srv, port = _serve(StubModel(["remote says hi"]), has_adapter=True)
    try:
        monkeypatch.setenv("CHESS_MODEL_SERVER", f"http://127.0.0.1:{port}")
        assert server_has_adapter() is True
        m = RemoteModel(has_adapter=True)
        assert m.generate([{"role": "user", "content": "x"}], 16, [], use_adapter=False) == "remote says hi"
        assert m.count_tokens("12345678") == 2
        assert m.context_limit() == 4096
    finally:
        srv.shutdown()


def test_app_runs_a_chat_through_the_service(monkeypatch):
    srv, port = _serve(StubModel(["Hello from the remote model."]), has_adapter=True)
    try:
        monkeypatch.setenv("CHESS_MODEL_SERVER", f"http://127.0.0.1:{port}")
        app = App(adapter=None)
        app.load_model()                       # connects to the service — no weights here
        assert app.loop is not None and app.loop_base is not None
        out = app.chat("hi", variant="sft")
        assert out["reply"] == "Hello from the remote model."
        assert app.game.board.fen() == chess.STARTING_FEN
    finally:
        srv.shutdown()
