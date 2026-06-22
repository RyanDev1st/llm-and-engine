"""App-level session persistence: the serve was single-global (board + chat lost on reload). The App
now adopts a disk-keyed session per conversation, so a page reload OR a server restart (= a fresh App
on the same CHESS_SESSIONS_DIR) restores the exact board + chat, and the user can switch/new/delete
games. No model/engine — scripted loops + the `move` tool (no Stockfish). Isolated via tmp_path."""
import chess

from backend.inference import CoachLoop
from backend.web_app import App


class ScriptedModel:
    def __init__(self, steps):
        self.steps = list(steps)
        self.i = 0

    def generate(self, messages, max_new_tokens, stop):
        out = self.steps[min(self.i, len(self.steps) - 1)]
        self.i += 1
        return out


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESS_SESSIONS_DIR", str(tmp_path))
    return App(adapter=None)


def test_chat_persists_then_a_fresh_app_restores_board_and_chat(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    app.loop = CoachLoop(
        ScriptedModel(["I'll take the centre.\n<tool>move san=e4", "e4 is a solid start."]), app.executor)
    app.chat("play e4 for me")
    sid = app.session_id
    assert sid and app.game.board.move_stack[-1].uci() == "e2e4"

    app2 = App(adapter=None)                      # a NEW App on the same dir = a server restart
    restored = app2.use_session(sid)
    assert [m.uci() for m in app2.game.board.move_stack] == ["e2e4"]   # board rebuilt from disk
    assert app2.session_id == sid
    assert any(t["role"] == "user" and "play e4" in t["content"] for t in restored["history"])


def test_switch_restores_each_sessions_own_board(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    app.new_session()
    app.sync(moves=["e2e4", "e7e5"])              # session A: 1.e4 e5
    aid = app.session_id
    app.new_session()
    app.sync(moves=["d2d4"])                       # session B: 1.d4
    bid = app.session_id
    assert aid != bid

    app.use_session(aid)
    assert [m.uci() for m in app.game.board.move_stack] == ["e2e4", "e7e5"]
    app.use_session(bid)
    assert [m.uci() for m in app.game.board.move_stack] == ["d2d4"]


def test_new_session_is_a_fresh_empty_game(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    app.sync(moves=["e2e4"])
    app.history.append({"role": "user", "content": "hi"})
    out = app.new_session()
    assert app.game.board.move_stack == [] and app.history == []
    assert out["state"]["fen"].split()[0] == chess.STARTING_FEN.split()[0]


def test_delete_active_session_clears_the_pointer(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    app.new_session()
    sid = app.session_id
    out = app.delete_session(sid)
    assert out["ok"] is True and app.session_id is None
    assert sid not in [r["id"] for r in out["sessions"]]


def test_sync_fen_position_persists_and_restores(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    app.sync(fen=fen)
    sid = app.session_id

    app2 = App(adapter=None)                      # server restart restores the loaded position
    app2.use_session(sid)
    assert app2.game.board.fen() == fen


def test_legacy_client_chat_auto_adopts_a_session(tmp_path, monkeypatch):
    # A client that never calls /api/session/* still gets its board+chat persisted (back-compat).
    app = _app(tmp_path, monkeypatch)
    app.loop = CoachLoop(ScriptedModel(["Hello there."]), app.executor)
    assert app.session_id is None
    app.chat("hi")
    assert app.session_id is not None and len(app.list_sessions()["sessions"]) == 1
