"""Per-client isolation: the serve was single-global (one user saw everyone's board/session). The
registry gives each cookie cid its own App + its own SessionStore namespace, sharing one model. These
verify two clients are fully isolated, the same cid is sticky, the cid validator blocks path traversal,
and LRU eviction is bounded. CPU, no model (board/session ops don't need weights)."""
from backend import client_registry as reg


def _hex(c):
    return c * 32                                  # a valid 32-hex cid


def test_valid_cid_only_accepts_32_hex():
    assert reg.valid_cid("a" * 32)
    assert not reg.valid_cid("")
    assert not reg.valid_cid(None)
    assert not reg.valid_cid("../../etc/passwd")   # path traversal blocked
    assert not reg.valid_cid("A" * 32)             # uppercase rejected
    assert not reg.valid_cid("a" * 31)             # wrong length
    assert reg.valid_cid(reg.new_cid())            # minted cids are valid


def test_same_cid_returns_the_same_app(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESS_SESSIONS_DIR", str(tmp_path)); reg.reset()
    assert reg.get_client(_hex("c")) is reg.get_client(_hex("c"))


def test_two_clients_have_isolated_boards_and_session_lists(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESS_SESSIONS_DIR", str(tmp_path)); reg.reset()
    a, b = reg.get_client(_hex("a")), reg.get_client(_hex("b"))
    assert a is not b
    a.new_session(); a.sync(moves=["e2e4", "e7e5"])
    b.new_session(); b.sync(moves=["d2d4"])
    # boards isolated
    assert [m.uci() for m in a.game.board.move_stack] == ["e2e4", "e7e5"]
    assert [m.uci() for m in b.game.board.move_stack] == ["d2d4"]
    # session lists isolated (each store sees only its own cid's dir)
    la, lb = a.list_sessions()["sessions"], b.list_sessions()["sessions"]
    assert len(la) == 1 and len(lb) == 1
    assert la[0]["id"] != lb[0]["id"]
    # and a fresh lookup of the same cid restores A's board, untouched by B
    assert [m.uci() for m in reg.get_client(_hex("a")).game.board.move_stack] == ["e2e4", "e7e5"]


def test_lru_eviction_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESS_SESSIONS_DIR", str(tmp_path)); reg.reset()
    monkeypatch.setattr(reg, "_MAX_CLIENTS", 2)
    reg.get_client(_hex("a")); reg.get_client(_hex("b")); reg.get_client(_hex("c"))  # 3 > cap 2
    assert len(reg._CLIENTS) == 2
    assert _hex("a") not in reg._CLIENTS          # oldest evicted
    assert _hex("c") in reg._CLIENTS


def test_shared_model_binds_existing_and_future_clients(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESS_SESSIONS_DIR", str(tmp_path)); reg.reset()
    early = reg.get_client(_hex("a"))             # built before the model is ready
    assert early.loop is None

    class FakeModel:
        def generate(self, *a, **k): return "ok"
        def count_tokens(self, t): return len(t)
        def context_limit(self): return 4096
    reg.set_shared_model(FakeModel(), has_adapter=False, error=None)

    assert early.loop is not None                 # existing client got rebound
    assert reg.get_client(_hex("b")).loop is not None  # future client binds on build
