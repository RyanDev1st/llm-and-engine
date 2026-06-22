"""Persistent multi-session store: the chess web app was single-global (board + chat lost on
reload). SessionStore keys each conversation by id and persists to disk so a reload OR a server
restart restores it. Data-only (moves/fen + chat turns + meta) — the App rebuilds a live Game per
request. CPU, no model. Isolated via CHESS_SESSIONS_DIR / the root= arg (tmp_path)."""
from backend import sessions
from backend.sessions import SessionStore


def _store(tmp_path):
    return SessionStore(root=tmp_path)


def test_create_persists_and_a_fresh_store_restores_it(tmp_path):
    sess = _store(tmp_path).create(title="My game")
    assert sess.id and sess.title == "My game"
    assert (tmp_path / f"{sess.id}.json").exists()
    got = _store(tmp_path).get(sess.id)                 # a NEW store = a server restart
    assert got is not None and got.id == sess.id and got.title == "My game"


def test_save_restores_board_and_chat_history(tmp_path):
    s = _store(tmp_path)
    sess = s.create()
    sess.moves = ["e2e4", "e7e5"]
    sess.history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    s.save(sess)
    got = _store(tmp_path).get(sess.id)
    assert got.moves == ["e2e4", "e7e5"] and len(got.history) == 2 and got.history[0]["content"] == "hi"


def test_list_is_sorted_by_last_active_desc(tmp_path):
    import time
    s = _store(tmp_path)
    a = s.create(title="A")
    s.create(title="B")
    time.sleep(0.01)
    s.touch(a.id)                                       # A becomes the most-recent
    ids = [r["id"] for r in s.list()]
    assert ids[0] == a.id and len(ids) == 2


def test_delete_removes_file_and_cache(tmp_path):
    s = _store(tmp_path)
    sess = s.create()
    assert s.delete(sess.id) is True
    assert s.get(sess.id) is None and not (tmp_path / f"{sess.id}.json").exists()
    assert s.delete("does-not-exist") is False


def test_history_is_capped(tmp_path):
    s = _store(tmp_path)
    sess = s.create()
    for i in range(sessions.HISTORY_CAP + 25):
        sess.history.append({"role": "user", "content": str(i)})
    s.save(sess)
    got = _store(tmp_path).get(sess.id)
    assert len(got.history) == sessions.HISTORY_CAP and got.history[-1]["content"] == str(sessions.HISTORY_CAP + 24)


def test_get_missing_returns_none(tmp_path):
    assert _store(tmp_path).get("missing") is None


def test_corrupt_session_file_is_skipped_not_fatal(tmp_path):
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "broken.json").write_text("{ not json", encoding="utf-8")
    s = _store(tmp_path)
    assert s.get("broken") is None and s.list() == []   # tolerates a hand-corrupted file
