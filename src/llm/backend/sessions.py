"""Persistent multi-session store for the chess web app.

The serve was single-global (one board, one history, both lost on reload). SessionStore keys each
conversation by id and persists it to disk, so a page reload OR a server restart restores the board
+ chat. It holds DATA only — the move list (or a loaded fen) + the chat turns + light metadata — so
it stays JSON-serializable; the App rebuilds a live Game from a session's moves per request. Single
JSON file per session under CHESS_SESSIONS_DIR (default data/sessions/, gitignored); sync that dir
for cross-machine continuity. Bounded history per the profile-store discipline."""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

HISTORY_CAP = 200            # chat turns kept per session (oldest dropped)
_TITLE_MAX = 80
_FIELDS = {"id", "title", "moves", "fen", "history", "created_at", "last_active"}


def _root() -> Path:
    return Path(os.environ.get("CHESS_SESSIONS_DIR")
                or Path(__file__).resolve().parents[3] / "data" / "sessions")


@dataclass
class Session:
    id: str
    title: str = "New game"
    moves: list = field(default_factory=list)      # uci move list (normal play)
    fen: str = ""                                  # a loaded position (puzzle/paste) instead of moves
    history: list = field(default_factory=list)    # [{role, content}, ...] — user/assistant turns only
    created_at: float = 0.0
    last_active: float = 0.0

    def summary(self) -> dict:
        """The lightweight row the sidebar lists (no board/history payload)."""
        return {"id": self.id, "title": self.title, "created_at": self.created_at,
                "last_active": self.last_active, "moves": len(self.moves)}


class SessionStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else _root()
        self._cache: dict[str, Session] = {}

    def _path(self, sid: str) -> Path:
        return self.root / f"{sid}.json"

    def create(self, title: str = "New game") -> Session:
        now = time.time()
        sess = Session(id=uuid.uuid4().hex[:12], title=(title or "New game").strip()[:_TITLE_MAX] or "New game",
                       created_at=now, last_active=now)
        return self.save(sess)

    def get(self, sid: str) -> Session | None:
        if sid in self._cache:
            return self._cache[sid]
        p = self._path(sid)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None                                # corrupt file -> treat as absent, never crash
        if not isinstance(d, dict) or "id" not in d:
            return None
        sess = Session(**{k: v for k, v in d.items() if k in _FIELDS})
        self._cache[sess.id] = sess
        return sess

    def save(self, sess: Session) -> Session:
        sess.last_active = time.time()
        if len(sess.history) > HISTORY_CAP:            # bounded: oldest turns drop
            sess.history = sess.history[-HISTORY_CAP:]
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(sess.id).write_text(
            json.dumps(asdict(sess), ensure_ascii=False, indent=2), encoding="utf-8")
        self._cache[sess.id] = sess
        return sess

    def touch(self, sid: str) -> Session | None:
        """Bump last_active (mark most-recent) without other changes."""
        sess = self.get(sid)
        return self.save(sess) if sess else None

    def list(self) -> list[dict]:
        """Session summaries, most-recently-active first (the sidebar order)."""
        out: list[dict] = []
        if self.root.exists():
            for p in self.root.glob("*.json"):
                sess = self.get(p.stem)
                if sess:
                    out.append(sess.summary())
        out.sort(key=lambda r: r["last_active"], reverse=True)
        return out

    def delete(self, sid: str) -> bool:
        existed = self._path(sid).exists()
        try:
            self._path(sid).unlink()
        except FileNotFoundError:
            pass
        self._cache.pop(sid, None)
        return existed
