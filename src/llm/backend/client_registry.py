"""Per-client (per-browser) App registry.

The serve was SINGLE-GLOBAL: `server.py` held one `App`, so every browser hitting the ngrok tunnel
shared ONE board, history, and active session — one user saw everyone else's game, and concurrent
session switches stomped each other. This registry gives each browser its OWN `App` (own Game,
history, session_id, and a SessionStore namespaced to `data/sessions/<cid>/`), keyed by a cookie
`cid`, while the heavy MODEL is loaded ONCE (`set_shared_model`) and shared via `App.bind_model()` —
the weights never duplicate. Bounded LRU so memory + Stockfish-process count stay capped; evicting a
client quits its engine. The cid is validated `^[a-f0-9]{32}$` by the caller BEFORE it becomes a
directory name (no path traversal)."""
from __future__ import annotations

import re
import threading
import uuid
from collections import OrderedDict

from .engine import Engine
from .sessions import SessionStore, _root
from .web_app import App

_CID_RE = re.compile(r"^[a-f0-9]{32}$")
_CLIENTS: "OrderedDict[str, App]" = OrderedDict()
_LOCK = threading.Lock()
_MAX_CLIENTS = 32
# The one shared model, set once at startup; every client App binds to it (no weight duplication).
_SHARED: dict = {"model": None, "has_adapter": False, "error": None, "loaded": False}


def new_cid() -> str:
    return uuid.uuid4().hex


def valid_cid(cid: str | None) -> bool:
    """A cid is used as a directory name (data/sessions/<cid>) — accept ONLY 32 lowercase hex so a
    forged cookie can't traverse the filesystem."""
    return bool(cid and _CID_RE.match(cid))


def set_shared_model(model, has_adapter: bool, error: str | None) -> None:
    """Register the ONE loaded model; clients built after this bind to it. Existing clients (built
    before load finished) are rebound so a slow model load doesn't leave early clients model-less."""
    with _LOCK:
        _SHARED.update(model=model, has_adapter=has_adapter, error=error, loaded=True)
        for app in _CLIENTS.values():
            app.bind_model(model, has_adapter, error)


def _build(cid: str) -> App:
    app = App(adapter=None, engine=Engine(), store=SessionStore(root=_root() / cid))
    if _SHARED["loaded"]:
        app.bind_model(_SHARED["model"], _SHARED["has_adapter"], _SHARED["error"])
    return app


def get_client(cid: str) -> App:
    """The App for this browser — created on first use, reused after, LRU-promoted on touch. Evicts
    the oldest (quitting its engine) past the cap so a flood of one-shot visitors can't grow unbounded."""
    with _LOCK:
        app = _CLIENTS.get(cid)
        if app is None:
            app = _build(cid)
            _CLIENTS[cid] = app
        else:
            _CLIENTS.move_to_end(cid)
        while len(_CLIENTS) > _MAX_CLIENTS:
            _, old = _CLIENTS.popitem(last=False)
            _quit(old)
        return app


def _quit(app: App) -> None:
    try:
        app.engine.quit()                       # close the client's Stockfish subprocess (if started)
    except Exception:
        pass


def reset() -> None:
    """Drop all clients + the shared model (tests; and a clean server restart)."""
    with _LOCK:
        for app in _CLIENTS.values():
            _quit(app)
        _CLIENTS.clear()
        _SHARED.update(model=None, has_adapter=False, error=None, loaded=False)
