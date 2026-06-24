"""Web App state: one shared board + engine, and (when a LoRA adapter is loaded)
TWO CoachLoops over ONE model — adapter ON (our SFT) and OFF (untrained base) —
so the demo can answer the same prompt with both, same harness/skills/tools, to
show why the SFT training matters.
"""
from __future__ import annotations

import os

from .game import Game
from .engine import Engine
from .inference import AdapterView, CoachLoop, PLUGIN_CONTEXT
from . import memory, skill_admin, state_api
from .memory import session as session_mem
from .sessions import SessionStore
from .tools import ToolExecutor


def agent_overlay() -> str:
    """Optional customization layer (tone/extra rules). Default empty → serving
    system text == the trained contract (no drift)."""
    return os.environ.get("CHESS_AGENT_OVERLAY", "")


def load_shared_model(adapter: str | None) -> tuple[object | None, bool, str | None]:
    """Build the heavy MODEL once (no loops): returns (model, has_adapter, error). Tries, in order,
    the persistent model service (CHESS_MODEL_SERVER, weightless), then an in-process HF adapter,
    then GGUF. The multi-user server calls this ONCE and shares the result across every client's
    App.bind_model(); a single-process App.load_model() calls it for itself. error!=None means the
    board/eval still work but chat is unavailable."""
    if os.environ.get("CHESS_MODEL_SERVER"):
        try:
            from .model_remote import RemoteModel, server_has_adapter, server_url
            has = server_has_adapter()
            print(f"connected to model service at {server_url()} (adapter={has})", flush=True)
            return RemoteModel(has_adapter=has), has, None
        except Exception as exc:  # service down -> fall through to an in-process load
            print(f"model service unreachable ({exc}); loading in-process", flush=True)
    adapter = adapter or os.environ.get("CHESS_HF_ADAPTER", "")
    from .model_gguf import pick_backend
    explicit = (os.environ.get("CHESS_BACKEND") or "").strip().lower() in ("hf", "gguf")
    if pick_backend(os.environ.get("CHESS_BACKEND"), adapter) == "hf":
        try:
            from .model_hf import HFModel
            # greedy (temp 0) — faithful to the tool number and reproducible for the demo.
            model = HFModel(adapter=adapter or None, temperature=0.0)
            print(f"model loaded (HF base + adapter {adapter}; compare ready)", flush=True)
            return model, bool(adapter), None
        except Exception as exc:
            if explicit:
                raise   # CHESS_BACKEND=hf -> fail loud, don't mask by serving GGUF
            print(f"HF adapter load failed ({exc}); falling back to GGUF", flush=True)
    from .model_gguf import GGUFModel, default_gguf_path, gguf_runtime_config
    try:
        gguf = default_gguf_path()
        if gguf.exists():
            n_ctx, n_gpu_layers = gguf_runtime_config()
            print(f"model loaded (GGUF {gguf.name})", flush=True)
            return GGUFModel(gguf=gguf, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers), False, None
        raise FileNotFoundError(f"no GGUF at {gguf}; set CHESS_GGUF_PATH")
    except Exception as exc:  # board + eval still work without the model
        print(f"model unavailable ({exc}); board/eval still work", flush=True)
        return None, False, str(exc)


class App:
    def __init__(self, adapter: str | None, *, engine: "Engine | None" = None,
                 store: "SessionStore | None" = None) -> None:
        # engine + store may be INJECTED so a multi-user server gives each client its own engine
        # (per-client Stockfish) and its own session namespace (data/sessions/<cid>/), while the
        # heavy MODEL is shared via bind_model(). Defaults preserve the single-process behavior.
        self.game = Game()
        self.engine = engine or Engine()
        self.executor = ToolExecutor(self.game, self.engine)
        # The base (untrained) loop runs on its OWN board so a base move/undo/
        # load_fen can never mutate the real, displayed game. It is mirrored from
        # the real board right before each base turn (see _mirror_base).
        self.base_executor = ToolExecutor(Game(), self.engine)
        self.history: list[dict] = []
        self.history_base: list[dict] = []
        self.loop: CoachLoop | None = None       # SFT (adapter on)
        self.loop_base: CoachLoop | None = None   # untrained base (adapter off)
        self.loop_mirror: CoachLoop | None = None  # adapter on, isolated board (coverage-off compare)
        self.history_off: list[dict] = []         # history for the coverage-OFF run in the ablation
        self._base_model = None                    # lazily-loaded HF base (demo dual); freed on toggle-off
        self.model_error: str | None = None
        self._adapter = adapter
        self.plugin_context = {k: list(v) for k, v in PLUGIN_CONTEXT.items()}
        # Persistent-memory identity (single-user demo -> "default"; keyed so multi-user is free).
        self.user_id = os.environ.get("CHESS_USER_ID", "default")
        # Session fact cache: analysis facts computed this session, FEN-keyed (see memory.session).
        self.session: dict = {}
        # Persistent CONVERSATION sessions (board + chat), disk-keyed so a reload / restart restores
        # them and the user can switch between games. `session_id` = the active one (None until set).
        self.store = store or SessionStore()
        self.session_id: str | None = None

    def load_model(self) -> None:
        """Single-process load: build the model AND wire this App's loops to it. A multi-user
        server instead calls load_shared_model() ONCE and bind_model() per client (shared weights)."""
        model, has_adapter, error = load_shared_model(self._adapter)
        self.bind_model(model, has_adapter, error)

    def bind_model(self, model, has_adapter: bool, error: str | None) -> None:
        """Wire THIS App's loops to a (possibly shared) model. The model holds the heavy weights;
        the loops are cheap and bind this App's OWN executor/board, so many clients share one model
        while keeping isolated boards. has_adapter -> the AdapterView on/off + mirror demo loops."""
        self.model_error = error
        if model is None:
            return
        ov, pc = agent_overlay(), self.plugin_context
        if has_adapter:
            self.loop = CoachLoop(AdapterView(model, True), self.executor, ov, pc)
            self.loop_base = CoachLoop(AdapterView(model, False), self.base_executor, ov, pc)
            self.loop_mirror = CoachLoop(AdapterView(model, True), self.base_executor, ov, pc)
        else:
            self.loop = CoachLoop(model, self.executor, ov, pc)

    def load_base(self) -> dict:
        """Lazy-load the UNTRAINED HF base Gemma (demo-only — the 'base' side of the dual
        comparison). Loaded on demand when the user confirms the dual toggle; freed by
        unload_base when they turn it off, so it never hogs the GPU during normal GGUF use.
        Runs on the isolated base_executor so it can't mutate the displayed board."""
        if self.loop_base is not None:
            return {"ok": True, "loaded": True}
        try:
            from .model_hf import HFModel
            model = HFModel(adapter=None, temperature=0.0)   # base only, no SFT adapter
            self.loop_base = CoachLoop(model, self.base_executor, agent_overlay(), self.plugin_context)
            self._base_model = model
            print("base HF model loaded (demo dual)", flush=True)
            return {"ok": True, "loaded": True}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "loaded": False, "error": str(exc)}

    def unload_base(self) -> dict:
        """Free the HF base model immediately (frees VRAM) when dual is turned off."""
        self.loop_base = None
        model = getattr(self, "_base_model", None)
        self._base_model = None
        if model is not None:
            del model
            try:
                import gc, torch
                gc.collect(); torch.cuda.empty_cache()
            except Exception:
                pass
        self.history_base = []
        print("base HF model unloaded", flush=True)
        return {"ok": True, "loaded": False}

    def chat_base(self, message: str, mode: str = "") -> dict:
        """Run the UNTRAINED base on the mirrored board (independent of the trained side)."""
        if self.loop_base is None:
            return {"reply": "(base model not loaded)", "tool_calls": [], "tool_results": [],
                    "elapsed_s": 0, "state": self.state()}
        self._mirror_base()
        out = self._run(self.loop_base, self.history_base, message, mode=mode)
        return {**out, "state": self.state()}

    def state(self) -> dict:
        return state_api.snapshot(self.game, self.engine)

    def reset(self) -> dict:
        self.game = Game()
        self.executor.game = self.game
        self.base_executor.game = Game()
        self.history = []
        self.history_base = []
        self.history_off = []
        session_mem.clear(self.session)         # a new game -> drop the stale fact cache
        self._persist_session(moves=[])         # clear the active session's board + chat on disk
        return self.state()

    # --- persistent conversation sessions (board + chat survive reload / restart) ---
    def _ensure_session(self) -> str:
        """Adopt a session for the active board if none is selected yet (back-compat: a legacy
        client that never calls /api/session/* still gets its board+chat persisted)."""
        if self.session_id is None:
            self.session_id = self.store.create().id
        return self.session_id

    def _persist_session(self, moves=None, fen=None) -> None:
        """Write the active board (moves XOR fen) + the chat history to the active session on disk.
        moves=None and fen=None keeps the stored board (e.g. a chat turn saves only the conversation)."""
        if not self.session_id:
            return
        sess = self.store.get(self.session_id)
        if not sess:
            return
        if fen is not None:
            sess.fen, sess.moves = fen, []
        elif moves is not None:
            sess.moves, sess.fen = list(moves), ""
        sess.history = list(self.history)
        if (sess.title or "New game") == "New game":    # title from the first user message
            first = next((t.get("content", "") for t in self.history if t.get("role") == "user"), "")
            if first.strip():
                sess.title = first.strip()[:60]
        self.store.save(sess)

    def _persist_current_board(self) -> None:
        """Snapshot the live board into the active session, preserving its representation: a normal
        game persists the full uci move list (so review_move/undo survive reload AND the model's move
        is captured); a loaded position (puzzle/paste) persists fen. A bare fen save would clobber a
        normal game's move history, so we branch on the stored session's kind."""
        sess = self.store.get(self.session_id) if self.session_id else None
        if sess is None:
            return
        if sess.fen:                                    # a loaded position -> keep it fen-based
            self._persist_session(fen=self.game.board.fen())
        else:                                           # startpos game -> full uci history (incl. the model's move)
            self._persist_session(moves=[m.uci() for m in self.game.board.move_stack])

    def use_session(self, sid: str | None = None) -> dict:
        """Switch to a session (or create one): rebuild the live board from its moves/fen and load
        its chat history, so the frontend can restore the exact game. Returns board + history."""
        sess = (self.store.get(sid) if sid else None) or self.store.create()
        self.session_id = sess.id
        self.game = Game()
        if sess.fen:
            self.game.load_fen(sess.fen)
        elif sess.moves:
            self.game.load_uci_moves([str(m) for m in sess.moves])
        self.executor.game = self.game
        self.history = list(sess.history)
        session_mem.clear(self.session)                 # facts are board-specific; the board just changed
        # `moves` lets the frontend REPLAY a normal game (so its move list + undo survive reload);
        # it's [] for a fen-loaded position, where the frontend restores from state.fen instead.
        return {"id": sess.id, "title": sess.title, "history": list(self.history),
                "moves": [str(m) for m in sess.moves], "state": self.state()}

    def list_sessions(self) -> dict:
        return {"sessions": self.store.list(), "current": self.session_id}

    def new_session(self) -> dict:
        return self.use_session(None)                   # create + switch to a fresh empty game

    def delete_session(self, sid: str) -> dict:
        ok = self.store.delete(sid)
        if sid == self.session_id:
            self.session_id = None
        return {"ok": ok, **self.list_sessions()}

    def sync(self, fen: str = "", moves=None) -> dict:
        """Mirror the client-authoritative board here AND persist it to the active session, so a
        reload restores the exact position. Normal play sends a uci move list (history preserved);
        a loaded position (puzzle/paste) sends fen (history starts fresh)."""
        self._ensure_session()
        fen = (fen or "").strip()
        if fen:
            ok = self.game.load_fen(fen)
            if ok:
                self._persist_session(fen=self.game.board.fen())
        else:
            mv = [str(m) for m in (moves or [])] if isinstance(moves, list) else []
            ok = self.game.load_uci_moves(mv)
            if ok:
                self._persist_session(moves=mv)
        self.executor.game = self.game
        return {"ok": ok, "state": self.state()}

    def _context_block(self, message: str = "") -> str:
        """The injected memory context: the persistent user profile + any session facts still
        fresh for the live board + a recalled how-to-operate hint for THIS request (episodic;
        no-op unless CHESS_EPISODIC=1). Read from the REAL game (base/mirror runs mirror it)."""
        parts = [memory.memory_block(self.user_id),
                 session_mem.render(self.session, self.game.board.fen()),
                 memory.episodic_block(message, self.plugin_context)]
        return "\n\n".join(p for p in parts if p)

    def _mirror_base(self) -> None:
        """Copy the real board onto the base loop's private board so the base
        model analyzes the SAME position as SFT but cannot mutate the displayed
        game. board.copy() carries the move stack so review_move/undo still work."""
        bg = self.base_executor.game
        bg.board = self.game.board.copy()
        bg.san_stack = list(self.game.san_stack)

    def _run(self, loop: CoachLoop, history: list[dict], message: str, coverage: bool = True,
             on_event=None, mode: str = "") -> dict:
        import time
        t0 = time.time()
        entry_fen = self.game.board.fen()      # position the turn's analysis pertains to
        # Inject memory (user profile + fresh session facts) read BEFORE the turn; run; then
        # capture durable user facts (idempotent). Only the PRIMARY trained loop updates the
        # session cache — base/mirror runs are demo comparisons on a copied board.
        result = loop.respond(history, message, coverage, on_event, reasoning_mode=mode,
                              memory_block=self._context_block(message))
        memory.capture(message, self.user_id)
        # Episodic learning: only the PRIMARY trained loop's turns are real lessons (base/mirror are
        # demo comparisons). Harvest a how-to-operate episode (no-op unless CHESS_EPISODIC=1).
        if loop is self.loop:
            memory.observe(message, result, self.plugin_context)
        # Cache analysis facts ONLY when the board didn't move this turn (else the facts are
        # ambiguous as to which position they describe). The render freshness-guard drops them
        # the moment the live FEN diverges anyway.
        if loop is self.loop and self.game.board.fen() == entry_fen:
            session_mem.update(self.session, entry_fen, result.get("tool_results", []))
        elapsed = round(time.time() - t0, 2)   # agent time: prompt received -> task finished
        # Thinking turns (tool calls + results) are ephemeral: respond() already
        # used them in-turn to write the reply. Persist ONLY the user message and
        # the final reply, so the reasoning scratchpad never pollutes future
        # context. The model re-derives via tools next turn if it needs to.
        history += [{"role": "user", "content": message},
                    {"role": "assistant", "content": result["reply"]}]
        if loop is self.loop:                  # persist the REAL conversation (+ current board) to disk
            self._ensure_session()
            self._persist_current_board()
        return {"reply": result["reply"], "tool_calls": result.get("tool_calls", []),
                "tool_results": result.get("tool_results", []),
                "context": result.get("context"), "elapsed_s": elapsed}

    def chat(self, message: str, variant: str = "sft", coverage: bool = True, on_event=None,
             mode: str = "") -> dict:
        if self.loop is None:
            return {"reply": f"(model not loaded: {self.model_error or 'no adapter'})",
                    "tool_calls": [], "tool_results": [], "state": self.state()}
        if variant == "both" and self.loop_base is not None:
            sft = self._run(self.loop, self.history, message, coverage, mode=mode)
            board = self.state()  # the visible board follows OUR model, snapshot before base runs
            self._mirror_base()   # base runs on a private copy — never touches the real board
            base = self._run(self.loop_base, self.history_base, message, coverage, mode=mode)
            return {"sft": sft, "base": base, "state": board}
        if variant == "coverage" and self.loop_mirror is not None:
            # Ablation: same prompt with the coverage layer ON vs OFF, side by side.
            on = self._run(self.loop, self.history, message, coverage=True, mode=mode)
            board = self.state()           # the ON run drives the visible board
            self._mirror_base()            # the OFF run uses a private copy
            off = self._run(self.loop_mirror, self.history_off, message, coverage=False, mode=mode)
            return {"on": on, "off": off, "state": board}
        out = self._run(self.loop, self.history, message, coverage, on_event, mode=mode)
        return {**out, "tool_call": out["tool_calls"][-1] if out["tool_calls"] else None,
                "tool_result": out["tool_results"][-1] if out["tool_results"] else None,
                "state": self.state()}

    def skills_payload(self) -> dict:
        payload = skill_admin.catalog_payload(self.plugin_context)
        payload["compare_ready"] = self.loop_base is not None
        return payload
