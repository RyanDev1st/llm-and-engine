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
from . import skill_admin, state_api
from .tools import ToolExecutor


def agent_overlay() -> str:
    """Optional customization layer (tone/extra rules). Default empty → serving
    system text == the trained contract (no drift)."""
    return os.environ.get("CHESS_AGENT_OVERLAY", "")


class App:
    def __init__(self, adapter: str | None) -> None:
        self.game = Game()
        self.engine = Engine()
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

    def load_model(self) -> None:
        # Dev mode: a persistent model service holds the weights, so this server is
        # weightless and restarts in ~1s. Set CHESS_MODEL_SERVER to its URL.
        if os.environ.get("CHESS_MODEL_SERVER"):
            try:
                self._connect_model_service()
                return
            except Exception as exc:  # service down -> fall through to in-process load
                self.model_error = str(exc)
                print(f"model service unreachable ({exc}); loading in-process", flush=True)
        adapter = self._adapter or os.environ.get("CHESS_HF_ADAPTER", "")
        if adapter:
            try:
                from .model_hf import HFModel
                # greedy (temp 0) — faithful to the tool number and reproducible
                # for the demo; sampling let it drift off the engine's eval.
                model = HFModel(adapter=adapter, temperature=0.0)
                ov, pc = agent_overlay(), self.plugin_context
                self.loop = CoachLoop(AdapterView(model, True), self.executor, ov, pc)
                self.loop_base = CoachLoop(AdapterView(model, False), self.base_executor, ov, pc)
                # adapter ON, on the isolated board — runs the coverage-OFF side of the ablation
                self.loop_mirror = CoachLoop(AdapterView(model, True), self.base_executor, ov, pc)
                print(f"model loaded (HF base + adapter {adapter}; compare ready)", flush=True)
                return
            except Exception as exc:
                self.model_error = str(exc)
                print(f"HF adapter load failed ({exc}); falling back to GGUF", flush=True)
        from .model_gguf import GGUFModel, default_gguf_path, gguf_runtime_config
        try:
            gguf = default_gguf_path()
            if gguf.exists():
                n_ctx, n_gpu_layers = gguf_runtime_config()
                model = GGUFModel(gguf=gguf, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
                self.loop = CoachLoop(model, self.executor, agent_overlay(), self.plugin_context)
                print(f"model loaded (GGUF {gguf.name})", flush=True)
                return
            raise FileNotFoundError(f"no GGUF at {gguf}; set CHESS_GGUF_PATH")
        except Exception as exc:  # board + eval still work without the model
            self.model_error = str(exc)
            print(f"model unavailable ({exc}); board/eval still work", flush=True)

    def _connect_model_service(self) -> None:
        """Build the loops against the remote model service (no weights here). Same
        loop wiring as the HF path: adapter on (SFT) + adapter off (base) + the
        isolated coverage-off mirror, all proxied to the one persistent service."""
        from .model_remote import RemoteModel, server_has_adapter, server_url
        has = server_has_adapter()
        ov, pc = agent_overlay(), self.plugin_context
        if has:
            model = RemoteModel(has_adapter=True)
            self.loop = CoachLoop(AdapterView(model, True), self.executor, ov, pc)
            self.loop_base = CoachLoop(AdapterView(model, False), self.base_executor, ov, pc)
            self.loop_mirror = CoachLoop(AdapterView(model, True), self.base_executor, ov, pc)
        else:
            model = RemoteModel(has_adapter=False)
            self.loop = CoachLoop(model, self.executor, ov, pc)
        self.model_error = None
        print(f"connected to model service at {server_url()} (adapter={has})", flush=True)

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

    def chat_base(self, message: str) -> dict:
        """Run the UNTRAINED base on the mirrored board (independent of the trained side)."""
        if self.loop_base is None:
            return {"reply": "(base model not loaded)", "tool_calls": [], "tool_results": [],
                    "elapsed_s": 0, "state": self.state()}
        self._mirror_base()
        out = self._run(self.loop_base, self.history_base, message)
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
        return self.state()

    def _mirror_base(self) -> None:
        """Copy the real board onto the base loop's private board so the base
        model analyzes the SAME position as SFT but cannot mutate the displayed
        game. board.copy() carries the move stack so review_move/undo still work."""
        bg = self.base_executor.game
        bg.board = self.game.board.copy()
        bg.san_stack = list(self.game.san_stack)

    def _run(self, loop: CoachLoop, history: list[dict], message: str, coverage: bool = True,
             on_event=None) -> dict:
        import time
        t0 = time.time()
        result = loop.respond(history, message, coverage, on_event)
        elapsed = round(time.time() - t0, 2)   # agent time: prompt received -> task finished
        # Thinking turns (tool calls + results) are ephemeral: respond() already
        # used them in-turn to write the reply. Persist ONLY the user message and
        # the final reply, so the reasoning scratchpad never pollutes future
        # context. The model re-derives via tools next turn if it needs to.
        history += [{"role": "user", "content": message},
                    {"role": "assistant", "content": result["reply"]}]
        return {"reply": result["reply"], "tool_calls": result.get("tool_calls", []),
                "tool_results": result.get("tool_results", []),
                "context": result.get("context"), "elapsed_s": elapsed}

    def chat(self, message: str, variant: str = "sft", coverage: bool = True, on_event=None) -> dict:
        if self.loop is None:
            return {"reply": f"(model not loaded: {self.model_error or 'no adapter'})",
                    "tool_calls": [], "tool_results": [], "state": self.state()}
        if variant == "both" and self.loop_base is not None:
            sft = self._run(self.loop, self.history, message, coverage)
            board = self.state()  # the visible board follows OUR model, snapshot before base runs
            self._mirror_base()   # base runs on a private copy — never touches the real board
            base = self._run(self.loop_base, self.history_base, message, coverage)
            return {"sft": sft, "base": base, "state": board}
        if variant == "coverage" and self.loop_mirror is not None:
            # Ablation: same prompt with the coverage layer ON vs OFF, side by side.
            on = self._run(self.loop, self.history, message, coverage=True)
            board = self.state()           # the ON run drives the visible board
            self._mirror_base()            # the OFF run uses a private copy
            off = self._run(self.loop_mirror, self.history_off, message, coverage=False)
            return {"on": on, "off": off, "state": board}
        out = self._run(self.loop, self.history, message, coverage, on_event)
        return {**out, "tool_call": out["tool_calls"][-1] if out["tool_calls"] else None,
                "tool_result": out["tool_results"][-1] if out["tool_results"] else None,
                "state": self.state()}

    def skills_payload(self) -> dict:
        payload = skill_admin.catalog_payload(self.plugin_context)
        payload["compare_ready"] = self.loop_base is not None
        return payload
