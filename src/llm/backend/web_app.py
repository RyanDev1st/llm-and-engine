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
        self.history: list[dict] = []
        self.history_base: list[dict] = []
        self.loop: CoachLoop | None = None       # SFT (adapter on)
        self.loop_base: CoachLoop | None = None   # untrained base (adapter off)
        self.model_error: str | None = None
        self._adapter = adapter
        self.plugin_context = {k: list(v) for k, v in PLUGIN_CONTEXT.items()}

    def load_model(self) -> None:
        adapter = self._adapter or os.environ.get("CHESS_HF_ADAPTER", "")
        if adapter:
            try:
                from .model_hf import HFModel
                # greedy (temp 0) — faithful to the tool number and reproducible
                # for the demo; sampling let it drift off the engine's eval.
                model = HFModel(adapter=adapter, temperature=0.0)
                ov, pc = agent_overlay(), self.plugin_context
                self.loop = CoachLoop(AdapterView(model, True), self.executor, ov, pc)
                self.loop_base = CoachLoop(AdapterView(model, False), self.executor, ov, pc)
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

    def state(self) -> dict:
        return state_api.snapshot(self.game, self.engine)

    def reset(self) -> dict:
        self.game = Game()
        self.executor.game = self.game
        self.history = []
        self.history_base = []
        return self.state()

    def _run(self, loop: CoachLoop, history: list[dict], message: str) -> dict:
        result = loop.respond(history, message)
        history += result["turns"]
        return {"reply": result["reply"], "tool_calls": result.get("tool_calls", []),
                "tool_results": result.get("tool_results", [])}

    def chat(self, message: str, variant: str = "sft") -> dict:
        if self.loop is None:
            return {"reply": f"(model not loaded: {self.model_error or 'no adapter'})",
                    "tool_calls": [], "tool_results": [], "state": self.state()}
        if variant == "both" and self.loop_base is not None:
            return {"sft": self._run(self.loop, self.history, message),
                    "base": self._run(self.loop_base, self.history_base, message),
                    "state": self.state()}
        out = self._run(self.loop, self.history, message)
        return {**out, "tool_call": out["tool_calls"][-1] if out["tool_calls"] else None,
                "tool_result": out["tool_results"][-1] if out["tool_results"] else None,
                "state": self.state()}

    def skills_payload(self) -> dict:
        payload = skill_admin.catalog_payload(self.plugin_context)
        payload["compare_ready"] = self.loop_base is not None
        return payload
