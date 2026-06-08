"""Stdlib HTTP server for the chess-coach web app (no extra deps).

Single shared game (a board is "loaded", per spec). The board is authoritative:
drag-moves (/api/move) and chat tool calls both mutate the same game, and the
frontend re-syncs from /api/state. The LLM is loaded lazily and optional, so the
board + eval bar work even before the adapter is trained.

Run from repo root:  python -m backend.server [adapter_dir]
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .engine import Engine
from .game import Game
from .inference import CoachLoop, PLUGIN_CONTEXT
from . import skill_admin, state_api
from .tools import ToolExecutor

WEB = Path(__file__).resolve().parents[1] / "gemma_chat_site" / "static"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8"}


def bind_address() -> tuple[str, int]:
    return os.environ.get("CHESS_HOST", "127.0.0.1"), int(os.environ.get("CHESS_PORT", "7860"))


def agent_overlay() -> str:
    """Optional customization layer (tone/extra rules) the deployer sets. Default
    empty → serving system text == the trained contract (no drift)."""
    return os.environ.get("CHESS_AGENT_OVERLAY", "")


class App:
    def __init__(self, adapter: str | None) -> None:
        self.game = Game()
        self.engine = Engine()
        self.executor = ToolExecutor(self.game, self.engine)
        self.history: list[dict] = []
        self.loop: CoachLoop | None = None
        self.model_error: str | None = None
        self._adapter = adapter
        # live plugin envelope (copy so /api/plugin edits don't mutate the const)
        self.plugin_context = {k: list(v) for k, v in PLUGIN_CONTEXT.items()}

    def load_model(self) -> None:
        # If a LoRA adapter dir is given (arg or CHESS_HF_ADAPTER), serve the HF
        # 4-bit base + that adapter directly — lets you chat with a freshly
        # trained adapter before it is merged/exported to GGUF.
        adapter = self._adapter or os.environ.get("CHESS_HF_ADAPTER", "")
        if adapter:
            try:
                from .model_hf import HFModel
                model = HFModel(adapter=adapter, temperature=0.6)
                self.loop = CoachLoop(model, self.executor, agent_overlay(), self.plugin_context)
                print(f"model loaded (HF 4-bit base + adapter {adapter})", flush=True)
                return
            except Exception as exc:
                self.model_error = str(exc)
                print(f"HF adapter load failed ({exc}); falling back to GGUF", flush=True)
        # GGUF serving (q4_0, ~4.5 GiB, light mmap) — the shipped artifact path.
        # No GGUF -> board/eval still work but the model is marked unavailable.
        from .model_gguf import GGUFModel, default_gguf_path, gguf_runtime_config
        try:
            gguf = default_gguf_path()
            if gguf.exists():
                n_ctx, n_gpu_layers = gguf_runtime_config()
                model = GGUFModel(gguf=gguf, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
                self.loop = CoachLoop(model, self.executor, agent_overlay(), self.plugin_context)
                print(f"model loaded (GGUF {gguf.name})", flush=True)
                return
            raise FileNotFoundError(
                f"no GGUF at {gguf}; set CHESS_GGUF_PATH (serving is GGUF-only)")
        except Exception as exc:  # board + eval still work without the model
            self.model_error = str(exc)
            print(f"model unavailable ({exc}); board/eval still work", flush=True)

    def state(self) -> dict:
        return state_api.snapshot(self.game, self.engine)

    def reset(self) -> dict:
        self.game = Game()
        self.executor.game = self.game
        self.history = []
        return self.state()

    def chat(self, message: str) -> dict:
        if self.loop is None:
            return {"reply": f"(model not loaded: {self.model_error or 'no adapter'})",
                    "tool_call": None, "tool_result": None, "state": self.state()}
        result = self.loop.respond(self.history, message)
        self.history += result["turns"]
        return {"reply": result["reply"], "tool_call": result["tool_call"],
                "tool_result": result["tool_result"], "tool_calls": result.get("tool_calls", []),
                "tool_results": result.get("tool_results", []), "state": self.state()}

    def skills_payload(self) -> dict:
        return skill_admin.catalog_payload(self.plugin_context)


APP: App


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            return self._json(APP.state())
        if path == "/api/skills":
            return self._json(APP.skills_payload())
        if path in ("/", "/index.html"):
            return self._file(WEB / "index.html")
        if path.startswith("/static/"):
            return self._file(WEB / path.removeprefix("/static/"))
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._read()
            if path == "/api/move":
                ok = APP.game.move_uci(str(body.get("uci", "")))
                return self._json({"ok": ok, "state": APP.state()}, 200 if ok else 400)
            if path == "/api/reset":
                return self._json({"ok": True, "state": APP.reset()})
            if path == "/api/chat":
                msg = str(body.get("message", "")).strip()
                if not msg:
                    raise ValueError("empty message")
                return self._json({"ok": True, **APP.chat(msg)})
            if path == "/api/skill":
                skill_admin.add_skill(str(body.get("name", "")), str(body.get("description", "")),
                                      str(body.get("body", "")))
                return self._json({"ok": True, **APP.skills_payload()})
            if path == "/api/skill/delete":
                skill_admin.delete_skill(str(body.get("name", "")))
                return self._json({"ok": True, **APP.skills_payload()})
            if path == "/api/plugin":
                skill_admin.apply_plugin(APP.plugin_context, body)
                return self._json({"ok": True, **APP.skills_payload()})
            self.send_error(404)
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)}, 500)

    def _read(self) -> dict:
        n = int(self.headers.get("content-length", "0"))
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Path) -> None:
        p = path.resolve()
        if not (p.exists() and p.is_file()) or WEB not in p.parents and p != WEB / "index.html":
            return self.send_error(404)
        data = p.read_bytes()
        self.send_response(200)
        self.send_header("content-type", TYPES.get(p.suffix, "application/octet-stream"))
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        return


def main() -> None:
    global APP
    skill_admin.register()  # runtime skills dir on CHESS_SKILLS_DIRS, wiped fresh
    adapter = sys.argv[1] if len(sys.argv) > 1 else None
    APP = App(adapter)
    print(f"loading model (adapter={adapter}) ...", flush=True)
    APP.load_model()
    host, port = bind_address()
    print(f"open http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
