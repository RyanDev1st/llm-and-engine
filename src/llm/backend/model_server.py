"""Persistent model service — loads the heavy model ONCE and serves generation over
localhost HTTP, so the logic/web server can restart in ~1s without reloading weights.

Dev workflow:
  1. once:  python -m backend.model_server "A:/Download/gemma4_chess_kaggle_adapter (1)"
            (loads the weights on the GPU, stays up)
  2. app:   CHESS_MODEL_SERVER=http://127.0.0.1:7861 python -m backend.server
            (weightless — connects to this service; restart it as often as you like)

Only restart THIS process when you change model-loading code. Editing inference/
tool logic only needs an app-server restart, which no longer touches the weights.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def build_model() -> tuple[object, bool]:
    """Return (model, has_adapter). HF adapter path first (live runtime + the
    adapter-on/off compare), GGUF fallback. Mirrors web_app.load_model."""
    adapter = (sys.argv[1] if len(sys.argv) > 1 else "") or os.environ.get("CHESS_HF_ADAPTER", "")
    if adapter:
        try:
            from .model_hf import HFModel
            return HFModel(adapter=adapter, temperature=0.0), True
        except Exception as exc:  # noqa: BLE001
            print(f"HF adapter load failed ({exc}); falling back to GGUF", flush=True)
    from .model_gguf import GGUFModel, default_gguf_path, gguf_runtime_config
    gguf = default_gguf_path()
    n_ctx, n_gpu_layers = gguf_runtime_config()
    return GGUFModel(gguf=gguf, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers), False


MODEL: object | None = None
HAS_ADAPTER = False
_LOCK = threading.Lock()  # one GPU generate at a time (demo is single-user/sequential)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        n = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        if self.path == "/generate":
            kw = {"use_adapter": bool(body.get("use_adapter", True))} if HAS_ADAPTER else {}
            with _LOCK:
                text = MODEL.generate(body["messages"], int(body.get("max_new_tokens", 128)),
                                      list(body.get("stop", [])), **kw)
            return self._json({"text": text})
        if self.path == "/count_tokens":
            return self._json({"n": MODEL.count_tokens(str(body.get("text", "")))})
        self.send_error(404)

    def do_GET(self) -> None:
        if self.path == "/context_limit":
            return self._json({"n": MODEL.context_limit()})
        if self.path == "/health":
            return self._json({"ok": MODEL is not None, "adapter": HAS_ADAPTER})
        self.send_error(404)

    def _json(self, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        return


def main() -> None:
    global MODEL, HAS_ADAPTER
    print("loading model service ...", flush=True)
    MODEL, HAS_ADAPTER = build_model()
    host = os.environ.get("CHESS_MODEL_HOST", "127.0.0.1")
    port = int(os.environ.get("CHESS_MODEL_PORT", "7861"))
    print(f"model service up on http://{host}:{port} (adapter={HAS_ADAPTER}) — leave it running", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
