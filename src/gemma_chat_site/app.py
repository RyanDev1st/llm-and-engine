from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from model_service import SERVICE

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
MAX_BODY_BYTES = 65536
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css"}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self.write_json({"ok": SERVICE.load_error is None, **SERVICE.status()})
            return
        if path in ("/", "/index.html"):
            self.write_file(STATIC / "index.html")
            return
        if path.startswith("/static/"):
            self.write_file(STATIC / path.removeprefix("/static/"))
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            self.send_error(404)
            return
        try:
            body = self.read_json()
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a non-empty list")
            clean = [self.clean_message(item) for item in messages]
            result = SERVICE.generate(clean, body.get("max_new_tokens", 128), body.get("temperature", 0.7))
            self.write_json({"ok": True, **result})
        except Exception as exc:
            self.write_json({"ok": False, "error": str(exc), "status": SERVICE.status()}, 500)

    def clean_message(self, item: object) -> dict[str, str]:
        if not isinstance(item, dict):
            raise ValueError("message must be an object")
        role = str(item.get("role", "user"))
        content = str(item.get("content", "")).strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError("invalid message role")
        if not content:
            raise ValueError("message content is required")
        return {"role": role, "content": content}

    def read_json(self) -> dict:
        length = int(self.headers.get("content-length", "0"))
        if length > MAX_BODY_BYTES:
            raise ValueError("request body too large")
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def write_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def write_file(self, path: Path) -> None:
        resolved = path.resolve()
        if STATIC not in resolved.parents and resolved != STATIC / "index.html":
            self.send_error(403)
            return
        if not resolved.exists() or not resolved.is_file():
            self.send_error(404)
            return
        data = resolved.read_bytes()
        self.send_response(200)
        self.send_header("content-type", TYPES.get(resolved.suffix, "application/octet-stream"))
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    host = "127.0.0.1"
    port = 7860
    print(f"Loading real model from {SERVICE.model_path}", flush=True)
    try:
        SERVICE.load()
    except Exception as exc:
        print(f"Model load failed: {exc}", flush=True)
    print(f"Open http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
