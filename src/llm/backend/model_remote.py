"""Thin HTTP client to the persistent model service (model_server.py). Lets the
logic/web server restart instantly: the heavy weights live in the model service;
this just forwards generate/count_tokens/context_limit over localhost. Drop-in for
HFModel/GGUFModel — same method shapes, so AdapterView and CoachLoop use it unchanged."""
from __future__ import annotations

import json
import os
import urllib.request


def server_url() -> str:
    return os.environ.get("CHESS_MODEL_SERVER", "http://127.0.0.1:7861").rstrip("/")


def server_has_adapter(timeout: float = 5.0) -> bool:
    """Ask the service whether it loaded an adapter (so the app knows to build the
    adapter-on/off compare loops). Raises if the service isn't reachable."""
    with urllib.request.urlopen(server_url() + "/health", timeout=timeout) as r:
        return bool(json.loads(r.read())["adapter"])


class RemoteModel:
    def __init__(self, has_adapter: bool = True) -> None:
        self.has_adapter = has_adapter
        self._limit: int | None = None

    def _post(self, path: str, payload: dict, timeout: float = 600.0) -> dict:
        req = urllib.request.Request(
            server_url() + path, data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def generate(self, messages: list[dict], max_new_tokens: int, stop: list[str],
                 use_adapter: bool = True, on_token=None, enable_thinking=None) -> str:
        # enable_thinking is forwarded so the CoachLoop's per-mode native thinking (think/auto on,
        # fast off) reaches the model service across the process split. The loop only passes it when
        # the signature advertises it — declaring it here is what flips the loop's can_think check on.
        payload = {"messages": messages, "max_new_tokens": max_new_tokens, "stop": stop}
        if enable_thinking is not None:
            payload["enable_thinking"] = bool(enable_thinking)
        if self.has_adapter:
            payload["use_adapter"] = use_adapter
        if on_token is None:
            return self._post("/generate", payload)["text"]
        # Stream: consume the service's token SSE, forward each delta, return the full text.
        payload["stream"] = True
        req = urllib.request.Request(
            server_url() + "/generate", data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"})
        full = ""
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                try:
                    ev = json.loads(line[6:])
                except Exception:
                    continue
                if "t" in ev:
                    full += ev["t"]
                    try:
                        on_token(ev["t"])
                    except Exception:
                        pass
                elif ev.get("done"):
                    full = ev.get("text", full)
        return full

    def count_tokens(self, text: str) -> int:
        return int(self._post("/count_tokens", {"text": text})["n"])

    def context_limit(self) -> int:
        if self._limit is None:
            with urllib.request.urlopen(server_url() + "/context_limit", timeout=30) as r:
                self._limit = int(json.loads(r.read())["n"])
        return self._limit
