"""Ollama serving backend."""
from __future__ import annotations

import json
import os
import urllib.request

DEFAULT_OLLAMA_MODEL = "qwen3.6:27b-q4_K_M"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


def ollama_model_name() -> str:
    return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")


class OllamaModel:
    def __init__(self, model: str | None = None, host: str | None = None, temperature: float = 0.5) -> None:
        self.model = model or ollama_model_name()
        self.host = (host or ollama_host()).rstrip("/")
        self.temperature = temperature

    def generate(self, messages: list[dict], max_new_tokens: int, stop: list[str]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": max(self.temperature, 0.0),
                "top_p": 0.9,
                "stop": list(stop or []),
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=300) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        text = out.get("message", {}).get("content", "").strip()
        if "</tool>" in (stop or []) and text.startswith("<tool>") and "</tool>" not in text:
            text += "</tool>"
        return text
