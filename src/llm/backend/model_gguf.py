"""GGUF serving backend (llama-cpp-python).

Loads the Q4_0 GGUF exported from the merged adapter. ~3.2 GiB on disk and on
GPU, no bnb 4-bit mmap pressure (which trips Windows "paging file too small"
on 16 GB laptops). This is the preferred runtime for `npm run dev`.
"""
from __future__ import annotations

from pathlib import Path
import os

LLM_DIR = Path(__file__).resolve().parents[1]
REPO = LLM_DIR.parent.parent
# Q5_K_M (not Q4_0): better number/interpretation fidelity for the eval-grounding the
# coach narrates, at ~3.6 GB (vs 3.35). Q4_0 fabricated eval numbers; Q5_K_M doesn't.
DEFAULT_GGUF = REPO / "runs" / "gemma4-E2B-chesscoach-Q5_K_M.gguf"


def default_gguf_path() -> Path:
    return Path(os.environ.get("CHESS_GGUF_PATH", DEFAULT_GGUF))


def gguf_runtime_config() -> tuple[int, int]:
    return int(os.environ.get("CHESS_N_CTX", "4096")), int(os.environ.get("CHESS_N_GPU_LAYERS", "-1"))


class GGUFModel:
    def __init__(self, gguf: str | Path | None = None, n_gpu_layers: int = -1,
                 n_ctx: int = 4096, temperature: float = 0.5) -> None:
        from llama_cpp import Llama, LlamaRAMCache
        self.temperature = temperature
        path = Path(gguf) if gguf is not None else default_gguf_path()
        if not path.exists():
            raise FileNotFoundError(f"GGUF not found: {path}")
        self.llm = Llama(
            model_path=str(path), n_gpu_layers=n_gpu_layers, n_ctx=n_ctx,
            n_batch=256, verbose=False, chat_format=None)
        # Prefix/KV cache: the agentic loop calls generate() several times per turn
        # with a GROWING prompt that shares a long prefix (system + history + earlier
        # steps). The cache lets llama.cpp reuse the KV for that shared prefix instead
        # of re-prefilling ~1000+ tokens every step — the main per-turn latency. On by
        # default; CHESS_GGUF_CACHE=0 disables it for an A/B.
        if os.environ.get("CHESS_GGUF_CACHE", "1") != "0":
            cap = int(os.environ.get("CHESS_GGUF_CACHE_BYTES", str(1 << 30)))  # 1 GiB
            self.llm.set_cache(LlamaRAMCache(capacity_bytes=cap))

    def generate(self, messages: list[dict], max_new_tokens: int, stop: list[str],
                 on_token=None) -> str:
        if on_token is not None:   # true token streaming (overlaps generation with delivery)
            return self.generate_stream(messages, max_new_tokens, stop, on_token)
        stops = list(stop or [])
        out = self.llm.create_chat_completion(
            messages=messages, max_tokens=max_new_tokens,
            temperature=max(self.temperature, 0.0), top_p=0.9, stop=stops)
        text = out["choices"][0]["message"]["content"].strip()
        finish = out["choices"][0].get("finish_reason")
        if finish == "stop" and "</tool>" in stops and text.startswith("<tool>") and "</tool>" not in text:
            text += "</tool>"
        return text

    def generate_stream(self, messages: list[dict], max_new_tokens: int, stop: list[str],
                        on_token) -> str:
        """Stream tokens as llama.cpp produces them, calling on_token(delta) per chunk;
        returns the full text. Lets the UI show output live instead of after the block."""
        stops = list(stop or [])
        full = ""
        for chunk in self.llm.create_chat_completion(
                messages=messages, max_tokens=max_new_tokens,
                temperature=max(self.temperature, 0.0), top_p=0.9, stop=stops, stream=True):
            delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
            if not delta:
                continue
            full += delta
            try:
                on_token(delta)
            except Exception:
                pass  # client disconnected — keep generating so the turn still completes
        full = full.strip()
        if "</tool>" in stops and full.startswith("<tool>") and "</tool>" not in full:
            full += "</tool>"
        return full

    def count_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))

    def context_limit(self) -> int:
        return int(self.llm.n_ctx())
