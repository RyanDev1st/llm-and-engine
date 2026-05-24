"""GGUF serving backend (llama-cpp-python).

Loads the Q4_0 GGUF exported from the merged adapter. ~3.2 GiB on disk and on
GPU, no bnb 4-bit mmap pressure (which trips Windows "paging file too small"
on 16 GB laptops). This is the preferred runtime for `npm run dev`.
"""
from __future__ import annotations

from pathlib import Path

LLM_DIR = Path(__file__).resolve().parents[1]
REPO = LLM_DIR.parent.parent
DEFAULT_GGUF = REPO / "runs" / "gemma4-E2B-chesscoach-Q4_0.gguf"


class GGUFModel:
    def __init__(self, gguf: str | Path = DEFAULT_GGUF, n_gpu_layers: int = -1,
                 n_ctx: int = 2048, temperature: float = 0.5) -> None:
        from llama_cpp import Llama
        self.temperature = temperature
        path = Path(gguf)
        if not path.exists():
            raise FileNotFoundError(f"GGUF not found: {path}")
        self.llm = Llama(
            model_path=str(path), n_gpu_layers=n_gpu_layers, n_ctx=n_ctx,
            n_batch=256, verbose=False, chat_format=None)

    def generate(self, messages: list[dict], max_new_tokens: int, stop: list[str]) -> str:
        # Use llama-cpp's chat completion (applies the model's chat template)
        stops = list(stop or [])
        if "</tool>" not in stops:
            stops = ["</tool>", *stops]
        out = self.llm.create_chat_completion(
            messages=messages, max_tokens=max_new_tokens,
            temperature=max(self.temperature, 0.0), top_p=0.9, stop=stops)
        text = out["choices"][0]["message"]["content"].strip()
        finish = out["choices"][0].get("finish_reason")
        # llama-cpp strips the stop sequence; restore </tool> if that was the stop
        if finish == "stop" and text.startswith("<tool>") and "</tool>" not in text:
            text += "</tool>"
        return text
