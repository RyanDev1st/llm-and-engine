"""Stress test Gemma 4 E4B q4_0 GGUF (text-only) on local GPU via llama.cpp."""
from __future__ import annotations

import subprocess
import time

from llama_cpp import Llama

REPO = "google/gemma-4-E4B-it-qat-q4_0-gguf"
FILENAME = "gemma-4-E4B_q4_0-it.gguf"


def gpu_used() -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def main() -> None:
    print("gpu before:", gpu_used(), flush=True)
    t0 = time.time()
    llm = Llama.from_pretrained(
        repo_id=REPO,
        filename=FILENAME,
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=False,
    )
    print(f"load {time.time() - t0:.1f}s", flush=True)
    print("gpu after load:", gpu_used(), flush=True)

    msgs = [
        {
            "role": "user",
            "content": "You are a chess coach. In 3 sentences, explain why "
            "controlling the center matters in the opening.",
        }
    ]
    t1 = time.time()
    out = llm.create_chat_completion(messages=msgs, max_tokens=128, temperature=0.0)
    dt = time.time() - t1
    txt = out["choices"][0]["message"]["content"]
    ntok = out["usage"]["completion_tokens"]
    print(f"gen {ntok} tok in {dt:.1f}s = {ntok / max(dt, 1e-6):.1f} tok/s", flush=True)
    print("gpu after gen:", gpu_used(), flush=True)
    print("=== OUTPUT ===", flush=True)
    print(txt, flush=True)


if __name__ == "__main__":
    main()
