"""Stress test google/gemma-4-E4B-it-qat-q4_0-unquantized on local GPU (4-bit)."""
from __future__ import annotations

import time

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

MODEL = "google/gemma-4-E4B-it-qat-q4_0-unquantized"


def main() -> None:
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
    print("gpu", torch.cuda.get_device_name(0), flush=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        quantization_config=bnb,
        device_map="cuda",
        dtype=torch.bfloat16,
    )
    model.eval()
    print(f"load {time.time() - t0:.1f}s", flush=True)
    print("vram alloc GB", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
    print("vram reserved GB", round(torch.cuda.memory_reserved() / 1e9, 2), flush=True)

    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "You are a chess coach. In 3 sentences, explain why "
                    "controlling the center matters in the opening.",
                }
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    torch.cuda.synchronize()
    t1 = time.time()
    out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    torch.cuda.synchronize()
    gen_s = time.time() - t1

    new = out[0][inputs["input_ids"].shape[-1]:]
    txt = processor.decode(new, skip_special_tokens=True)
    n = int(new.shape[-1])
    print(f"gen {n} tok in {gen_s:.1f}s = {n / max(gen_s, 1e-6):.1f} tok/s", flush=True)
    print("peak vram GB", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
    print("=== OUTPUT ===", flush=True)
    print(txt, flush=True)


if __name__ == "__main__":
    main()
