"""Standalone DDP fit + throughput probe (the go/no-go BEFORE a real 2-GPU run).

Run via accelerate (or torchrun) so there are NO notebook_launcher constraints:
    accelerate launch --num_processes 2 --multi_gpu -m llm_training.ddp_probe   # data-parallel
    accelerate launch --num_processes 1            -m llm_training.ddp_probe    # 1-GPU baseline

Loads 4-bit base + attn-only LoRA on EACH process's GPU (a full replica = DDP), times
a few real steps at the REAL seq, prints per-rank peak VRAM + throughput. Run the 2-proc
and 1-proc forms; the ratio is your DDP speedup, and a non-OOM 2-proc run is the gate.

Env: CHESS_BASE (model dir, required), PROBE_SEQ (default 1664), PROBE_RANK (16),
PROBE_STEPS (8), PROBE_WARMUP (3).
"""
import os
import time

import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

BASE = os.environ.get("CHESS_BASE", "")
SEQ = int(os.environ.get("PROBE_SEQ", "1664"))
RANK = int(os.environ.get("PROBE_RANK", "16"))
STEPS = int(os.environ.get("PROBE_STEPS", "8"))
WARMUP = int(os.environ.get("PROBE_WARMUP", "3"))
_ATTN = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$"


def main() -> None:
    if not BASE:
        raise SystemExit("set CHESS_BASE to the model dir (e.g. .../models/gemma4_e4b)")
    acc = Accelerator()
    # T4 (Turing) has NO bf16 — fp16 there or bnb misleads/errors. Matches the real path.
    dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dt, bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, local_files_only=True, quantization_config=quant,
        torch_dtype=dt, device_map={"": acc.process_index})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=RANK, lora_alpha=2 * RANK, lora_dropout=0.05,
        target_modules=_ATTN))
    # THE checkpointing fix — frozen base means the embed output needs an input-grad hook
    # or torch.checkpoint no-ops and activations blow up (the whole single-GPU OOM saga).
    model.train()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model, opt = acc.prepare(model, opt)

    ids = torch.randint(0, 1000, (1, SEQ), device=acc.device)
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(WARMUP):
            loss = model(input_ids=ids, labels=ids).loss
            acc.backward(loss); opt.step(); opt.zero_grad()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(STEPS):
            loss = model(input_ids=ids, labels=ids).loss
            acc.backward(loss); opt.step(); opt.zero_grad()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        acc.print(f"OOM at seq={SEQ} rank={RANK} world={acc.num_processes} — DDP replica too big "
                  f"for this GPU. Lower PROBE_RANK or seq, or DDP not viable here.")
        raise SystemExit(1)
    sps = (time.time() - t) / STEPS
    peak = torch.cuda.max_memory_allocated() / 1e9
    acc.print(f"world={acc.num_processes} seq={SEQ} rank={RANK}  {sps:.2f}s/step/proc  "
              f"peak/GPU={peak:.2f}GB  throughput={acc.num_processes / sps:.3f} ex/s")


if __name__ == "__main__":
    main()
