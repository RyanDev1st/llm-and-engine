"""Standalone 2-GPU DDP throughput probe (experiment, not training).

Run via accelerate (subprocess) so there are NO notebook_launcher main-process
constraints:
    accelerate launch --num_processes 2 --multi_gpu ddp_probe.py   # data-parallel
    accelerate launch --num_processes 1            ddp_probe.py     # single-GPU baseline

Loads 4-bit E2B + RANK=16 attn-only LoRA on each process's GPU, times 10 dummy
steps, prints throughput on the main process. Set CHESS_BASE to the model dir.
"""
import os
import time

import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

BASE = os.environ.get("CHESS_BASE", "/kaggle/working/e2b_test")
SEQ = 512
STEPS = 10
WARMUP = 3


def main() -> None:
    acc = Accelerator()
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, local_files_only=True, quantization_config=quant,
        torch_dtype=torch.bfloat16, device_map={"": acc.process_index})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$"))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model, opt = acc.prepare(model, opt)

    ids = torch.randint(0, 1000, (1, SEQ), device=acc.device)
    for _ in range(WARMUP):
        loss = model(input_ids=ids, labels=ids).loss
        acc.backward(loss); opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(STEPS):
        loss = model(input_ids=ids, labels=ids).loss
        acc.backward(loss); opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    sps = (time.time() - t) / STEPS
    acc.print(f"world={acc.num_processes}  {sps:.2f}s/step/proc  ->  throughput {acc.num_processes / sps:.3f} ex/s")


if __name__ == "__main__":
    main()
