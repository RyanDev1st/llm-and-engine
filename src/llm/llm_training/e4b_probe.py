"""Single-GPU E4B QLoRA memory probe — does E4B fit ONE 16 GB T4 at a given seq len,
BEFORE committing a multi-hour Colab/Kaggle slot? Standalone, no DDP (the past OOMs were
DDP replication on 2xT4, not E4B being too big — see ddp_probe.py + the train/host split).

Run (Colab single T4, or any one GPU):
    CHESS_BASE=/content/gemma4_e4b PROBE_SEQ=1152 python -m llm_training.e4b_probe

It loads 4-bit E4B + LoRA exactly like train_cuda (nf4 + double-quant + bf16 compute,
gradient checkpointing, 8-bit paged AdamW), runs a few real train steps at batch=1, and
prints PEAK VRAM + a FIT / TIGHT / OOM verdict. Defaults to the CONSERVATIVE config
(all-linear LoRA, the heavier target set) so a FIT verdict also holds for attn-only.

Env: CHESS_BASE (model dir, required), PROBE_SEQ (default 1152), PROBE_STEPS (6),
PROBE_TARGETS ("all-linear" | "attn-only"), PROBE_RANK (16)."""
import os

# Set BEFORE importing torch: the #1 anti-OOM-from-fragmentation flag (matches the trainer).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

BASE = os.environ.get("CHESS_BASE", "")
SEQ = int(os.environ.get("PROBE_SEQ", "1152"))
STEPS = int(os.environ.get("PROBE_STEPS", "6"))
TARGETS = os.environ.get("PROBE_TARGETS", "all-linear")
RANK = int(os.environ.get("PROBE_RANK", "16"))
_ATTN = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$"


def _lora() -> LoraConfig:
    tm = "all-linear" if TARGETS == "all-linear" else _ATTN
    return LoraConfig(task_type="CAUSAL_LM", r=RANK, lora_alpha=2 * RANK,
                      lora_dropout=0.05, target_modules=tm)


def _optimizer(params):
    try:
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(params, lr=2e-4)   # matches paged_adamw_8bit
    except Exception as e:  # noqa: BLE001
        print(f"bnb 8-bit optim unavailable ({e}); using torch AdamW (uses MORE memory)", flush=True)
        return torch.optim.AdamW(params, lr=2e-4)


def main() -> None:
    if not BASE:
        raise SystemExit("set CHESS_BASE to the E4B model dir (e.g. /content/gemma4_e4b)")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device — run this on a GPU (Colab T4 / Kaggle T4)")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {name}  ({total:.1f} GB)  | seq={SEQ} targets={TARGETS} rank={RANK}", flush=True)

    # T4 (Turing sm_75) has NO bf16 — use fp16 there or bnb errors / silently misleads.
    # Match the real Unsloth path (dtype=None auto-picks fp16 on T4) so the probe is honest.
    dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"compute dtype = {dt} (bf16_supported={torch.cuda.is_bf16_supported()})", flush=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dt, bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, local_files_only=True, quantization_config=quant,
        torch_dtype=dt, device_map={"": 0})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model = get_peft_model(model, _lora())
    model.train()
    opt = _optimizer([p for p in model.parameters() if p.requires_grad])

    ids = torch.randint(0, 1000, (1, SEQ), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    try:
        for i in range(STEPS):
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward(); opt.step(); opt.zero_grad()
            torch.cuda.synchronize()
            print(f"  step {i+1}/{STEPS} ok  peak={torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f"\nOOM at seq={SEQ}. Lower PROBE_SEQ (try {max(512, SEQ - 256)}) or use attn-only "
              f"targets / a shorter cap.", flush=True)
        raise SystemExit(1)

    peak = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9
    head = total - reserved
    verdict = "FIT (comfortable)" if head > 2.0 else ("TIGHT (<2 GB headroom)" if head > 0.3 else "OOM-RISK")
    print(f"\nPEAK allocated={peak:.2f} GB  reserved={reserved:.2f} GB  headroom={head:.2f} GB  -> {verdict}",
          flush=True)
    print("note: 'all-linear' is the heavy config; attn-only will use less. Eval/generation adds "
          "a transient spike — keep ~1-2 GB headroom.", flush=True)


if __name__ == "__main__":
    main()
