from __future__ import annotations

from typing import Any

import torch


def build_optimizer(model: Any, name: str, lr: float, weight_decay: float) -> Any:
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "paged_adamw_8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        except Exception as e:
            print(f"paged_adamw_8bit unavailable ({e}); falling back to torch AdamW", flush=True)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))


def build_scheduler(optimizer: Any, num_warmup: int, num_total: int) -> Any:
    from transformers import get_cosine_schedule_with_warmup
    return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup, num_training_steps=num_total)


def lora_target_modules(spec: str) -> str | list[str]:
    if spec == "all-linear":
        return r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    if spec == "attn-only":
        return r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$"
    if spec == "qv":
        return r".*language_model.*\.(q_proj|v_proj)$"
    return spec
