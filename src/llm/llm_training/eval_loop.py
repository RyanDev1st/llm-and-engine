from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def evaluate(model: Any, batches: list[dict], target_device: torch.device) -> float:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from .train_cuda import fused_masked_loss
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        for raw in batches:
            batch = {k: v.to(target_device) for k, v in raw.items()}
            labels = batch.pop("labels")
            batch.pop("weights", None)  # val loss stays unweighted for comparability
            loss = fused_masked_loss(model, batch, labels)
            n = int((labels != -100).sum().item())
            total_loss += loss.item() * n
            total_tokens += n
    model.train()
    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens
