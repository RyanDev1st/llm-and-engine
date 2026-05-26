from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@dataclass(frozen=True)
class TrainConfig:
    phase: str = "unified"
    device: str = "cpu"
    allow_cuda: bool = False
    dry_run: bool = True
    smoke: bool = False
    max_steps: int = 0
    max_examples: int = 100000
    batch_size: int = 1
    grad_accum_steps: int = 16
    epochs: int = 3
    max_seq_len: int = 1024
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_targets: str = "qv"
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    optimizer: str = "paged_adamw_8bit"
    eval_every: int = 50
    shuffle: bool = True
    loss_mask: str = "assistant-only"
    model_path: Path = Path("src/models/gemma4")
    output_dir: Path = Path("runs/gemma4_lora")
    max_gpu_memory: str = "7GiB"
    load_in_4bit: bool = True
    data_path: Path | None = None
    val_path: Path | None = None
    seed: int = 42


def run_training(config: TrainConfig) -> dict:
    if config.phase not in {"router", "narrator", "unified"}:
        raise ValueError("phase must be router, narrator, or unified")
    if config.device == "cuda" and not config.allow_cuda:
        raise ValueError("CUDA requires --allow-cuda")
    if config.smoke:
        if config.max_steps > 5:
            raise ValueError("smoke max_steps must be <= 5")
        if config.max_examples > 32:
            raise ValueError("smoke max_examples must be <= 32")
    exists = config.model_path.exists()
    if config.dry_run or not exists:
        return _dry_result(config, exists)
    from .train_cuda import _real_training
    return _real_training(config)


def _dry_result(config: TrainConfig, exists: bool) -> dict:
    return {
        "phase": config.phase,
        "device": config.device,
        "dry_run": config.dry_run,
        "smoke": config.smoke,
        "max_steps": config.max_steps,
        "max_examples": config.max_examples,
        "model_path": str(config.model_path),
        "model_exists": exists,
        "training_started": False,
    }
