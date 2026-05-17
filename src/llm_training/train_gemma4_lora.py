from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    phase: str
    device: str = "cpu"
    allow_cuda: bool = False
    dry_run: bool = True
    smoke: bool = False
    max_steps: int = 1
    max_examples: int = 8
    model_path: Path = Path("src/models/gemma4_q4")


def run_training(config: TrainConfig) -> dict:
    if config.phase not in {"router", "narrator"}:
        raise ValueError("phase must be router or narrator")
    if config.device == "cuda" and not config.allow_cuda:
        raise ValueError("CUDA requires --allow-cuda")
    if config.max_steps > 5 and config.smoke:
        raise ValueError("smoke max_steps must be <= 5")
    if config.max_examples > 32 and config.smoke:
        raise ValueError("smoke max_examples must be <= 32")
    exists = config.model_path.exists()
    return {
        "phase": config.phase,
        "device": config.device,
        "dry_run": config.dry_run,
        "smoke": config.smoke,
        "max_steps": config.max_steps,
        "max_examples": config.max_examples,
        "model_path": str(config.model_path),
        "model_exists": exists,
        "training_started": not config.dry_run and exists,
    }
