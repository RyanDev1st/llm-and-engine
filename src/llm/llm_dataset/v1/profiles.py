from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import ROOT


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    gold_dir: Path
    train_path: Path
    val_path: Path
    accepted_target: int
    rejected_target: int
    rejected_min: int
    rejected_max: int
    min_plugin_sources: int
    max_prompt_concentration: float


V1_2 = DatasetProfile(
    name="v1.2",
    gold_dir=ROOT / "data" / "sft" / "v1_2",
    train_path=ROOT / "data" / "sft" / "v1_2_train.jsonl",
    val_path=ROOT / "data" / "sft" / "v1_2_val.jsonl",
    accepted_target=75_000,
    rejected_target=7_500,
    rejected_min=5_000,
    rejected_max=10_000,
    min_plugin_sources=4,
    max_prompt_concentration=0.02,
)


def profile(name: str = "v1.2") -> DatasetProfile:
    if name in ("v1.2", "v1_2"):
        return V1_2
    raise ValueError(f"unknown profile: {name} (only v1.2 is active)")
