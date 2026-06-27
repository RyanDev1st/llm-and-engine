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
    pure_chess: bool = False        # v5: flat chess catalog -> skip the cross-domain/plugin
                                    # audit checks, enforce the grounded-why floor instead
    max_seq: int = 1664             # train seq ceiling the token-length gate enforces


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


V5 = DatasetProfile(
    name="v5",
    gold_dir=ROOT / "data" / "sft" / "v5",
    train_path=ROOT / "data" / "sft" / "v5_train.jsonl",
    val_path=ROOT / "data" / "sft" / "v5_val.jsonl",
    accepted_target=12_000,
    rejected_target=1_200,
    rejected_min=800,
    rejected_max=2_000,
    min_plugin_sources=0,           # flat catalog: no plugin provenance to diversify
    max_prompt_concentration=0.02,
    pure_chess=True,
)


def profile(name: str = "v1.2") -> DatasetProfile:
    if name in ("v1.2", "v1_2"):
        return V1_2
    if name in ("v5", "v5.0"):
        return V5
    raise ValueError(f"unknown profile: {name} (active: v1.2, v5)")
