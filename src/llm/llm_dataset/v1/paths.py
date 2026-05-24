from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "data" / "sft" / "v1_gold"
TRAIN = ROOT / "data" / "sft" / "v1_train.jsonl"
VAL = ROOT / "data" / "sft" / "v1_val.jsonl"
