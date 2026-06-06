from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
# Active corpus is v1_2 (the only one trainers read). Older v1_gold/v1_train/v1_val
# were archived to "legacy [ignore]/"; defaults point at v1_2 so no live path is dead.
OUT = ROOT / "data" / "sft" / "v1_2"
TRAIN = ROOT / "data" / "sft" / "v1_2_train.jsonl"
VAL = ROOT / "data" / "sft" / "v1_2_val.jsonl"
