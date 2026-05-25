from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .paths import OUT
from .validate import assert_valid

ROOT = Path(__file__).resolve().parents[4]
TRAIN = ROOT / "data" / "sft" / "v1_train.jsonl"
VAL = ROOT / "data" / "sft" / "v1_val.jsonl"


def load_accepted(gold_dir: Path = OUT) -> list[dict]:
    path = gold_dir / "accepted.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(gold_dir: Path = OUT, train_path: Path = TRAIN, val_path: Path = VAL) -> tuple[int, int]:
    rows = load_accepted(gold_dir)
    for row in rows:
        assert_valid(row)
    train, val = [], []
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["slice"]].append(row)
    for slice_name in sorted(buckets):
        for idx, row in enumerate(buckets[slice_name]):
            target = val if idx % 10 == 0 else train
            target.append(row)
    _write(train_path, train)
    _write(val_path, val)
    return len(train), len(val)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    train, val = build()
    print(f"wrote train={train} val={val}")
