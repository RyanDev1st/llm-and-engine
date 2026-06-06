from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .paths import OUT, TRAIN, VAL
from .profiles import profile
from .validate import assert_valid

ROOT = Path(__file__).resolve().parents[4]


def load_accepted(gold_dir: Path = OUT) -> list[dict]:
    path = gold_dir / "accepted.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _final(row: dict) -> str:
    for message in reversed(row.get("messages", [])):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def split_train_val(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Stratified 10% holdout, then de-leaked: any val row whose final-answer
    text already appears in train is moved into train so the val metric measures
    generalization, not memorization."""
    train: list[dict] = []
    val: list[dict] = []
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["slice"]].append(row)
    for slice_name in sorted(buckets):
        for idx, row in enumerate(buckets[slice_name]):
            (val if idx % 10 == 0 else train).append(row)
    train_finals = {_final(r) for r in train}
    deleaked: list[dict] = []
    for row in val:
        if _final(row) in train_finals:
            train.append(row)            # leak -> move into train (no data lost)
        else:
            deleaked.append(row)
    return train, deleaked


def build(gold_dir: Path = OUT, train_path: Path = TRAIN, val_path: Path = VAL) -> tuple[int, int]:
    rows = load_accepted(gold_dir)
    for row in rows:
        assert_valid(row)
    train, val = split_train_val(rows)
    _write(train_path, train)
    _write(val_path, val)
    return len(train), len(val)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="v1.2")
    args = parser.parse_args()
    p = profile(args.profile)
    train, val = build(p.gold_dir, p.train_path, p.val_path)
    print(f"wrote train={train} val={val}")
