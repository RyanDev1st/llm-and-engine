from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .jsonl_io import read_rows, write_rows
from .paths import OUT, TRAIN, VAL
from .profiles import profile
from .validate import assert_valid

ROOT = Path(__file__).resolve().parents[4]


def load_accepted(gold_dir: Path = OUT) -> list[dict]:
    return list(read_rows(gold_dir / "accepted.jsonl"))


def _final(row: dict) -> str:
    for message in reversed(row.get("messages", [])):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def _row_hash(row: dict) -> str:
    """Full-row identity (messages), so the floor never keeps an EXACT train dup in
    val (that would be real leakage). Matches final_corpus_audit.row_hash."""
    return hashlib.sha1(
        json.dumps(row.get("messages", []), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


VAL_FLOOR = 12   # min val rows kept PER SLICE for post-train per-slice eval coverage


def split_train_val(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Stratified 10% holdout, then de-leaked with a per-slice floor.

    De-leak: a val row whose final-answer text already appears in train is a
    memorization probe, not a generalization one, so prefer to move it into train.
    BUT low-answer-diversity slices (chess templates, fixed-lesson V1_Q/I/J, ...)
    have FEW distinct finals, so blanket de-leak empties their val entirely — which
    blinds post-train per-slice eval (eval_routing reports per-slice routing accuracy
    and needs rows present; routing cares about the first-turn tool, not final text).
    So we keep all CLEAN (non-overlapping) val rows, then top each slice back up to
    VAL_FLOOR from its overlapping rows. The real leak guard stays exact-row-dup
    (still 0); aggregate val_loss is dominated by the clean majority, and the floored
    rows give routing eval the coverage it needs."""
    train: list[dict] = []
    val_by_slice: dict[str, list[dict]] = defaultdict(list)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row["slice"]].append(row)
    for slice_name in sorted(buckets):
        for idx, row in enumerate(buckets[slice_name]):
            (val_by_slice[slice_name].append(row) if idx % 10 == 0 else train.append(row))
    train_finals = {_final(r) for r in train}
    train_hashes = {_row_hash(r) for r in train}
    deleaked: list[dict] = []
    for slice_name, vrows in sorted(val_by_slice.items()):
        clean = [r for r in vrows if _final(r) not in train_finals]
        # floor top-up candidates: final overlaps train BUT the full row is NOT an exact
        # train dup (keeping an exact dup = real leakage). Truly degenerate slices (every
        # row identical to a train row, e.g. V1_Q) yield none -> they just get less/no val.
        topup = [r for r in vrows if _final(r) in train_finals and _row_hash(r) not in train_hashes]
        keep = list(clean)
        floor = min(VAL_FLOOR, len(vrows))
        if len(keep) < floor:
            keep.extend(topup[: floor - len(keep)])
        kept = {id(r) for r in keep}
        deleaked.extend(keep)
        train.extend(r for r in vrows if id(r) not in kept)   # rest -> train
    # Final guard: a val row can have an EXACT twin among the rows we just moved to
    # train (identical full row, same slice). Drop those train dups so no val row is
    # byte-identical to a train row (exact dups in train are redundant memorization).
    val_hashes = {_row_hash(r) for r in deleaked}
    train = [r for r in train if _row_hash(r) not in val_hashes]
    return train, deleaked


def build(gold_dir: Path = OUT, train_path: Path = TRAIN, val_path: Path = VAL) -> tuple[int, int]:
    rows = load_accepted(gold_dir)
    for row in rows:
        assert_valid(row)
    train, val = split_train_val(rows)
    write_rows(train_path, train)
    write_rows(val_path, val)
    return len(train), len(val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="v1.2")
    args = parser.parse_args()
    p = profile(args.profile)
    train, val = build(p.gold_dir, p.train_path, p.val_path)
    print(f"wrote train={train} val={val}")
