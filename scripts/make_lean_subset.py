"""M1 — stratified LEAN subset of v1_2 for faster epochs on the 2xT4 budget.

The corpus is lopsided: V1_O cross-domain routing is ~25% (18.7k rows) and several
universality slices sit near the same size, so one epoch wastes compute on redundant
routing while rarer behaviors (e.g. V1_R python-verify, chess slices) are thin. This
downsamples the heavy slices and KEEPS the rare/valuable ones in full, producing a
smaller, BETTER-BALANCED corpus -> ~1 epoch costs far fewer GPU-hours, and the balance
is itself a quality win (less canned-routing over-fit).

It samples (seeded, reproducible) from the COMMITTED v1_2 split — every row already
passed the gate, so the subset is gate-clean by construction (no new rows, no re-gate).

Run from repo root:
    python scripts/make_lean_subset.py            # writes data/sft/v1_2_lean_{train,val}.jsonl.gz
    python scripts/make_lean_subset.py --o-cap 4500 --global-cap 2200
Point the trainer at it with DATA=v1_2_lean in the notebook (or --data-path).
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "data" / "sft"
SEED = 20260614

# Slices kept in FULL regardless of caps: the Stage-0 compute slice + the chess
# flagship slices (already small). Everything else is capped.
KEEP_FULL = {"V1_R_compute_grounding", *list("ABCDEFGHIJK")}


def read_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def subsample(rows: list[dict], o_cap: int, global_cap: int) -> list[dict]:
    by_slice: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_slice[r.get("slice")].append(r)
    rng = random.Random(SEED)
    out: list[dict] = []
    for sl, group in by_slice.items():
        if sl in KEEP_FULL:
            cap = len(group)
        elif sl == "V1_O_cross_domain_skill_routing":
            cap = o_cap
        else:
            cap = global_cap
        if len(group) > cap:
            group = rng.sample(group, cap)
        out.extend(group)
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--o-cap", type=int, default=5000, help="cap for V1_O routing slice")
    ap.add_argument("--global-cap", type=int, default=2200, help="cap for other non-kept slices")
    ap.add_argument("--val-global-cap", type=int, default=200, help="per-slice cap for val")
    args = ap.parse_args()

    train = read_rows(SFT / "v1_2_train.jsonl.gz")
    val = read_rows(SFT / "v1_2_val.jsonl.gz")
    lean_train = subsample(train, args.o_cap, args.global_cap)
    lean_val = subsample(val, args.val_global_cap, args.val_global_cap)

    write_rows(SFT / "v1_2_lean_train.jsonl.gz", lean_train)
    write_rows(SFT / "v1_2_lean_val.jsonl.gz", lean_val)

    before = Counter(r.get("slice") for r in train)
    after = Counter(r.get("slice") for r in lean_train)
    print(f"train {len(train)} -> {len(lean_train)} ({100*len(lean_train)/len(train):.0f}%)  "
          f"val {len(val)} -> {len(lean_val)}")
    o = "V1_O_cross_domain_skill_routing"
    print(f"  V1_O {before[o]} -> {after[o]} ({100*after[o]/len(lean_train):.1f}% of lean, was "
          f"{100*before[o]/len(train):.1f}%)")
    print(f"  V1_R kept full: {after.get('V1_R_compute_grounding')}")
    print("wrote", SFT / "v1_2_lean_train.jsonl.gz", "+", SFT / "v1_2_lean_val.jsonl.gz")
    print("NOTE: strict subset of gated rows -> gate-clean; point trainer DATA at v1_2_lean.")


if __name__ == "__main__":
    main()
