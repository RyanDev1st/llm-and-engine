"""Serious routing BENCHMARK for the report — a clean ablation over the validation set:
  1) e4b-v4 adapter + harness — the product (trained v4 LoRA on the full harness contract)
  2) e4b base + harness       — SAME E4B model, LoRA DISABLED (isolates what the SFT WEIGHTS bought)
  3) e2b adapter + harness     — the PRIOR production model (E2B base + its LoRA), optional
For each: confusion matrix + precision/recall/F1 + exact-name + format validity + throughput; then
deltas vs the product (1-vs-2 = SFT weights on the same base; 1-vs-3 = E4B product vs prior E2B
production). Cond 2 reuses the loaded E4B model with the LoRA off (AdapterView — no 2nd load). Cond
3 has a DIFFERENT base, so it loads separately AFTER freeing the E4B model (sequential — no OOM).
No fabricated external baselines; reproducible from our own artifacts.

Kaggle (T4, up to 12h):
  python -m llm_training.eval_benchmark --adapter <e4b_best> --per-slice 25 --time-budget 9000 \
      [--e2b-adapter <e2b_best_dir> --e2b-base <e2b_base_dir>]
--per-slice 0 = the FULL val set. Saves docs/findings/<date>-routing-benchmark.md + a PNG/condition.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_training import bench_misses, bench_report  # noqa: E402
from llm_training.eval_confusion import (  # noqa: E402  — reuse the routing primitives
    CLASSES, VAL, _load_model, _sample, _system, first_action, gold_action)

REPO = Path(__file__).resolve().parents[3]
_TAG = re.compile(r"</?([a-zA-Z_][\w]*)")
_CONTRACT = {"think", "/think", "goal", "/goal", "plan", "/plan", "skill", "/skill", "tool", "/tool"}


def _bench(model, rows, *, max_new_tokens, time_budget_s, label, with_harness=True, native_mode=False):
    """One condition. with_harness=False feeds only the user turn. native_mode=True scores each row
    in its TRAINED reasoning mode (goal/think before the action) — the FAIR test for mode-dependent
    slices; default fast mode forces the action first (cheaper, but handicaps auto/think-trained
    rows). Tallies the matrix, exact-name, 'soup', per-slice, misses, wall-time. Interrupt-safe."""
    cm = {g: {p: 0 for p in CLASSES} for g in CLASSES}
    nh = nt = soup = done = 0
    sc, st = defaultdict(int), defaultdict(int)
    misses: list = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        user = next(m for m in r["messages"] if m.get("role") == "user")
        msgs = ([{"role": "system", "content": _system(r, not native_mode)}, user] if with_harness else [user])
        out = model.generate(msgs, max_new_tokens=max_new_tokens, stop=["</skill>", "</tool>"])
        if [t for t in _TAG.findall(out) if t not in _CONTRACT]:
            soup += 1
        pv, pn = first_action(out)
        gv, gn = gold_action(r["messages"])
        cm[gv][pv] += 1
        st[r["slice"]] += 1
        ok = pv == gv and pn == gn
        if ok:
            sc[r["slice"]] += 1
        else:                                 # keep WHAT it emitted so the failure mode is knowable
            bench_misses.record(misses, slice_=r["slice"], user=user.get("content", ""),
                                gold=(gv, gn), pred=(pv, pn), out=out)
        if gv != "none":
            nt += 1
            nh += int(ok)
        done = i
        if i % 50 == 0:
            el = time.time() - t0
            print(f"  [{label}] {i}/{len(rows)} ({el:.0f}s, {el/i:.1f}s/row)", flush=True)
        if time_budget_s and time.time() - t0 > time_budget_s:
            print(f"  [{label}] budget reached at {done}/{len(rows)}", flush=True)
            break
    return {"cm": cm, "nh": nh, "nt": nt, "soup": soup, "n": done, "sec": time.time() - t0,
            "sc": sc, "st": st, "misses": misses}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="adapter dir (loads HFModel)")
    ap.add_argument("--server", default="http://127.0.0.1:7861", help="model service URL")
    ap.add_argument("--per-slice", type=int, default=25, help="rows/slice (0 = full val); val suite only")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--time-budget", type=float, default=0, help="seconds PER condition (0 = none)")
    ap.add_argument("--suite", choices=["val", "stress"], default="val",
                    help="val = in-distribution held-out; stress = held-out wild/out-of-domain")
    ap.add_argument("--native-mode", action="store_true", help="score each row in its TRAINED "
                    "reasoning mode (fair test for auto/think-trained slices); needs a larger "
                    "--max-new-tokens so the action lands after the goal/think preamble (~160).")
    ap.add_argument("--slices", default=None, help="comma-separated slice filter (e.g. G,H,V1_M) — "
                    "scope a focused probe to specific slices")
    ap.add_argument("--e2b-only", action="store_true", help="evaluate ONLY the prior E2B production "
                    "model (its own base). Disk-safe on Kaggle: frees --free-base + downloads the E2B "
                    "base first, so both bases never sit on the ~20GB disk at once. Run this LAST.")
    ap.add_argument("--e2b-adapter", default=None, help="local dir of the prior E2B production LoRA")
    ap.add_argument("--e2b-base", default=None, help="local dir for the E2B base (download target)")
    ap.add_argument("--e2b-base-repo", default=None, help="HF repo to fetch the E2B base from if "
                    "--e2b-base is missing (e.g. unsloth/gemma-4-E2B-it)")
    ap.add_argument("--free-base", default=None, help="a base dir to delete from disk before the E2B "
                    "base download (frees the E4B base — only safe AFTER all E4B work is done)")
    args = ap.parse_args()

    from datetime import date
    from llm_dataset.v1.jsonl_io import read_rows
    from backend.inference import AdapterView
    all_rows = (list(read_rows(VAL)) if args.suite == "val"
                else __import__("llm_training.bench_suites", fromlist=["stress_rows"]).stress_rows())
    if args.slices:                          # focused probe: keep only the named slices
        keep = set(args.slices.split(","))
        all_rows = [r for r in all_rows if r["slice"] in keep]
    rows = _sample(all_rows, args.per_slice or None) if args.suite == "val" else all_rows
    tb, mnt, nat = args.time_budget or None, args.max_new_tokens, args.native_mode
    if args.e2b_only:                       # prior E2B production model — its OWN base, run LAST
        _e2b_only(args, rows, mnt, tb, date)
        return
    model = _load_model(args)
    base = AdapterView(model, False)        # same E4B weights, LoRA OFF (isolates SFT weights)
    tag = "native-mode" if nat else "fast-mode"
    print(f"benchmark [{args.suite}/{tag}]: {len(rows)} rows · adapter vs base (both +harness)",
          flush=True)
    conds = [
        ("e4b-v4 adapter+harness", _bench(model, rows, max_new_tokens=mnt, time_budget_s=tb, label="e4b-v4 adapter+harness", native_mode=nat)),
        ("e4b base+harness", _bench(base, rows, max_new_tokens=mnt, time_budget_s=tb, label="e4b base+harness", native_mode=nat)),
    ]
    suite_tag = args.suite if not nat else f"{args.suite}-native"
    bench_report.write_report(conds, f"{date.today():%Y-%m-%d}", suite_tag)


def _e2b_only(args, rows, mnt, tb, date) -> None:
    """Disk-safe standalone eval of the prior E2B production model: delete the E4B base from disk
    (--free-base) and download the E2B base (--e2b-base-repo) BEFORE loading, so Kaggle's ~20GB disk
    never holds both bases. Writes a 1-condition routing report tagged 'e2b'."""
    import os
    import shutil
    if args.free_base and os.path.isdir(args.free_base):
        shutil.rmtree(args.free_base, ignore_errors=True)
        print(f"freed E4B base disk: {args.free_base}", flush=True)
    if args.e2b_base_repo and not os.path.isdir(args.e2b_base):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=args.e2b_base_repo, local_dir=args.e2b_base,
                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"])
    from backend.model_hf import HFModel
    e2b = HFModel(base=args.e2b_base, adapter=args.e2b_adapter, temperature=0.0)
    res = _bench(e2b, rows, max_new_tokens=mnt, time_budget_s=tb, label="e2b adapter+harness")
    bench_report.write_report([("e2b adapter+harness", res)], f"{date.today():%Y-%m-%d}", "e2b")


if __name__ == "__main__":
    main()
