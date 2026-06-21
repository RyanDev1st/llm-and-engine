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


def _bench(model, rows, *, max_new_tokens, time_budget_s, label, with_harness=True):
    """One condition (fast mode). with_harness=False feeds only the user turn. Tallies the matrix,
    exact-name, 'soup' (any non-contract tag), per-slice, misses, wall-time. Interrupt-safe."""
    cm = {g: {p: 0 for p in CLASSES} for g in CLASSES}
    nh = nt = soup = done = 0
    sc, st = defaultdict(int), defaultdict(int)
    misses: list = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        user = next(m for m in r["messages"] if m.get("role") == "user")
        msgs = ([{"role": "system", "content": _system(r, True)}, user] if with_harness else [user])
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
    ap.add_argument("--e2b-adapter", default=None, help="local dir of the PRIOR E2B production LoRA "
                    "adapter; adds a 3rd condition (e2b adapter+harness). Needs --e2b-base.")
    ap.add_argument("--e2b-base", default=None, help="local dir of the E2B base the e2b adapter was "
                    "trained on (e.g. a downloaded unsloth/gemma-4-E2B-it)")
    args = ap.parse_args()

    from datetime import date
    from llm_dataset.v1.jsonl_io import read_rows
    from backend.inference import AdapterView
    if args.suite == "stress":
        from llm_training.bench_suites import stress_rows
        rows = stress_rows()                              # full held-out stress set (small)
    else:
        rows = _sample(list(read_rows(VAL)), args.per_slice or None)
    model = _load_model(args)
    base = AdapterView(model, False)        # same E4B weights, LoRA OFF (isolates SFT weights)
    tb, mnt = args.time_budget or None, args.max_new_tokens
    n3 = " + e2b adapter+harness" if args.e2b_adapter else ""
    print(f"benchmark [{args.suite}]: {len(rows)} rows · e4b-v4 adapter+harness, e4b base+harness{n3}",
          flush=True)
    conds = [
        ("e4b-v4 adapter+harness", _bench(model, rows, max_new_tokens=mnt, time_budget_s=tb, label="e4b-v4 adapter+harness")),
        ("e4b base+harness", _bench(base, rows, max_new_tokens=mnt, time_budget_s=tb, label="e4b base+harness")),
    ]
    if args.e2b_adapter:                    # prior production model: own E2B base -> SEPARATE load
        del base, model                     # free the E4B model first (sequential -> no T4 OOM)
        import gc; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
        from backend.model_hf import HFModel
        e2b = HFModel(base=args.e2b_base, adapter=args.e2b_adapter, temperature=0.0)
        conds.append(("e2b adapter+harness",
                      _bench(e2b, rows, max_new_tokens=mnt, time_budget_s=tb, label="e2b adapter+harness")))
    bench_report.write_report(conds, f"{date.today():%Y-%m-%d}", args.suite)


if __name__ == "__main__":
    main()
