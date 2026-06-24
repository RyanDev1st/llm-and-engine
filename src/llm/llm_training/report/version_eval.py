"""Measured per-version routing trend for the report (Kaggle/T4). Loops the trained adapters
v2 -> v3 -> v4 (from chart_data.VERSIONS), runs the SAME fast adapter+harness routing eval on each,
and emits: the training-timeline chart with REAL accuracy per version, the per-slice bars (v4), the
per-slice MISS analysis (v4 — settles slice G/H), and a trend report .md. One Kaggle pass covers the
whole "we retrained 3 times and got better" story PLUS the miss log.

Each version loads its adapter fresh and is freed before the next (no adapter-swap engineering, no
OOM). A version whose HF repo can't be downloaded is SKIPPED with a warning (a wrong id in
chart_data won't kill the run). Reuses eval_benchmark._bench/_summary so numbers match the main
benchmark exactly.
  python -m llm_training.report.version_eval --per-slice 15 --time-budget 1800
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_dataset.v1.jsonl_io import read_rows  # noqa: E402
from llm_training import bench_misses  # noqa: E402
from llm_training.bench_report import summary as _summary  # noqa: E402
from llm_training.eval_benchmark import _bench  # noqa: E402
from llm_training.eval_confusion import VAL, _sample  # noqa: E402
from llm_training.report import chart_data as D  # noqa: E402
from llm_training.report import charts  # noqa: E402

ASSETS = D.REPO / "docs" / "findings" / "report_assets"


def _gpu_free() -> None:
    """Release CUDA memory. The caller must drop its own reference to the model FIRST — a `del` on a
    function parameter alone leaves the caller's binding alive (that was the v3 OOM: v2 still resident
    when v3 loaded -> 2 E4B models on one T4)."""
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _resolve_adapter(v: dict, workdir: str) -> str | None:
    """Download a version's adapter subfolder from HF; return its local dir, or None if absent."""
    from huggingface_hub import snapshot_download
    local = os.path.join(workdir, v["label"])
    try:
        snapshot_download(repo_id=v["repo"], local_dir=local, allow_patterns=[f"{v['sub']}/*"])
    except Exception as exc:
        print(f"  [skip {v['label']}] cannot download {v['repo']}/{v['sub']}: {exc}", flush=True)
        return None
    adir = os.path.join(local, v["sub"])
    need = ("adapter_config.json", "adapter_model.safetensors")
    if all(os.path.exists(os.path.join(adir, f)) for f in need):
        return adir
    print(f"  [skip {v['label']}] adapter files missing under {adir}", flush=True)
    return None


def run(per_slice: int, time_budget: float, max_new_tokens: int, workdir: str) -> Path:
    rows = _sample(list(read_rows(VAL)), per_slice or None)
    print(f"version trend: {len(rows)} val rows x {len(D.VERSIONS)} versions (fast, adapter+harness)",
          flush=True)
    from backend.model_hf import HFModel
    versions = [dict(v) for v in D.VERSIONS]   # local copy we fill with measured numbers
    v4_res = None
    model = None
    for v in versions:
        adir = _resolve_adapter(v, workdir)
        if not adir:
            v["verb"] = None
            continue
        if model is not None:           # free the PREVIOUS version BEFORE loading the next, in THIS
            del model                   # scope (the only binding) -> never 2 E4B models on one T4
            model = None
            _gpu_free()
        print(f"\n=== {v['label']} ({v['repo']}/{v['sub']}) ===", flush=True)
        model = HFModel(adapter=adir, temperature=0.0)
        res = _bench(model, rows, max_new_tokens=max_new_tokens, time_budget_s=time_budget or None,
                     label=v["label"])
        summ = _summary(res)
        v["verb"], v["exact"] = summ["acc"], summ["name"]
        print(f"  {v['label']}: verb {summ['acc']:.1%} | exact-name {summ['name']:.1%} | n={res['n']}",
              flush=True)
        if v["label"] == "v4":
            v4_res = res
    if model is not None:
        del model
        _gpu_free()

    ASSETS.mkdir(parents=True, exist_ok=True)
    charts.training_timeline(versions, ASSETS / "chart-training-timeline.png")
    if v4_res:
        sl_acc = {s: v4_res["sc"][s] / v4_res["st"][s] for s in v4_res["st"]}
        charts.per_slice_bars(sl_acc, ASSETS / "chart-per-slice-v4.png")
        bench_misses.write_jsonl(v4_res["misses"], ASSETS / "v4-val-misses.jsonl")

    from datetime import date
    L = ["Parent: docs/findings/2026-06-21-routing-benchmark-interpretation.md", "",
         "# Measured per-version routing trend (v2 -> v3 -> v4)", "",
         "Same fast adapter+harness routing eval on each trained adapter. Each retrain targeted a "
         "specific diagnosed failure (the 'why' column, from git).", "",
         "| version | date | verb acc | exact-name | why retrained | fix |",
         "|---|---|---|---|---|---|"]
    for v in versions:
        va = f"{v['verb']:.1%}" if v.get("verb") is not None else "n/a (adapter missing)"
        ex = f"{v.get('exact'):.1%}" if v.get("exact") is not None else "-"
        L.append(f"| {v['label']} | {v['date']} | {va} | {ex} | {v['why']} | {v['fix']} |")
    if v4_res:
        L += ["", "## v4 per-slice MISS analysis (what the misses actually emitted)",
              bench_misses.breakdown_md(v4_res["misses"])]
    out = D.REPO / "docs" / "findings" / f"{date.today():%Y-%m-%d}-version-trend.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[2:]), flush=True)
    print(f"\nwrote {out}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-slice", type=int, default=15, help="val rows/slice/version (small = fast)")
    ap.add_argument("--time-budget", type=float, default=0, help="seconds per version (0 = none)")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--workdir", default="/kaggle/working/adapters")
    args = ap.parse_args()
    run(args.per_slice, args.time_budget, args.max_new_tokens, args.workdir)


if __name__ == "__main__":
    main()
    from llm_training.clean_exit import flush_and_exit
    flush_and_exit()   # benign torch/CUDA exit-time SIGABRT must not fail the notebook run
