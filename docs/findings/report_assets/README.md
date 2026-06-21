# Report assets — charts for the project writeup/presentation

Generated PNGs for the report. Every number traces to a real artifact (see
`src/llm/llm_training/report/chart_data.py` for sources). Regenerate, don't hand-edit.

| File | What it shows | How to regenerate |
|---|---|---|
| `chart-layer-contribution.png` | adapter+harness vs base+harness vs base-no-harness × verb-acc / macro-prec / exact-name — what the harness contract buys vs what the SFT weights add | `python -m llm_training.report.charts` (GPU-free; from the 2026-06-21 3-condition benchmark) |
| `chart-corpus-composition.png` | reasoning-mode mix (fast/think/auto/plan) + top slices by size + train/val totals | `python -m llm_training.report.charts` (GPU-free; from `data/sft/v1_2_*.jsonl.gz`) |
| `chart-training-timeline.png` | v2→v3→v4 diagnose→fix→measure; verb accuracy per version once measured | GPU-free preview via `charts`; the **measured** version runs on Kaggle via `version_eval` |
| `chart-per-slice-v4.png` | per-slice exact routing accuracy (v4) — routing breadth | Kaggle: `python -m llm_training.report.version_eval` (needs GPU) |
| `v4-val-misses.jsonl` | every missed val row + what the model emitted (settles slice G/H) | Kaggle: `version_eval` |

**Measured trend + per-slice + miss-analysis** come from one Kaggle pass (Cell 6 of
`kaggle_benchmark.ipynb`). If the v2/v3 adapter repo ids differ from
`chart_data.VERSIONS`, edit that file's `repo`/`sub` and re-run — a missing repo is skipped,
not fatal.
