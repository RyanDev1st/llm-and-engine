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

## The prior E2B production model (3rd benchmark condition)

The benchmark's optional `e2b adapter+harness` condition is the **prior production model**: an
attn-only r=8 LoRA on the **E2B** base, trained in the E2B production era. The adapter lives
**locally** at `runs/gemma4_e2b_unified/best/` (5.4 MB) and is **gitignored** (`*.safetensors`),
so it is NOT in the Kaggle clone. Get it onto Kaggle one of two ways, then set the matching var
in `kaggle_benchmark.ipynb` Cell 1:

- **(a) HF (default)** — already pushed to the **private** repo
  `RyanDev1st/gemma4-chesscoach-e2b` (subfolder `best`, the two adapter files; the adapter's local
  README was skipped — its YAML `base_model` is a local path HF rejects). Cell 1 points
  `E2B_ADAPTER_REPO` here; Cell 3 downloads it with the Kaggle `HF_TOKEN` secret (same account → the
  private repo is visible). Re-push with:

  ```bash
  python -c "from huggingface_hub import HfApi; HfApi(token='<HF_TOKEN>').upload_folder(\
    folder_path='runs/gemma4_e2b_unified/best', repo_id='RyanDev1st/gemma4-chesscoach-e2b', \
    path_in_repo='best', repo_type='model', ignore_patterns=['README.md'])"
  ```
- **(b) Kaggle Dataset** — alternatively upload `runs/gemma4_e2b_unified/best/` via *Add Input* and
  set `E2B_ADAPTER_DIR='/kaggle/input/<name>/best'` (overrides the HF download).

Leave both blank to run the 2-condition E4B benchmark only.

**Disk note (Kaggle ~20GB):** the E4B base (~9GB) and E2B base (~10GB) can't both sit on disk, so
the E2B condition runs in the **last cell** via `eval_benchmark --e2b-only --free-base <e4b base>
--e2b-base-repo unsloth/gemma-4-E2B-it`: it deletes the E4B base, then downloads the E2B base, then
evaluates the e2b adapter on the same val rows → a standalone `…-routing-benchmark-e2b.md` (same
metrics as the E4B report, directly comparable). This is why it must run AFTER the E4B benchmark +
transcript + version-trend cells, which still need the E4B base.
