# Report assets — the presentation deck (numbered in talk order)

GPU-free PNGs for the talk. Every number traces to a real artifact (see
`src/llm/llm_training/report/chart_data.py` + `docs/report/README.md §3`). Regenerate, don't
hand-edit: **`python -m llm_training.report.deck`**. The beat-by-beat map (which slide for which
spoken line, plus the AI-gen brand-slide prompts) is **`docs/report/slide-visuals.md`**.

Files are **numbered by talk position** so they sort in order and interleave with the presenter's
own slides (01-02 = AI-generated brand cards; 07 = the live chat screenshots — the presenter supplies
both):

| File | Talk beat | What it shows |
|---|---|---|
| `03-how-it-works.png` | call flow | the TWO-VERB loop as a Z-pattern flowchart. User → `<skill>` → body → `<tool>` → data → answer. Real trace from training |
| `03b-reasoning-modes.png` | 4 modes | FAST/THINK/AUTO/PLAN as visual cards — big name, one rule line, real example |
| `04-how-trained.png` | pipeline | train (Kaggle 2× T4) → adapter → serve + knobs (seq 1664, rank-16, loss-weight ×8) and why |
| `05-the-data.png` | the data | reasoning-mode mix + top slices + train/val totals (from `data/sft/v1_2_*.jsonl.gz`) |
| `06-floors-out.png` | "floors out fast" | REAL training-loss curve (`runs/full_train.log`) |
| `08-result-comparison.png` | benchmark | grouped bars: E4B base+harness 49.6% vs v4 adapter+harness 88.7% (+39.1% delta) |
| `09-result-generalizes.png` | benchmark Q2 | 91.7% task completion on unseen domains (n=60) |
| `backup-confusion.png` | (backup, if asked) | per-class routing matrix — the same 88.7%, proof |

`SAMPLE-*.png` are CPU-gate fixtures (`report.gate`), NOT deck slides — gitignored, regenerated on
each gate run. Ignore/delete them.

## Measured Kaggle charts (separate, GPU)

`version_eval` (Kaggle Cell 6 of `kaggle_benchmark.ipynb`) produces the **measured** cross-version
trend + per-slice + miss-analysis when the v2/v3/v4 adapters are evaluated on GPU. The cross-model
line chart (`report.ppt_charts.model_lines`) fills in once the Q5/Q6/E2B runs land. These are not in
the committed deck because they need a real multi-version GPU run; the static placeholders were
removed (a single-point trend / sparse line chart read as broken).

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
