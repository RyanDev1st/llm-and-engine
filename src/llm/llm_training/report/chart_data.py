"""Data backing the report charts — every number traces to a real artifact, so the charts are
reproducible and honest.

SOURCES:
- COND_VAL / COND_STRESS: the 2026-06-21 Kaggle 3-condition routing benchmark
  (docs/findings/2026-06-21-routing-benchmark*.md). val n=692, stress n=20.
- VERSIONS: git history (the 'why retrained' per version) + dates from the retrain commits.
  The `repo`/`sub` are the Hugging Face adapter locations — EDIT to your exact repo ids/tags if
  they differ; version_eval validates each and skips (with a warning) any it can't download.
  `verb` is filled by version_eval at run time; v4's is the measured benchmark value as a seed.
"""
from __future__ import annotations

import collections
import gzip
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

# --- routing ablation (val), from the committed benchmark report. The e2b condition (prior
# production model) is filled by the next measured benchmark run; until then the chart shows the two
# E4B conditions (SFT-weights contribution). base-no-harness was dropped from the design. ---
COND_VAL = {
    "e4b-v4 adapter+harness": {"verb": 0.964, "macro": 0.783, "exact": 0.739},
    "e4b base+harness": {"verb": 0.829, "macro": 0.462, "exact": 0.176},
    # "e2b adapter+harness": filled after the measured rerun (eval_benchmark --e2b-adapter ...)
}

# --- E4B training history: each retrain fixed a specific diagnosed failure (git log) ---
VERSIONS = [
    {"label": "v2", "date": "2026-06-17", "repo": "RyanDev1st/gemma4-chesscoach-ckpt-v2",
     "sub": "best", "why": "attn-only adapter couldn't emit the harness format",
     "fix": "all-linear LoRA targets + honor lora_dropout", "verb": None},
    {"label": "v3", "date": "2026-06-18", "repo": "RyanDev1st/gemma4-chesscoach-ckpt-v3",
     "sub": "best", "why": "format correct, but skill/tool NAMES corrupted on copy",
     "fix": "up-weight control tags in loss + decode penalties OFF", "verb": None},
    {"label": "v4", "date": "2026-06-19", "repo": "RyanDev1st/gemma4-chesscoach-ckpt-v4",
     "sub": "best", "why": "names still imperfect; extend the loss weight to names + train longer",
     "fix": "FORMAT_WEIGHT on skill/tool names + 1000 steps + hardened base harness", "verb": 0.964},
]


# --- ALL models for the cross-model performance line chart (report). Ordered earliest -> latest along
# the x-axis. SEEDS are the MEASURED 2026-06-24 numbers (the FAIR native run + the OOD STRESS completion)
# so the chart matches the report headline, not the stale 2026-06-21 forced-fast numbers:
#   verb = eval_benchmark val-NATIVE (base 49.6%, adapter 88.7%); completed/grounded = eval_completion
#   --stress (adapter 91.7%/95%). tok/s + the GGUF quants + E2B are filled at run time via merge_measured
#   (chat_showcase --tag for tok/s; eval_completion --gguf for the quants). ---
MODELS = [
    {"key": "e2b-adapter", "label": "E2B adapter\n(prior prod)"},
    {"key": "e4b-base",    "label": "E4B base\n+harness", "verb": 0.496},
    {"key": "e4b-nf4",     "label": "E4B v4 nf4\n(current)", "verb": 0.887,
     "completed": 0.917, "grounded": 0.95},
    {"key": "e4b-q5",      "label": "E4B Q5_K_M\nGGUF"},
    {"key": "e4b-q6",      "label": "E4B Q6_K\nGGUF"},
]


def merge_measured(models: list, measured: dict) -> list:
    """Overlay a {key: {metric: value}} dict of measured numbers onto a COPY of `models` (pure;
    unit-tested). Unknown keys are ignored; a None value never clobbers an existing seed."""
    out = [dict(m) for m in models]
    by = {m["key"]: m for m in out}
    for key, vals in (measured or {}).items():
        m = by.get(key)
        if not m:
            continue
        for k, v in (vals or {}).items():
            if v is not None:
                m[k] = v
    return out


def model_table_md(models: list) -> str:
    """A markdown table of the cross-model numbers (the report's text mirror of model_lines)."""
    L = ["| model | routing verb | completion | grounded | tok/s |", "|---|---|---|---|---|"]
    for m in models:
        def c(k, pct=True):
            v = m.get(k)
            return "—" if v is None else (f"{v:.0%}" if pct else f"{v:.0f}")
        L.append(f"| {m['label'].replace(chr(10), ' ')} | {c('verb')} | {c('completed')} | "
                 f"{c('grounded')} | {c('tok_s', pct=False)} |")
    return "\n".join(L)


def load_train_losses(log_path: Path | None = None) -> list[float]:
    """The REAL per-update training loss series from runs/full_train.log (lines like
    'upd 12/164 ep 1 loss=2.47 lr=...'). Used by the 'floors out fast' slide so the curve is
    measured, not drawn. Returns [] if the log is absent (the slide then skips, never fabricates)."""
    import re
    log_path = log_path or REPO / "runs" / "full_train.log"
    if not log_path.exists():
        return []
    out = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"loss=([0-9.]+)", line)
        if m:
            out.append(float(m.group(1)))
    return out


def corpus_stats(train_gz: Path | None = None, val_gz: Path | None = None) -> dict:
    """Measured corpus composition (reasoning-mode mix, per-slice sizes, train/val totals). The
    general/chess DESIGN target is ~75/25; we do not assert a measured domain split because chess
    also appears inside several V1_ slices (board grounding, special rules, eval language)."""
    train_gz = train_gz or REPO / "data" / "sft" / "v1_2_train.jsonl.gz"
    val_gz = val_gz or REPO / "data" / "sft" / "v1_2_val.jsonl.gz"

    def load(p):
        return [json.loads(l) for l in gzip.open(p, "rt", encoding="utf-8") if l.strip()]

    tr, va = load(train_gz), load(val_gz)
    modes = collections.Counter((r.get("reasoning_mode") or "fast") for r in tr)
    slices = collections.Counter(r["slice"] for r in tr)
    return {"n_train": len(tr), "n_val": len(va), "modes": dict(modes),
            "slices": dict(slices), "n_slices": len(slices)}
