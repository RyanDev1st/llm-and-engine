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

# --- 3-condition ablation (val + stress), from the committed benchmark report ---
COND_VAL = {
    "adapter+harness": {"verb": 0.964, "macro": 0.783, "exact": 0.739},
    "base+harness": {"verb": 0.829, "macro": 0.462, "exact": 0.176},
    "base no-harness": {"verb": 0.030, "macro": 0.010, "exact": 0.000},
}
COND_STRESS = {  # verb accuracy only (n=20; base+harness 0.95 is a 1-row artifact, noted in report)
    "adapter+harness": 0.90, "base+harness": 0.95, "base no-harness": 0.25,
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
