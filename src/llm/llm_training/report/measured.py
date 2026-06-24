"""Shared measured-numbers sidecar so every eval feeds the ONE cross-model line chart unattended.

Each tagged run merges its headline rates into `report_assets/measured-<KEY>.json` (KEY = a
chart_data.MODELS key, e.g. e4b-nf4 / e4b-q5 / e2b-adapter). Read-merge-write so the verb accuracy
(eval_confusion), task-completion + grounded (eval_completion), and tok/s (chat_showcase) — three
DIFFERENT evals, possibly hours apart — coexist in one file per model with no clobber. The
model-lines cell globs them, merges onto the seeds, and renders. Pure JSON, no GPU.
"""
from __future__ import annotations

import json
from pathlib import Path


def update(assets_dir: str | Path, key: str, **vals) -> Path:
    """Merge non-None `vals` into measured-<key>.json under assets_dir (read-merge-write)."""
    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    p = assets_dir / f"measured-{key}.json"
    cur: dict = {}
    if p.exists():
        try:
            cur = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
    for k, v in vals.items():
        if v is not None:
            cur[k] = v
    p.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    return p


def collect(assets_dir: str | Path) -> dict:
    """{key: {metric: value}} from every measured-*.json under assets_dir (for merge_measured)."""
    out: dict = {}
    for p in Path(assets_dir).glob("measured-*.json"):
        key = p.stem[len("measured-"):]
        try:
            out[key] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return out
