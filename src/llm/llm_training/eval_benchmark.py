"""Serious routing BENCHMARK for the report: the trained adapter (our SFT) vs the untrained
base Gemma on the SAME harness, over the validation set. For each condition we report a
confusion matrix + precision/recall/F1 + exact-name accuracy + format-validity + per-slice
routing + throughput, plus the adapter-over-base LIFT. The base comparison is the HONEST
"against others" — it isolates what the SFT bought (no fabricated external-model numbers); the
base run reuses the SAME loaded model with the LoRA disabled (no 2nd load, no OOM).

Built for Kaggle (2x T4, up to 12h) so a large stratified sample fits:
  python -m llm_training.eval_benchmark --adapter <best> --per-slice 30 --time-budget 18000
--per-slice 0 = the FULL val set. Saves docs/findings/<date>-routing-benchmark.md + matrix PNGs.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_training.eval_confusion import (  # noqa: E402  — reuse the routing primitives
    CLASSES, VAL, _load_model, _metrics, _png, _sample, _system, first_action, gold_action)

REPO = Path(__file__).resolve().parents[3]
_TAG = re.compile(r"</?([a-zA-Z_][\w]*)")
_CONTRACT = {"think", "/think", "goal", "/goal", "plan", "/plan", "skill", "/skill", "tool", "/tool"}


def _bench(model, rows, *, max_new_tokens, time_budget_s, label):
    """One condition: generate the first action per row (fast mode), tally the matrix, exact-name,
    'soup' (any non-contract tag), per-slice, and wall-time. Interrupt-safe via time_budget_s."""
    cm = {g: {p: 0 for p in CLASSES} for g in CLASSES}
    nh = nt = soup = done = 0
    sc, st = defaultdict(int), defaultdict(int)
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        user = next(m for m in r["messages"] if m.get("role") == "user")
        out = model.generate([{"role": "system", "content": _system(r, True)}, user],
                             max_new_tokens=max_new_tokens, stop=["</skill>", "</tool>"])
        if [t for t in _TAG.findall(out) if t not in _CONTRACT]:
            soup += 1
        pv, pn = first_action(out)
        gv, gn = gold_action(r["messages"])
        cm[gv][pv] += 1
        st[r["slice"]] += 1
        if pv == gv and pn == gn:
            sc[r["slice"]] += 1
        if gv != "none":
            nt += 1
            nh += int(pv == gv and pn == gn)
        done = i
        if i % 50 == 0:
            el = time.time() - t0
            print(f"  [{label}] {i}/{len(rows)} ({el:.0f}s, {el/i:.1f}s/row)", flush=True)
        if time_budget_s and time.time() - t0 > time_budget_s:
            print(f"  [{label}] budget reached at {done}/{len(rows)}", flush=True)
            break
    return {"cm": cm, "nh": nh, "nt": nt, "soup": soup, "n": done,
            "sec": time.time() - t0, "sc": sc, "st": st}


def _summary(b: dict) -> dict:
    cm = b["cm"]
    tot = sum(cm[g][p] for g in CLASSES for p in CLASSES) or 1
    met = _metrics(cm)
    wsup = sum(met[c]["support"] for c in CLASSES) or 1
    return {"met": met,
            "acc": sum(cm[c][c] for c in CLASSES) / tot,
            "macro": sum(met[c]["precision"] for c in CLASSES) / len(CLASSES),
            "wp": sum(met[c]["precision"] * met[c]["support"] for c in CLASSES) / wsup,
            "name": b["nh"] / b["nt"] if b["nt"] else 0.0,
            "fmt": 1 - b["soup"] / max(b["n"], 1),
            "spr": b["sec"] / max(b["n"], 1)}


def _matrix_md(cm: dict) -> str:
    rows = ["| gold \\ pred | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for g in CLASSES:
        rows.append(f"| {g} | " + " | ".join(str(cm[g][p]) for p in CLASSES) + " |")
    return "\n".join(rows)


def _prf_md(met: dict) -> str:
    rows = ["| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    for c in CLASSES:
        m = met[c]
        rows.append(f"| {c} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['support']} |")
    return "\n".join(rows)


def _hl(metric: str, a, b, pct=True) -> str:
    fmt = (lambda x: f"{x:.1%}") if pct else (lambda x: f"{x:.2f}")
    if b is None:
        return f"| {metric} | {fmt(a)} | — | — |"
    lift = f"{a - b:+.1%}" if pct else f"{a - b:+.2f}"
    return f"| {metric} | {fmt(a)} | {fmt(b)} | {lift} |"


def _write_report(a: dict, b: dict | None, date_str: str) -> Path:
    sa = _summary(a)
    sb = _summary(b) if b else None
    L = ["Parent: docs/reference/sft-corpus-generation.md", "",
         "# Routing benchmark — Gemma 4 E4B chess-coach (v4 adapter)", "",
         f"n = {a['n']} stratified val rows · scored in FAST mode (routing is mode-independent) · "
         "base = the same model with the LoRA disabled (isolates the SFT lift).", "",
         "## Headline (trained adapter vs untrained base)",
         "| metric | adapter (SFT) | base Gemma | lift |", "|---|---|---|---|",
         _hl("verb accuracy", sa["acc"], sb["acc"] if sb else None),
         _hl("macro precision", sa["macro"], sb["macro"] if sb else None),
         _hl("weighted precision", sa["wp"], sb["wp"] if sb else None),
         _hl("exact-name accuracy", sa["name"], sb["name"] if sb else None),
         _hl("format validity (no foreign tags)", sa["fmt"], sb["fmt"] if sb else None),
         _hl("throughput (s/row)", sa["spr"], sb["spr"] if sb else None, pct=False),
         "", "## Confusion matrix — adapter (rows = gold, cols = pred)", _matrix_md(a["cm"]),
         "", "### Per-class precision / recall / F1 — adapter", _prf_md(sa["met"])]
    if b:
        L += ["", "## Confusion matrix — base (adapter disabled)", _matrix_md(b["cm"]),
              "", "### Per-class precision / recall / F1 — base", _prf_md(sb["met"])]
    L += ["", "## Per-slice routing accuracy (adapter)"]
    for sl in sorted(a["st"]):
        L.append(f"- {sl}: {a['sc'][sl]}/{a['st'][sl]} = {a['sc'][sl] / a['st'][sl]:.0%}")
    out = REPO / "docs" / "findings" / f"{date_str}-routing-benchmark.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    _png(a["cm"], out.with_name(out.stem + "-adapter.png"))
    if b:
        _png(b["cm"], out.with_name(out.stem + "-base.png"))
    print("\n".join(L[2:]), flush=True)
    print(f"\nwrote {out}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="adapter dir (loads HFModel)")
    ap.add_argument("--server", default="http://127.0.0.1:7861", help="model service URL")
    ap.add_argument("--per-slice", type=int, default=30, help="rows/slice (0 = full val)")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--time-budget", type=float, default=0, help="seconds PER condition (0 = none)")
    ap.add_argument("--no-base", action="store_true", help="skip the base comparison (adapter only)")
    args = ap.parse_args()

    from datetime import date
    from llm_dataset.v1.jsonl_io import read_rows
    rows = _sample(list(read_rows(VAL)), args.per_slice or None)
    model = _load_model(args)
    tb = args.time_budget or None
    print(f"benchmark: {len(rows)} rows | adapter vs {'(skipped)' if args.no_base else 'base'} ...", flush=True)
    a = _bench(model, rows, max_new_tokens=args.max_new_tokens, time_budget_s=tb, label="adapter")
    b = None
    if not args.no_base:
        from backend.inference import AdapterView
        b = _bench(AdapterView(model, False), rows,
                   max_new_tokens=args.max_new_tokens, time_budget_s=tb, label="base")
    _write_report(a, b, f"{date.today():%Y-%m-%d}")


if __name__ == "__main__":
    main()
