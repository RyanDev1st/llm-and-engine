"""Serious routing BENCHMARK for the report — a clean ablation over the validation set:
  A) adapter + harness  — the product (trained v4 LoRA on the full harness contract)
  B) base + harness      — SAME harness, LoRA DISABLED (isolates what the SFT WEIGHTS bought)
  C) base, NO harness    — raw base Gemma, no contract at all (isolates what the HARNESS bought)
For each we report a confusion matrix + precision/recall/F1 + exact-name + format validity +
throughput; then the layer deltas (A-B = SFT weights, B-C = harness contract, A-C = full system).
B and C reuse the SAME loaded model with the LoRA off (AdapterView) — no 2nd load, no OOM, no
fabricated external baselines. Honest, reproducible from our own artifacts.

Built for Kaggle (T4, up to 12h):
  python -m llm_training.eval_benchmark --adapter <best> --per-slice 25 --time-budget 9000
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
from llm_training import bench_misses  # noqa: E402
from llm_training.eval_confusion import (  # noqa: E402  — reuse the routing primitives
    CLASSES, VAL, _load_model, _metrics, _png, _sample, _system, first_action, gold_action)

REPO = Path(__file__).resolve().parents[3]
_TAG = re.compile(r"</?([a-zA-Z_][\w]*)")
_CONTRACT = {"think", "/think", "goal", "/goal", "plan", "/plan", "skill", "/skill", "tool", "/tool"}


def _bench(model, rows, *, max_new_tokens, time_budget_s, label, with_harness=True):
    """One condition. with_harness=True feeds the SAME system contract (fast mode); False feeds
    ONLY the user turn (raw base, no contract). Tallies the matrix, exact-name, 'soup' (any
    non-contract tag), per-slice, wall-time. Interrupt-safe via time_budget_s."""
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


def _summary(b: dict) -> dict:
    cm = b["cm"]
    tot = sum(cm[g][p] for g in CLASSES for p in CLASSES) or 1
    met = _metrics(cm)
    wsup = sum(met[c]["support"] for c in CLASSES) or 1
    return {"met": met, "acc": sum(cm[c][c] for c in CLASSES) / tot,
            "macro": sum(met[c]["precision"] for c in CLASSES) / len(CLASSES),
            "wp": sum(met[c]["precision"] * met[c]["support"] for c in CLASSES) / wsup,
            "name": b["nh"] / b["nt"] if b["nt"] else 0.0,
            "fmt": 1 - b["soup"] / max(b["n"], 1), "spr": b["sec"] / max(b["n"], 1)}


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


def _headline(summ: list) -> str:
    labels = [lab for lab, _ in summ]
    L = ["| metric | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]

    def row(name, key, pct=True):
        f = (lambda x: f"{x:.1%}") if pct else (lambda x: f"{x:.2f}")
        return f"| {name} | " + " | ".join(f(s[key]) for _, s in summ) + " |"
    L += [row("verb accuracy", "acc"), row("macro precision", "macro"),
          row("weighted precision", "wp"), row("exact-name accuracy", "name"),
          row("format validity (no foreign tags)", "fmt"), row("throughput (s/row)", "spr", False)]
    return "\n".join(L)


def _delta(a: dict, b: dict) -> str:
    return (f"verb acc {a['acc']-b['acc']:+.1%}, format {a['fmt']-b['fmt']:+.1%}, "
            f"macro-prec {a['macro']-b['macro']:+.1%}, exact-name {a['name']-b['name']:+.1%}")


def _write_report(conds: list, date_str: str, suite: str = "val") -> Path:
    """conds = [(label, bench_result), ...] in order: adapter+harness, base+harness, base no-harness.
    `suite` ('val' in-distribution held-out, or 'stress' held-out wild/out-of-domain) names the
    row source — it tags the report title + output filename so the two tiers don't overwrite."""
    summ = [(lab, _summary(b)) for lab, b in conds]
    s = {lab: sm for lab, sm in summ}
    a, bh, nh = "adapter+harness", "base+harness", "base no-harness"
    src = ("held-out STRESS rows (messy/slang/typo phrasing + UNSEEN out-of-domain catalogs + "
           "decline cases — hand-written, in NO training row)" if suite == "stress"
           else "stratified val rows (held out from training; in-distribution phrasing)")
    L = ["Parent: docs/reference/sft-corpus-generation.md", "",
         f"# Routing benchmark ({suite}) — Gemma 4 E4B chess-coach (v4 adapter)", "",
         f"n = {conds[0][1]['n']} {src} · fast mode (routing is mode-independent) · "
         "B/C reuse the same model with the LoRA disabled (B keeps the harness contract; C drops "
         "it entirely — raw base Gemma). This isolates the two layers separately.", "",
         "## Headline — three conditions", _headline(summ), "",
         "## What each layer contributes",
         f"- **SFT weights** (adapter+harness vs base+harness): {_delta(s[a], s[bh])}",
         f"- **Harness contract** (base+harness vs base no-harness): {_delta(s[bh], s[nh])}",
         f"- **Full system** (adapter+harness vs base no-harness): {_delta(s[a], s[nh])}", ""]
    for lab, b in conds:
        sm = s[lab]
        L += [f"## Confusion matrix — {lab} (rows = gold, cols = pred)", _matrix_md(b["cm"]), "",
              f"### Per-class precision / recall / F1 — {lab}", _prf_md(sm["met"]), ""]
    pa = conds[0][1]
    L += ["## Per-slice routing accuracy (adapter+harness)"]
    for sl in sorted(pa["st"]):
        L.append(f"- {sl}: {pa['sc'][sl]}/{pa['st'][sl]} = {pa['sc'][sl] / pa['st'][sl]:.0%}")
    L += ["", "## Per-slice MISS analysis (adapter+harness) — what the misses actually emitted",
          "wrong-name = right verb, wrong target (over-specialization); wrong-verb = wrong KIND "
          "of action. This is the ground truth a bare accuracy number hides.",
          bench_misses.breakdown_md(pa["misses"])]
    stem = f"{date_str}-routing-benchmark" + ("" if suite == "val" else f"-{suite}")
    out = REPO / "docs" / "findings" / f"{stem}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    bench_misses.write_jsonl(pa["misses"], out.with_name(out.stem + "-misses.jsonl"))
    for lab, b in conds:
        _png(b["cm"], out.with_name(out.stem + "-" + lab.replace(" ", "_").replace("+", "-") + ".png"))
    print("\n".join(L[2:]), flush=True)
    print(f"\nwrote {out}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="adapter dir (loads HFModel)")
    ap.add_argument("--server", default="http://127.0.0.1:7861", help="model service URL")
    ap.add_argument("--per-slice", type=int, default=25, help="rows/slice (0 = full val); val suite only")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--time-budget", type=float, default=0, help="seconds PER condition (0 = none)")
    ap.add_argument("--suite", choices=["val", "stress"], default="val",
                    help="val = in-distribution held-out; stress = held-out wild/out-of-domain")
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
    base = AdapterView(model, False)        # same weights, LoRA OFF
    tb, mnt = args.time_budget or None, args.max_new_tokens
    print(f"benchmark [{args.suite}]: {len(rows)} rows x 3 conditions "
          "(adapter+harness, base+harness, base no-harness)", flush=True)
    conds = [
        ("adapter+harness", _bench(model, rows, max_new_tokens=mnt, time_budget_s=tb, label="adapter+harness")),
        ("base+harness", _bench(base, rows, max_new_tokens=mnt, time_budget_s=tb, label="base+harness")),
        ("base no-harness", _bench(base, rows, max_new_tokens=mnt, time_budget_s=tb,
                                   label="base no-harness", with_harness=False)),
    ]
    _write_report(conds, f"{date.today():%Y-%m-%d}", args.suite)


if __name__ == "__main__":
    main()
