"""Markdown rendering for the routing benchmark — the report-writing half of eval_benchmark, split
out to keep each file under the size cap. Pure functions over the per-condition bench dicts that
_bench produces: summary metrics, the confusion-matrix / precision tables, the headline, the deltas,
and the full report file (+ per-condition PNGs + the miss-log JSONL)."""
from __future__ import annotations

from pathlib import Path

from llm_training import bench_misses
from llm_training.eval_confusion import CLASSES, _metrics, _png

REPO = Path(__file__).resolve().parents[3]


def summary(b: dict) -> dict:
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


def write_report(conds: list, date_str: str, suite: str = "val") -> Path:
    """conds = [(label, bench_result), ...]; the FIRST is the product (E4B v4 adapter+harness) the
    others are measured against. `suite` ('val' or 'stress') tags the title + filename."""
    summ = [(lab, summary(b)) for lab, b in conds]
    s = {lab: sm for lab, sm in summ}
    prod = conds[0][0]
    src = ("held-out STRESS rows (messy/slang/typo phrasing + UNSEEN out-of-domain catalogs + "
           "decline cases — hand-written, in NO training row)" if suite == "stress"
           else "stratified val rows (held out from training; in-distribution phrasing)")
    L = ["Parent: docs/reference/sft-corpus-generation.md", "",
         f"# Routing benchmark ({suite}) — Gemma 4 chess-coach", "",
         f"n = {conds[0][1]['n']} {src} · fast mode (routing is mode-independent) · the E4B "
         "base+harness condition reuses the loaded E4B model with the LoRA disabled; the E2B "
         "condition (if present) is the prior-production adapter on its OWN E2B base.", "",
         f"## Headline — {len(conds)} conditions (product = {prod})", _headline(summ), "",
         "## What each delta isolates (vs the product)"]
    for lab, _ in conds[1:]:
        L.append(f"- **{prod} vs {lab}**: {_delta(s[prod], s[lab])}")
    L.append("")
    for lab, b in conds:
        L += [f"## Confusion matrix — {lab} (rows = gold, cols = pred)", _matrix_md(b["cm"]), "",
              f"### Per-class precision / recall / F1 — {lab}", _prf_md(s[lab]["met"]), ""]
    pa = conds[0][1]
    L += [f"## Per-slice routing accuracy ({prod})"]
    for sl in sorted(pa["st"]):
        L.append(f"- {sl}: {pa['sc'][sl]}/{pa['st'][sl]} = {pa['sc'][sl] / pa['st'][sl]:.0%}")
    L += ["", f"## Per-slice MISS analysis ({prod}) — what the misses actually emitted",
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
