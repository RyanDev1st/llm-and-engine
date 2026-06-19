"""Routing CONFUSION MATRIX + precision/recall on the validation set.

Routing is framed as a 3-class decision on the model's FIRST action:
  skill  -> it loaded a <skill>NAME</skill>
  tool   -> it called a <tool>NAME ...</tool>
  none   -> it answered directly (no action)

We compare the model's first-action verb to the gold first-action verb and build a
3x3 confusion matrix (rows = gold/actual, cols = predicted), then derive precision,
recall and F1 per class. This makes the skill-vs-tool VERB bias visible (the
gold=tool / pred=skill cell). Also reports exact-NAME accuracy and per-slice routing
accuracy.

Run on the serve box (reuses a running model service if one is up; else loads the
adapter directly):
  python -m llm_training.eval_confusion --adapter runs/gemma4_chess_e4b_kaggle/best
  python -m llm_training.eval_confusion --server http://127.0.0.1:7861   # running service
Optional: --per-slice N  (stratified sample of N rows/slice; default = all).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_training.system_prompt import build_system  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
VAL = REPO / "data" / "sft" / "v1_2_val.jsonl"
CLASSES = ["skill", "tool", "none"]
_ACTION = re.compile(r"<(skill|tool)>\s*([\w./-]+)")


def first_action(text: str) -> tuple[str, str | None]:
    """(verb, name) of the FIRST <skill>/<tool> in text; ('none', None) if absent.
    Tolerates a missing close tag (truncated generation) — keys off the open tag."""
    m = _ACTION.search(text or "")
    return (m.group(1), m.group(2)) if m else ("none", None)


def gold_action(messages: list[dict]) -> tuple[str, str | None]:
    for msg in messages:
        if msg.get("role") == "assistant":
            verb, name = first_action(msg.get("content", ""))
            if verb != "none":
                return verb, name
    return "none", None


def _system(row: dict) -> str:
    return build_system(row.get("skills_index", []), row.get("tool_manifest", []),
                        row.get("plugin_context", {}), reasoning_mode=row.get("reasoning_mode", ""))


def _load_model(args):
    if args.adapter:
        from backend.model_hf import HFModel
        print(f"loading adapter {args.adapter} (one-time)...", flush=True)
        return HFModel(adapter=args.adapter, temperature=0.0)
    import os
    os.environ.setdefault("CHESS_MODEL_SERVER", args.server)
    from backend.model_remote import RemoteModel, server_has_adapter
    print(f"using running model service at {args.server}", flush=True)
    return RemoteModel(has_adapter=server_has_adapter())


def _sample(rows: list[dict], per_slice: int | None) -> list[dict]:
    if not per_slice:
        return rows
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["slice"]].append(r)
    out: list[dict] = []
    for sl in sorted(by):
        out.extend(by[sl][:per_slice])    # first-N is deterministic; val already shuffled at build
    return out


def evaluate(model, rows: list[dict]):
    # cm[gold][pred] counts; name_hit = exact-name matches among same-verb-correct rows
    cm = {g: {p: 0 for p in CLASSES} for g in CLASSES}
    name_hit = name_tot = 0
    slice_c: dict[str, int] = defaultdict(int)
    slice_t: dict[str, int] = defaultdict(int)
    for i, r in enumerate(rows, 1):
        user = next(m for m in r["messages"] if m.get("role") == "user")
        out = model.generate([{"role": "system", "content": _system(r)}, user],
                             max_new_tokens=96, stop=["</skill>", "</tool>"])
        p_verb, p_name = first_action(out)
        g_verb, g_name = gold_action(r["messages"])
        cm[g_verb][p_verb] += 1
        slice_t[r["slice"]] += 1
        if p_verb == g_verb and p_name == g_name:
            slice_c[r["slice"]] += 1
        if g_verb != "none":
            name_tot += 1
            name_hit += int(p_verb == g_verb and p_name == g_name)
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    return cm, (name_hit, name_tot), (slice_c, slice_t)


def _metrics(cm: dict) -> dict:
    out = {}
    for c in CLASSES:
        tp = cm[c][c]
        fp = sum(cm[g][c] for g in CLASSES) - tp
        fn = sum(cm[c][p] for p in CLASSES) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def _png(cm: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                       # text report still stands without the plot
        print(f"(matplotlib unavailable: {exc}; skipping PNG)", flush=True)
        return
    mat = [[cm[g][p] for p in CLASSES] for g in CLASSES]
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(3), [f"pred\n{c}" for c in CLASSES])
    ax.set_yticks(range(3), [f"gold {c}" for c in CLASSES])
    thr = max(max(r) for r in mat) / 2 or 1
    for g in range(3):
        for p in range(3):
            ax.text(p, g, mat[g][p], ha="center", va="center",
                    color="white" if mat[g][p] > thr else "black", fontsize=12)
    ax.set_title("Routing verb confusion (val)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="adapter dir (loads HFModel)")
    ap.add_argument("--server", default="http://127.0.0.1:7861", help="model service URL")
    ap.add_argument("--per-slice", type=int, default=0, help="rows/slice (0 = all)")
    args = ap.parse_args()

    from llm_dataset.v1.jsonl_io import read_rows
    rows = _sample(list(read_rows(VAL)), args.per_slice or None)
    model = _load_model(args)
    cm, (nh, nt), (sc, st) = evaluate(model, rows)
    met = _metrics(cm)

    total = sum(cm[g][p] for g in CLASSES for p in CLASSES)
    acc = sum(cm[c][c] for c in CLASSES) / total if total else 0.0
    macro_p = sum(met[c]["precision"] for c in CLASSES) / len(CLASSES)
    wsupport = sum(met[c]["support"] for c in CLASSES) or 1
    weighted_p = sum(met[c]["precision"] * met[c]["support"] for c in CLASSES) / wsupport

    from datetime import date
    L = ["Parent: docs/reference/sft-corpus-generation.md", "",
         "# Routing confusion matrix + precision (val)", "", "## Status",
         f"Verb accuracy: {acc:.1%} | macro-precision: {macro_p:.1%} | "
         f"weighted-precision: {weighted_p:.1%} | exact-name (skill/tool rows): "
         f"{nh}/{nt} = {nh/nt:.1%}" if nt else f"Verb accuracy: {acc:.1%}", "",
         "## Confusion matrix (rows = gold, cols = predicted)",
         "| gold \\ pred | " + " | ".join(CLASSES) + " |",
         "|---|" + "---|" * len(CLASSES)]
    for g in CLASSES:
        L.append(f"| {g} | " + " | ".join(str(cm[g][p]) for p in CLASSES) + " |")
    L += ["", "## Per-class precision / recall / F1",
          "| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    for c in CLASSES:
        m = met[c]
        L.append(f"| {c} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['support']} |")
    L += ["", "## Per-slice exact routing accuracy"]
    for sl in sorted(st):
        L.append(f"- {sl}: {sc[sl]}/{st[sl]} = {sc[sl]/st[sl]:.0%}")

    out = REPO / "docs" / "findings" / f"{date.today():%Y-%m-%d}-routing-confusion.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    _png(cm, out.with_suffix(".png"))
    print("\n".join(L[3:]), flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
