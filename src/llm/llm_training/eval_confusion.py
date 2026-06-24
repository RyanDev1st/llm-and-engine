"""Routing CONFUSION MATRIX + precision/recall on the validation set.

Routing = a 3-class decision on the FIRST action: skill (<skill>NAME</skill>), tool
(<tool>NAME ...</tool>), or none. We compare the model's first-action verb to the gold verb ->
a 3x3 matrix (rows=gold, cols=pred) + per-class precision/recall/F1 (surfacing the skill-vs-tool
VERB bias), plus exact-NAME + per-slice accuracy. Light + interrupt-safe: small --per-slice, a
--time-budget, and FAST-mode scoring keep it short enough to beat a Colab disconnect; a stop
writes the matrix from completed rows.
  python -m llm_training.eval_confusion --adapter <best> [--per-slice 8 --time-budget 900]
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


def _system(row: dict, force_fast: bool = True) -> str:
    # Force FAST mode: routing (skill/tool/none) is mode-independent, but fast skips the
    # goal/think preamble so the action lands fast and early-stops (~5x faster on a T4).
    mode = "fast" if force_fast else row.get("reasoning_mode", "")
    return build_system(row.get("skills_index", []), row.get("tool_manifest", []),
                        row.get("plugin_context", {}), reasoning_mode=mode)


def _load_model(args):
    gguf = getattr(args, "gguf", None)
    if gguf:
        from backend.model_gguf import GGUFModel, gguf_runtime_config
        n_ctx, n_gpu = gguf_runtime_config()
        print(f"loading GGUF {gguf} (one-time)...", flush=True)
        return GGUFModel(gguf=gguf, n_ctx=n_ctx, n_gpu_layers=n_gpu, temperature=0.0)
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


def evaluate(model, rows: list[dict], max_new_tokens: int = 24, time_budget_s: float | None = None,
             progress_every: int = 20, force_fast: bool = True):
    """Build the confusion matrix over `rows`. INTERRUPT-SAFE: stops early when `time_budget_s`
    is exceeded and returns the matrix from the rows DONE so far (a Colab disconnect / usage cap
    still yields a usable partial result). Stop tokens end most rows in a few tokens."""
    import time
    cm = {g: {p: 0 for p in CLASSES} for g in CLASSES}
    name_hit = name_tot = 0
    slice_c: dict[str, int] = defaultdict(int)
    slice_t: dict[str, int] = defaultdict(int)
    t0 = time.time()
    done = 0
    for i, r in enumerate(rows, 1):
        user = next(m for m in r["messages"] if m.get("role") == "user")
        out = model.generate([{"role": "system", "content": _system(r, force_fast)}, user],
                             max_new_tokens=max_new_tokens, stop=["</skill>", "</tool>"])
        p_verb, p_name = first_action(out)
        g_verb, g_name = gold_action(r["messages"])
        cm[g_verb][p_verb] += 1
        slice_t[r["slice"]] += 1
        if p_verb == g_verb and p_name == g_name:
            slice_c[r["slice"]] += 1
        if g_verb != "none":
            name_tot += 1
            name_hit += int(p_verb == g_verb and p_name == g_name)
        done = i
        if i % progress_every == 0:
            el = time.time() - t0
            print(f"  {i}/{len(rows)} ({el:.0f}s, {el/i:.1f}s/row)", flush=True)
        if time_budget_s and (time.time() - t0) > time_budget_s:
            print(f"  budget {time_budget_s:.0f}s reached — stop at {done}/{len(rows)} (partial matrix)", flush=True)
            break
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


def confusion_caption(cm: dict, name_hits: int | None = None, name_tot: int | None = None) -> str:
    """Plain-English description baked UNDER the matrix in the PNG (pure — unit-tested). Explains the
    3-class verb routing + the diagonal, then the headline numbers computed from the matrix itself.
    When the exact-NAME tally is passed (the standalone run has it), add that line too."""
    met = _metrics(cm)
    total = sum(cm[g][p] for g in CLASSES for p in CLASSES) or 1
    acc = sum(cm[c][c] for c in CLASSES) / total
    macro_p = sum(met[c]["precision"] for c in CLASSES) / len(CLASSES)
    lines = [
        "Routing = the model's FIRST action on each held-out request, scored as one of three",
        "verbs: skill (load a skill body) · tool (call a function) · none (answer directly).",
        "Rows = the correct (gold) verb, columns = what the model chose; the diagonal is correct.",
        "",
        f"Verb accuracy {acc:.1%}  ·  macro-precision {macro_p:.1%}  ·  n={total} val rows.",
    ]
    if name_tot:
        lines.append(f"Exact tool/skill NAME match (on skill/tool rows): "
                     f"{name_hits}/{name_tot} = {name_hits / name_tot:.1%}.")
    return "\n".join(lines)


def _png(cm: dict, path: Path, caption: str | None = None) -> None:
    """Render the matrix WITH its description baked into the image (one copy-paste for a slide). The
    per-condition matrices in bench_report flow through here too, so each carries its own legend."""
    from llm_training.report.ppt_charts import confusion_matrix
    try:
        confusion_matrix(cm, CLASSES, path, caption or confusion_caption(cm))
    except Exception as exc:                       # text report still stands without the plot
        print(f"(matplotlib unavailable: {exc}; skipping PNG)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="adapter dir (loads HFModel)")
    ap.add_argument("--gguf", default=None, help="GGUF path (loads GGUFModel) — for the quant A/B")
    ap.add_argument("--server", default="http://127.0.0.1:7861", help="model service URL")
    ap.add_argument("--per-slice", type=int, default=8, help="rows/slice (0 = all; default 8 = light)")
    ap.add_argument("--max-new-tokens", type=int, default=24, help="gen cap per row (fast mode acts early)")
    ap.add_argument("--time-budget", type=float, default=0, help="seconds; 0 = no budget. Stops "
                    "early + writes the matrix from completed rows (survives a Colab disconnect)")
    ap.add_argument("--tag", default=None, help="model KEY (e.g. e4b-nf4/e4b-q5/e2b-adapter): writes "
                    "the verb accuracy to report_assets/measured-<tag>.json for the cross-model chart")
    args = ap.parse_args()

    from llm_dataset.v1.jsonl_io import read_rows
    rows = _sample(list(read_rows(VAL)), args.per_slice or None)
    model = _load_model(args)
    cm, (nh, nt), (sc, st) = evaluate(model, rows, max_new_tokens=args.max_new_tokens,
                                      time_budget_s=args.time_budget or None)
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
    _png(cm, out.with_suffix(".png"), confusion_caption(cm, nh, nt))
    if args.tag:                                        # feed the cross-model line chart (verb only)
        from llm_training.report.measured import update
        update(REPO / "docs" / "findings" / "report_assets", args.tag,
               verb=acc, exact=(nh / nt if nt else None))
    print("\n".join(L[3:]), flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
