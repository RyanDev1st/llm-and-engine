"""Routing-accuracy audit on the validation set.

For each val conversation we feed the model the first user turn (Mode 1) and
check the tool it routes to against the gold assistant turn. We also probe Mode 2
discipline: after a real tool result, the model must NOT emit another <tool>.

Run from repo root (after training):
  python -m llm_training.eval_routing runs/gemma4_chess
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.model_hf import HFModel  # noqa: E402
from backend.toolfmt import parse_call  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
VAL = REPO / "data" / "sft" / "chess_assistant_v3_val.jsonl"


def gold_tool(messages: list[dict]) -> str | None:
    """The tool the gold assistant used for the first user turn (None = direct)."""
    a = messages[2]["content"] if len(messages) > 2 else ""
    name, _ = parse_call(a)
    return name


def first_turn(messages: list[dict]) -> list[dict]:
    return messages[:2]  # system + first user


def mode2_messages(messages: list[dict]) -> list[dict] | None:
    """system,user,assistant(tool),tool -> prompt that should yield narration."""
    for i, m in enumerate(messages):
        if m["role"] == "tool":
            return messages[: i + 1]
    return None


def main() -> None:
    adapter = sys.argv[1] if len(sys.argv) > 1 else None
    model = HFModel(adapter=adapter, temperature=0.0)
    rows = [json.loads(l) for l in open(VAL, encoding="utf-8")]

    correct = defaultdict(int)
    total = defaultdict(int)
    confusion = defaultdict(int)
    mode2_leak = mode2_total = 0

    for r in rows:
        sl = r["slice"]
        msgs = r["messages"]
        pred_raw = model.generate(first_turn(msgs), max_new_tokens=48, stop=["</tool>"]).strip()
        if pred_raw.startswith("<tool>"):
            if not pred_raw.endswith("</tool>"):
                pred_raw += "</tool>"
            pred, _ = parse_call(pred_raw)
        else:
            pred = None
        gold = gold_tool(msgs)
        total[sl] += 1
        if pred == gold:
            correct[sl] += 1
        else:
            confusion[f"{sl}: gold={gold} pred={pred}"] += 1

        m2 = mode2_messages(msgs)
        if m2:
            mode2_total += 1
            narr = model.generate(m2, max_new_tokens=64, stop=[])
            if "<tool>" in narr:
                mode2_leak += 1

    _report(adapter, correct, total, confusion, mode2_leak, mode2_total)


def _report(adapter, correct, total, confusion, leak, m2total) -> None:
    overall_c = sum(correct.values())
    overall_t = sum(total.values())
    lines = [
        "Parent: docs/superpowers/specs/2026-05-23-chess-coach-sft-design.md", "",
        "# Routing-accuracy audit", "", "## Status",
        f"Overall tool-routing accuracy: {overall_c}/{overall_t} = {overall_c/overall_t:.1%}", "",
        "## Scope", f"Adapter: `{adapter}`. Validation set: {overall_t} conversations.", "",
        "## Evidence", "Per-slice routing accuracy:",
    ]
    for sl in sorted(total):
        lines.append(f"- {sl}: {correct[sl]}/{total[sl]} = {correct[sl]/total[sl]:.0%}")
    lines += ["",
              f"Mode-2 discipline: {m2total - leak}/{m2total} clean "
              f"({leak} records emitted a <tool> after a tool result).", "",
              "## Top routing confusions"]
    for k, v in sorted(confusion.items(), key=lambda x: -x[1])[:12]:
        lines.append(f"- {v}x {k}")
    lines += ["", "## Next", "1. Export merged adapter to Q4_0 GGUF.",
              "2. Wire adapter into the web app and run end-to-end."]
    out = REPO / "docs" / "2026-05-23-routing-audit.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[3:]), flush=True)


if __name__ == "__main__":
    main()
