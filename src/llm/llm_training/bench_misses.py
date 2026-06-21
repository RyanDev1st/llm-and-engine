"""Per-row MISS log for the routing benchmark — turns a bare per-slice score (e.g. "slice G 0/25")
into an EXPLAINED one: among a slice's missed rows, what did the model ACTUALLY emit? It separates
the two failure modes a single accuracy number hides:
  - WRONG-NAME  (right verb, wrong target)  -> over-specialization / sibling confusion
  - WRONG-VERB  (skill vs tool vs none)      -> the model chose the wrong KIND of action

Without this the eval discards predictions and the failure mode is UNKNOWABLE. Concretely, the
2026-06-21 benchmark showed val slice G/H near 0% and it was read as over-specialization — but
slice H's prompt is "I'm worried here - undo that", which could equally be a verb-miss to the
`undo` tool. On the STRESS suite the same log reveals whether `<skill>metronome_bpm</skill>` (a
tool emitted as a skill) actually occurs. Writes a bounded JSONL + an inline markdown table."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

CAP = 600  # bound the JSONL so a full-val run can't bloat docs/findings


def record(misses: list, *, slice_: str, user: str, gold: tuple, pred: tuple, out: str) -> None:
    """Append one missed row (capped). gold/pred are (verb, name) tuples from first_action."""
    if len(misses) >= CAP:
        return
    gv, gn = gold
    pv, pn = pred
    misses.append({"slice": slice_, "user": (user or "")[:160], "gold_verb": gv, "gold_name": gn,
                   "pred_verb": pv, "pred_name": pn, "out": (out or "")[:160]})


def _kind(m: dict) -> str:
    return f"wrong-verb->{m['pred_verb']}" if m["pred_verb"] != m["gold_verb"] else "wrong-name"


def breakdown_md(misses: list) -> str:
    """Per-slice table: miss count, the kind split (wrong-name vs wrong-verb->X), and the single
    most-common wrong target. Only slices that actually missed appear."""
    by: dict[str, list] = defaultdict(list)
    for m in misses:
        by[m["slice"]].append(m)
    if not by:
        return "_no misses recorded_"
    L = ["| slice | misses | kind x count | top wrong target (verb:name) |", "|---|---|---|---|"]
    for sl in sorted(by):
        ms = by[sl]
        kinds: dict[str, int] = defaultdict(int)
        tgt: dict[str, int] = defaultdict(int)
        for m in ms:
            kinds[_kind(m)] += 1
            tgt[f"{m['pred_verb']}:{m['pred_name']}"] += 1
        kd = ", ".join(f"{k}:{v}" for k, v in sorted(kinds.items(), key=lambda x: -x[1]))
        top, n = max(tgt.items(), key=lambda x: x[1])
        L.append(f"| {sl} | {len(ms)} | {kd} | {top} ({n}) |")
    return "\n".join(L)


def write_jsonl(misses: list, path: Path) -> Path:
    path.write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in misses) + "\n",
                    encoding="utf-8")
    return path
