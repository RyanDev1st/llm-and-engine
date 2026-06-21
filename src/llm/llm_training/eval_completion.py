"""Completion-grading eval tier — runs the FULL CoachLoop per row (not just first-action routing)
and scores task COMPLETION + recovery. Answers the peer-review gap that strict first-action scoring
UNDERCOUNTS the harness: a wrong first route that the loop corrects to a grounded answer is a WIN
that routing accuracy records as a loss.

Per-row metrics (grade), aggregated adapter-vs-base:
  first_ok    the FIRST action matched gold (the strict routing metric, for the delta)
  completed   every expected tool name fired
  exec_ok     every expected tool's (last) result was non-error
  args_ok     every executed <tool> call passed validate_call (no missing-required / bad-enum)
  grounded    the final reply cites the last fact tool's result token (reuse _result_signal) — a
              PROXY (token presence), report it as such
  recovered   the first action was wrong (!= gold) yet the loop still completed + grounded

The rubric (grade) is unit-tested offline with scripted results — no GPU. Full run on Kaggle:
  python -m llm_training.eval_completion --adapter <best> [--per-slice N --stress --coverage/--no-coverage]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.inference import _result_signal              # noqa: E402  reuse the grounding signal
from backend.toolfmt import parse_call                    # noqa: E402
from backend.tools import validate_call                   # noqa: E402
from llm_training.eval_confusion import (                 # noqa: E402
    VAL, _load_model, _sample, first_action, gold_action)

METRICS = ("first_ok", "completed", "exec_ok", "args_ok", "grounded", "recovered")


def _is_err(text: str | None) -> bool:
    return bool(text) and text.strip().startswith("error")


def _expected(row: dict, gverb: str, gname: str | None) -> list[str]:
    """The tool names this row should fire. Val rows carry an explicit list; stress/derived rows
    fall back to the single gold action's name (a tool fires itself; a skill appears as its name)."""
    exp = row.get("expected_tool_calls")
    if exp:
        return list(exp)
    return [gname] if gverb in ("tool", "skill") and gname else []


def _result_for(name: str, calls: list[str], results: list[str]) -> str | None:
    """The LAST result whose call has this name (so a corrective error followed by a successful
    retry reads as success — what the user actually got)."""
    found = None
    for c, r in zip(calls, results):
        if first_action(c)[1] == name:
            found = r
    return found


def _grounded(reply: str, calls: list[str], results: list[str]) -> bool:
    """Did the reply reflect the last FACT (non-error <tool>) result? Skill loads and errors carry
    no fact token. No fact tool -> a non-empty direct answer counts (decline). A None signal (no
    extractable token, e.g. a piece list) can't be disproven -> count it grounded (conservative,
    same restraint as the serve-side grounding guard)."""
    facts = [(first_action(c)[1], r) for c, r in zip(calls, results)
             if first_action(c)[0] == "tool" and not _is_err(r)]
    if not facts:
        return bool((reply or "").strip())
    sig = _result_signal(facts[-1][1])
    return sig is None or sig.lower() in (reply or "").lower()


def grade(row: dict, result: dict) -> dict:
    """Score one loop result against the row's gold + expected tools. Pure — no model."""
    calls = result.get("tool_calls", []) or []
    results = result.get("tool_results", []) or []
    reply = result.get("reply", "") or ""
    actions = [first_action(c) for c in calls]              # (verb, name) per executed step
    exec_names = {n for _, n in actions if n}
    gverb, gname = gold_action(row["messages"])
    expected = _expected(row, gverb, gname)
    first = actions[0] if actions else ("none", None)

    args_ok = True
    for c in calls:
        nm, ar = parse_call(c)
        if nm and validate_call(nm, ar) is not None:
            args_ok = False
            break
    completed = set(expected) <= exec_names
    exec_ok = (all(not _is_err(_result_for(n, calls, results)) for n in expected)
               if expected else not any(_is_err(r) for r in results))
    grounded = _grounded(reply, calls, results)
    first_ok = first == (gverb, gname) if gverb != "none" else first == ("none", None)
    recovered = gverb != "none" and not first_ok and completed and grounded
    return {"first_ok": first_ok, "completed": completed, "exec_ok": exec_ok,
            "args_ok": args_ok, "grounded": grounded, "recovered": recovered}


def run_completion(model, rows: list[dict], *, engine=None, coverage: bool = True,
                   time_budget_s: float | None = None, progress_every: int = 10) -> dict:
    """Run each row through a fresh CoachLoop (real serve loop) and aggregate the rubric. Chess
    rows need a Stockfish `engine`; OOD (life-skills) rows run with engine=None. INTERRUPT-SAFE:
    stops on the time budget and returns what's done (a Kaggle disconnect still yields a result)."""
    import time
    from backend.game import Game
    from backend.inference import CoachLoop
    from backend.tools import ToolExecutor
    totals = {k: 0 for k in METRICS}
    by_slice: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    t0 = time.time()
    done = 0
    for i, row in enumerate(rows, 1):
        pc = row.get("plugin_context") or {}
        loop = CoachLoop(model, ToolExecutor(Game(), engine, pc), plugin_context=pc)
        user = next(m for m in row["messages"] if m.get("role") == "user")
        res = loop.respond([], user["content"], coverage=coverage,
                           reasoning_mode=row.get("reasoning_mode", ""))
        g = grade(row, res)
        for k, v in g.items():
            totals[k] += int(v)
            by_slice[row["slice"]][k] += int(v)
        by_slice[row["slice"]]["n"] += 1
        done += 1
        if progress_every and i % progress_every == 0:
            print(f"  {i}/{len(rows)} graded...", flush=True)
        if time_budget_s and time.time() - t0 > time_budget_s:
            print(f"  time budget hit at {done} rows", flush=True)
            break
    return {"n": done, "totals": totals, "by_slice": {k: dict(v) for k, v in by_slice.items()}}


def _report(res: dict, label: str) -> str:
    n = res["n"] or 1
    L = [f"## completion eval — {label} (n={res['n']})", "",
         "| metric | rate |", "|---|---|"]
    for k in METRICS:
        L.append(f"| {k} | {res['totals'][k]}/{res['n']} ({100 * res['totals'][k] / n:.1f}%) |")
    L += ["", "_grounded is a token-presence PROXY. recovered = wrong first route that still "
          "completed+grounded (the win strict routing misses)._"]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="adapter dir (loads HFModel); else --server")
    ap.add_argument("--server", default="http://127.0.0.1:7861")
    ap.add_argument("--per-slice", type=int, default=0, help="cap rows per slice (0 = all)")
    ap.add_argument("--stress", action="store_true", help="grade the held-out STRESS suite (OOD)")
    ap.add_argument("--no-coverage", dest="coverage", action="store_false")
    ap.add_argument("--time-budget", type=float, default=None)
    args = ap.parse_args()
    if args.stress:
        from llm_training.bench_suites import stress_rows
        rows, label, engine = stress_rows(), "STRESS (OOD, life-skills)", None
    else:
        from llm_dataset.v1.jsonl_io import read_rows
        from backend.engine import Engine
        rows, label, engine = list(read_rows(VAL)), "VAL (chess)", Engine()
    rows = _sample(rows, args.per_slice or None)
    model = _load_model(args)
    res = run_completion(model, rows, engine=engine, coverage=args.coverage,
                         time_budget_s=args.time_budget)
    print("\n" + _report(res, label), flush=True)


if __name__ == "__main__":
    main()
