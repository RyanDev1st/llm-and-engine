"""Completion-grading eval tier — runs the FULL CoachLoop per row (not just first-action routing)
and scores task COMPLETION + recovery. Answers the peer-review gap that strict first-action scoring
UNDERCOUNTS the harness: a wrong first route that the loop corrects to a grounded answer is a WIN
that routing accuracy records as a loss.

Per-row metrics (grade): first_ok (first action == gold), completed (every expected tool fired),
exec_ok (each expected tool's last result non-error), args_ok (validate_call passed), grounded (reply
cites the last fact tool's result token — a token-presence PROXY), recovered (first action wrong yet
still completed + grounded). The rubric is unit-tested offline (no GPU). Full run on Kaggle:
  python -m llm_training.eval_completion --adapter <best> [--per-slice N --stress]
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


def _failure_detail(row: dict, res: dict, g: dict) -> dict:
    """One failing row, captured so the aggregate is DIAGNOSABLE: which metric(s) failed, what the
    model routed first, and the actual erroring tool results (the cause of an exec_ok miss)."""
    user = next(m for m in row["messages"] if m.get("role") == "user")["content"]
    gv, gn = gold_action(row["messages"])
    calls = res.get("tool_calls", []) or []
    results = res.get("tool_results", []) or []
    first = first_action(calls[0]) if calls else ("none", None)
    errs = [f"{first_action(c)[1]}:{r[:70]}" for c, r in zip(calls, results) if _is_err(r)]
    return {"slice": row["slice"], "user": user[:80], "gold": f"{gv}:{gn}",
            "first": f"{first[0]}:{first[1]}", "failed": [k for k in METRICS if not g[k]],
            "errors": errs[:3]}


def _game_for(row: dict):
    """The board the loop must run on: a chess val row carries a `position_fen` (the position the
    row's expected actions were generated for). Loading it is REQUIRED — without it the loop runs
    at the starting position, so `eval` short-circuits to 0.00 and best_move/review/threats analyze
    the wrong board, making chess completion meaningless. OOD rows have no fen -> starting board."""
    from backend.game import Game
    g = Game()
    fen = row.get("position_fen")
    if fen:
        g.load_fen(fen)
    return g


def run_completion(model, rows: list[dict], *, engine=None, coverage: bool = True,
                   time_budget_s: float | None = None, progress_every: int = 10) -> dict:
    """Run each row through a fresh CoachLoop (real serve loop) and aggregate the rubric. Chess
    rows need a Stockfish `engine`; OOD (life-skills) rows run with engine=None. INTERRUPT-SAFE:
    stops on the time budget and returns what's done (a Kaggle disconnect still yields a result).
    Logs each FAILING row (failures) so a low metric like exec_ok is explained, not a mystery."""
    import time
    from backend.inference import CoachLoop
    from backend.tools import ToolExecutor
    totals = {k: 0 for k in METRICS}
    by_slice: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    failures: list[dict] = []
    t0 = time.time()
    done = 0
    for i, row in enumerate(rows, 1):
        pc = row.get("plugin_context") or {}
        loop = CoachLoop(model, ToolExecutor(_game_for(row), engine, pc), plugin_context=pc)
        user = next(m for m in row["messages"] if m.get("role") == "user")
        res = loop.respond([], user["content"], coverage=coverage,
                           reasoning_mode=row.get("reasoning_mode", ""))
        g = grade(row, res)
        for k, v in g.items():
            totals[k] += int(v)
            by_slice[row["slice"]][k] += int(v)
        by_slice[row["slice"]]["n"] += 1
        # log the actionable failures (not first_ok alone — a recovered route is a WIN, not a bug)
        if not (g["completed"] and g["exec_ok"] and g["grounded"] and g["args_ok"]):
            failures.append(_failure_detail(row, res, g))
        done += 1
        if progress_every and i % progress_every == 0:
            print(f"  {i}/{len(rows)} graded...", flush=True)
        if time_budget_s and time.time() - t0 > time_budget_s:
            print(f"  time budget hit at {done} rows", flush=True)
            break
    return {"n": done, "totals": totals, "by_slice": {k: dict(v) for k, v in by_slice.items()},
            "failures": failures}


def _report(res: dict, label: str) -> str:
    n = res["n"] or 1
    L = [f"## completion eval — {label} (n={res['n']})", "",
         "| metric | rate |", "|---|---|"]
    for k in METRICS:
        L.append(f"| {k} | {res['totals'][k]}/{res['n']} ({100 * res['totals'][k] / n:.1f}%) |")
    L += ["", "_grounded is a token-presence PROXY. recovered = wrong first route that still "
          "completed+grounded (the win strict routing misses)._"]
    fails = res.get("failures") or []
    if fails:
        L += ["", f"### failing rows ({len(fails)}) — why a metric missed",
              "| slice | gold | first | failed | erroring results |", "|---|---|---|---|---|"]
        for f in fails:
            errs = " · ".join(f["errors"]) or "—"
            L.append(f"| {f['slice']} | {f['gold']} | {f['first']} | {','.join(f['failed'])} | {errs} |")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="adapter dir (loads HFModel); else --server")
    ap.add_argument("--gguf", default=None, help="GGUF path (loads GGUFModel) — for the quant A/B")
    ap.add_argument("--server", default="http://127.0.0.1:7861")
    ap.add_argument("--per-slice", type=int, default=0, help="cap rows per slice (0 = all)")
    ap.add_argument("--stress", action="store_true", help="grade the held-out STRESS suite (OOD)")
    ap.add_argument("--no-coverage", dest="coverage", action="store_false")
    ap.add_argument("--time-budget", type=float, default=None)
    ap.add_argument("--tag", default=None, help="model KEY (e.g. e4b-nf4/e4b-q5): writes completion + "
                    "grounded rates to report_assets/measured-<tag>.json for the cross-model chart")
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
    if args.tag and res["n"]:                          # feed the cross-model line chart
        from pathlib import Path as _P
        from llm_training.report.measured import update
        n = res["n"]
        update(_P(__file__).resolve().parents[3] / "docs" / "findings" / "report_assets", args.tag,
               completed=res["totals"]["completed"] / n, grounded=res["totals"]["grounded"] / n)


if __name__ == "__main__":
    main()
