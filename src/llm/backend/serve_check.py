"""Live serve check: load the trained LoRA adapter on the local GPU + real
Stockfish, run the demo scenarios through the real CoachLoop, and print the
tool trace + reply + context stats. Lets us eyeball routing, qualitative
narration, grounding, and multi-turn back-reference before the presentation.

Run:
  PYTHONPATH=src/llm python -m backend.serve_check \
      --adapter "A:/Download/gemma4_chess_kaggle_adapter"
"""
from __future__ import annotations

import argparse
import re

from backend.engine import Engine
from backend.game import Game
from backend.inference import CoachLoop
from backend.model_hf import HFModel
from backend.tools import ToolExecutor


def _trace(out: dict) -> None:
    for call, res in zip(out["tool_calls"], out["tool_results"]):
        m = re.search(r"<tool>\s*([a-z_]+)", call)
        name = m.group(1) if m else "?"
        print(f"    tool {name:12} -> {res[:100].replace(chr(10), ' ')}")
    ctx = out["context"]
    print(f"    ctx kept={ctx['turns_kept']} evicted={ctx['turns_evicted']} "
          f"used={ctx['used_tokens']}/{ctx['budget']}")
    print(f"    REPLY: {out['reply']}")


def _position(moves: list[str]) -> Game:
    g = Game()
    for san in moves:
        r = g.move(san)
        if r.startswith("error"):
            raise SystemExit(f"bad setup move {san!r}: {r}")
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    print("loading base + adapter (4-bit on GPU)...", flush=True)
    kw = {"adapter": args.adapter, "temperature": args.temperature}
    if args.base:
        kw["base"] = args.base
    model = HFModel(**kw)
    engine = Engine()

    # Ground truth for the headline position, straight from the engine.
    g0 = _position(["d4", "e5", "Nc3", "exd4", "Bg5"])
    kind, val = engine.eval_white_cp(g0.board, 18)
    truth = f"{kind}:{val}"
    print(f"\n[ground truth] after 1.d4 e5 2.Nc3 exd4 3.Bg5  ->  engine eval = {truth} "
          f"(white POV centipawns; negative = Black better)\n")

    # --- Scenario A: losing-for-White position, qualitative narration -------
    print("=" * 78)
    print("A. 'how am I doing?' (White, down a pawn) — expect qualitative, no fake number")
    gA = _position(["d4", "e5", "Nc3", "exd4", "Bg5"])
    loopA = CoachLoop(model, ToolExecutor(gA, engine))
    outA = loopA.respond([], "I'm playing White. How am I doing here?")
    _trace(outA)

    # --- Scenario C: multi-turn follow-up — should back-reference, not re-dump
    print("=" * 78)
    print("C. follow-up 'why?' — expect it to build on the prior turn, not restate FEN")
    historyC = [{"role": "user", "content": "I'm playing White. How am I doing here?"},
                {"role": "assistant", "content": outA["reply"]}]
    gC = _position(["d4", "e5", "Nc3", "exd4", "Bg5"])
    loopC = CoachLoop(model, ToolExecutor(gC, engine))
    outC = loopC.respond(historyC, "why?")
    _trace(outC)

    # --- Scenario D: explicit number request — now the exact eval is allowed -
    print("=" * 78)
    print("D. 'what's the exact eval in pawns?' — expect the real number, grounded")
    gD = _position(["d4", "e5", "Nc3", "exd4", "Bg5"])
    loopD = CoachLoop(model, ToolExecutor(gD, engine))
    outD = loopD.respond([], "What's the exact evaluation in pawns?")
    _trace(outD)

    engine.quit()
    print("=" * 78)
    print("done. judge: is the eval sign/magnitude consistent with ground truth above,")
    print("or fabricated? does D's number match the engine? does C reference the prior turn?")


if __name__ == "__main__":
    main()
