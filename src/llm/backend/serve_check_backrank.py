"""Regression check for the 'back-rank / phantom-bishop' failure (the EasyChess
screenshot, 2026-06-26): on a back-rank mate puzzle the model hallucinated a
bishop that isn't on the board, never called board_state, and stated wrong piece
squares. This drives the REAL CoachLoop on a pinned FEN through the three
follow-up turns that failed, and reports, per turn:

  (1) did board_state actually FIRE?  (current code force-routes it -> expect yes)
  (2) does the REPLY ground in it, i.e. NOT assert a bishop the board lacks?
      (there is no fabricated-PIECE corrector, so this rides on the model)

Non-mocked: it loads a real model so a load/serve regression shows up here, not
just in scripted unit tests (cf. the silent-fallback-mocked-tests lesson).

Run on the local E2B GGUF (default):
  PYTHONPATH=src/llm python -m backend.serve_check_backrank
Run on the E4B adapter (authoritative, HF):
  PYTHONPATH=src/llm python -m backend.serve_check_backrank --adapter PATH
Point at any GGUF (e.g. an E4B export) with --gguf PATH or CHESS_GGUF_PATH.
"""
from __future__ import annotations

import argparse
import re

from backend.game import Game
from backend.inference import CoachLoop
from backend.tools import ToolExecutor

PUZZLE_FEN = "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"  # Re8# back-rank mate; NO bishops on the board

# Turn 1 (a random_position puzzle request) is seeded as prior context so the FEN
# stays fixed; turns 2-4 are the live follow-ups that failed in the screenshot.
SEED_HISTORY = [
    {"role": "user", "content": "hi, can you kindly give me a puzzle?"},
    {"role": "assistant",
     "content": "It's white's turn to find mate in one on the back rank. What do you think of it?"},
]
TURNS = [
    "why is it backrank?",
    "why is it backrank in this case though",
    "there isnt any bishop here",
]


def _names(out: dict) -> list[str]:
    names = []
    for call in out["tool_calls"]:
        m = re.search(r"<(?:tool|skill)>\s*([a-z_][a-z_-]*)", call)
        names.append(m.group(1) if m else "?")
    return names


def phantom_bishop(reply: str) -> bool:
    """Reply asserts a bishop is/was a factor on a board that has none. The
    screenshot's failures ('if the bishop isn't traded', 'the bishop is gone
    but...'); acknowledging 'there's no bishop' is NOT a phantom."""
    r = reply.lower()
    if "bishop" not in r:
        return False
    ack = ("no bishop" in r or "isn't a bishop" in r or "isnt a bishop" in r
           or "there is no bishop" in r or "you're right" in r or "youre right" in r)
    return not ack


def _build_model(args):
    if args.adapter:
        from backend.model_hf import HFModel
        kw = {"adapter": args.adapter, "temperature": args.temperature}
        if args.base:
            kw["base"] = args.base
        print(f"loading HF base + adapter: {args.adapter}", flush=True)
        return HFModel(**kw)
    from backend.model_gguf import GGUFModel, default_gguf_path
    from pathlib import Path
    path = Path(args.gguf) if args.gguf else default_gguf_path()
    if not path.exists():
        raise SystemExit(f"GGUF not found: {path} (set --gguf or CHESS_GGUF_PATH)")
    print(f"loading GGUF: {path}", flush=True)
    return GGUFModel(gguf=path, temperature=args.temperature)


def run(model) -> list[tuple]:
    try:
        from backend.engine import Engine
        engine = Engine()
    except Exception as e:  # board_state / legal_moves still work without it
        engine = None
        print(f"engine unavailable ({e}); board facts still work", flush=True)

    game = Game()
    if not game.load_fen(PUZZLE_FEN):
        raise SystemExit(f"bad FEN: {PUZZLE_FEN}")
    loop = CoachLoop(model, ToolExecutor(game, engine))

    print(f"\nFEN: {PUZZLE_FEN}")
    print("white Re1,Kg1,pawns f2/g2/h2 | black Kg8,pawns f7/g7/h7 | NO bishops\n")

    history, verdicts = list(SEED_HISTORY), []
    for i, msg in enumerate(TURNS, start=2):
        print("=" * 78)
        print(f"TURN {i} (live):  user> {msg}")
        out = loop.respond(history, msg)
        for call, res in zip(out["tool_calls"], out["tool_results"]):
            cn = re.search(r"<(?:tool|skill)>\s*([a-z_][a-z_-]*)", call)
            print(f"    {('call ' + (cn.group(1) if cn else '?')):22} -> "
                  f"{res[:120].replace(chr(10), ' ')}")
        grounded, phantom = "board_state" in _names(out), phantom_bishop(out["reply"])
        print(f"    board_state fired? {grounded}   |   phantom-bishop? {phantom}")
        print(f"    REPLY: {out['reply']}")
        verdicts.append((i, msg, grounded, phantom))
        history = history + out["turns"]

    if engine is not None:
        engine.quit()
    return verdicts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="HF LoRA adapter path (E4B); omit to use GGUF")
    ap.add_argument("--base", default=None)
    ap.add_argument("--gguf", default=None, help="GGUF path override (else CHESS_GGUF_PATH/default)")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    verdicts = run(_build_model(args))

    print("=" * 78)
    print("VERDICT:")
    ok = True
    for i, msg, grounded, phantom in verdicts:
        good = grounded and not phantom
        ok = ok and good
        print(f"  turn {i}: board_state={grounded} phantom_bishop={phantom}"
              f"  -> {'OK' if good else 'FAIL'}  ({msg})")
    print("\nThe phantom_bishop column is the real signal: board_state firing is")
    print("force-routed, so judge whether the REPLY actually uses the tool result.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
