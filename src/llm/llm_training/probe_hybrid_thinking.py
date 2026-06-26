"""PROBE (v4.1 hybrid): keep our custom <skill>/<tool> XML harness, but add Gemma's
NATIVE reasoning via enable_thinking instead of the shallow <think> stubs v4 trained
(which suppressed reasoning). Precondition check before any retrain:

  (1) Does the base reason natively when given OUR harness system prompt
      (build_system, which describes the custom <skill>/<tool> contract)?
  (2) What action format does it gravitate to — our <tool>/<skill>, or native
      <|tool_call|>? (scopes the format-collision risk the retrain must beat.)

This is the cheap go/no-go for the hybrid. The REAL test — does training make it emit
OUR XML after a native thought — is a micro-overfit on GPU (the next gate).

RUN ON A FRESH COLAB RUNTIME:
  PROBE_BASE=/content/llm-and-engine/src/llm/models/gemma4_e4b \
    PYTHONPATH=src/llm python -m llm_training.probe_hybrid_thinking
"""
from __future__ import annotations

import re

from llm_training.probe_native_thinking import load, parse
from llm_training.system_prompt import build_system

FEN = "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"

# Minimal but realistic harness catalog (chess), shaped like the corpus rows.
SKILLS = [{"name": "chess-coach", "description": "analyze a position, choose/explain a move, inspect the board",
           "plugin": "chess-official", "source": "official_plugin", "enabled": True}]
TOOLS = [
    {"name": "board_state", "description": "pieces, side to move, legal moves", "args": {"fields": "basic|all"},
     "applies_when": "", "plugin": "chess-official", "source": "official", "enabled": True},
    {"name": "best_move", "description": "engine's best move + evaluation", "args": {"depth": "int"},
     "applies_when": "", "plugin": "chess-official", "source": "official", "enabled": True},
    {"name": "eval", "description": "who stands better and by how much", "args": {"depth": "int"},
     "applies_when": "", "plugin": "chess-official", "source": "official", "enabled": True},
]
PC = {"installed": ["chess-official"], "enabled": ["chess-official"]}

QUESTIONS = [
    f"FEN {FEN}. What's the best move here, and why?",
    "is my king safe?",
]


def render(tok, messages):
    # ACTION contract only (reasoning_mode="") so the system prompt describes <skill>/<tool>
    # but NOT a custom <think>; reasoning comes from enable_thinking. No native tools= passed —
    # our tools live in the system text, the whole point of the hybrid.
    for kw in ({"enable_thinking": True}, {}):
        try:
            return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kw), kw
        except (TypeError, ValueError):
            continue
    raise RuntimeError("apply_chat_template failed")


def main():
    tok, proc, model = load()
    system = build_system(SKILLS, TOOLS, PC, reasoning_mode="")
    print("=== SYSTEM PROMPT (our harness contract, action-only) — head ===")
    print(system[:700] + "\n...")
    for q in QUESTIONS:
        print("\n" + "=" * 80 + f"\nUSER: {q}")
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": q}]
        text, kw = render(tok, msgs)
        enc = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=512, do_sample=False)
        raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
        th = re.search(r"<\|channel>thought\n(.*?)<channel\|>", raw, re.S)
        print(f"kwargs={kw} | native THOUGHT? {bool(th)} | "
              f"emits OUR <tool>/<skill>? {('<tool>' in raw) or ('<skill>' in raw)} | "
              f"drifts NATIVE <|tool_call>? {'tool_call' in raw}")
        if th:
            print("THOUGHT:", th.group(1).strip()[:280])
        print("RAW (first 700):\n", raw[:700])
    print("\n" + "=" * 80)
    print("VERDICT: (1) does native reasoning fire under our harness contract? (quality?)")
    print("(2) action format — our XML or native drift? (the gap the v4.1 retrain must close)")


if __name__ == "__main__":
    main()
