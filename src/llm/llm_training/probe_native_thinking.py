"""PROBE: does BASE Gemma 4 E4B's NATIVE thinking + native tool-calling reason and
ground correctly out of the box — WITHOUT our custom-XML SFT?

Answers the v5 pivot question: our v1-v4 replaced the model's native trained reasoning
(`<|think|>` / `<|channel>thought`) with custom `<think>` XML + shallow traces. If the
BASE, using native thinking + native `<|tool_call|>`, reasons through a tool result and
answers GROUNDED, that's the green light to build v5 native.

BREADTH (Phase 0 gate): three scenarios — a forced mate (easy), a SUBTLE middlegame (no
mate, positional "why"), and a NON-CHESS domain (cooking) — so we confirm the win
generalizes, not just easy mates. Each feeds canned-CORRECT tool results so the only
question is whether the model's reasoning USES them.

RUN ON A FRESH COLAB RUNTIME (stop the serve first — two E4B won't fit one T4):
  PROBE_BASE=/content/llm-and-engine/src/llm/models/gemma4_e4b \
    PYTHONPATH=src/llm python -m llm_training.probe_native_thinking
"""
from __future__ import annotations

import json
import os
import re

import torch
from transformers import (AutoModelForImageTextToText, AutoProcessor, AutoTokenizer,
                          BitsAndBytesConfig)

BASE = os.environ.get("PROBE_BASE", "unsloth/gemma-4-E4B-it")
MATE = "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"          # White: Re8# back-rank mate. NO bishops.
RUY = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"  # subtle: no forced mate


def _fn(name, desc, props=None):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props or {}, "required": []}}}


CHESS_TOOLS = [
    _fn("board_state", "Return the pieces, side to move, and legal moves of the current board."),
    _fn("best_move", "Return the engine's best move and evaluation.", {"depth": {"type": "integer"}}),
    _fn("eval", "Return who stands better and by how much.", {"depth": {"type": "integer"}}),
    _fn("threats", "Return the opponent's strongest threat."),
]
COOK_TOOLS = [
    _fn("scale_recipe", "Scale a recipe's ingredient quantities by a factor.", {"factor": {"type": "number"}}),
    _fn("convert_units", "Convert a quantity between units.", {"qty": {"type": "string"}, "to": {"type": "string"}}),
]

CHESS_SYS = ("You are a chess coach. The board may be hidden — use the tools to ground every claim, "
             "and never invent pieces, moves, or evaluations.")
COOK_SYS = ("You are a kitchen assistant. Use the tools to compute quantities; never invent amounts.")

# Canned CORRECT results — the only question is whether reasoning USES them.
MATE_CANNED = {
    "board_state": ("board_state: turn=white, check=no. pieces: White Re1, Kg1, pawns f2 g2 h2; "
                    "Black Kg8, pawns f7 g7 h7. NO bishops, knights, or queens. legal incl. Re8 (mate)."),
    "best_move": "best_move: Re8#  — mate in 1 (back-rank). score: White winning (forced mate).",
    "eval": "eval: White has a forced mate (M1).",
    "threats": "threats: White threatens Re8# (back-rank mate). Black has no threat.",
}
RUY_CANNED = {
    "board_state": ("board_state: turn=black, check=no. White Bb5 pins/eyes Nc6; pawns e4; Nf3. "
                    "Black Nc6, e5, standard Ruy Lopez. ~30 legal moves."),
    "best_move": ("best_move: a6 (the Morphy defence). score: +0.2 (roughly level). idea: question the "
                  "b5 bishop — it must decide to take on c6 or retreat to a4, easing Black's game."),
    "eval": "eval: roughly level, a tiny edge for White (+0.2).",
    "threats": "threats: nothing forcing; White's Bb5 eyes c6 but there's no immediate threat.",
}
COOK_CANNED = {
    "scale_recipe": "scale_recipe(factor=2): flour 480 g, butter 226 g, sugar 200 g, eggs 4, vanilla 2 tsp.",
    "convert_units": "convert_units: 480 g flour = 3.8 cups.",
}

SCENARIOS = [
    {"label": "CHESS / forced mate (easy)", "sys": CHESS_SYS, "tools": CHESS_TOOLS, "canned": MATE_CANNED,
     "turns": [f"Here's my position (FEN {MATE}). What's the best move here?",
               "why is that a good move?", "is there a bishop on the board?"]},
    {"label": "CHESS / subtle middlegame (no forced mate — positional why)", "sys": CHESS_SYS,
     "tools": CHESS_TOOLS, "canned": RUY_CANNED,
     "turns": [f"Position FEN {RUY}. What should I play here, and why?"]},
    {"label": "COOKING / non-chess domain (does native reasoning generalize?)", "sys": COOK_SYS,
     "tools": COOK_TOOLS, "canned": COOK_CANNED,
     "turns": ["I'm doubling this cookie recipe — how much flour and butter do I need?"]},
]


def load():
    print(f"loading BASE (4-bit): {BASE}", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)          # the chat template lives on the TOKENIZER
    try:
        proc = AutoProcessor.from_pretrained(BASE)     # only used for parse_response, if present
    except Exception:
        proc = None
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, quantization_config=quant, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": 0})
    return tok, proc, model


def _calls_from(raw_calls):
    out = []
    for c in raw_calls:
        m = re.search(r"call:(\w+)\s*(\{.*\})?", c, re.S)
        if not m:
            continue
        try:
            args = json.loads(m.group(2)) if m.group(2) else {}
        except Exception:
            args = {"_raw": (m.group(2) or "")}
        out.append({"name": m.group(1), "arguments": args})
    return out


def _norm_calls(d):
    norm = []
    for tc in d.get("tool_calls") or []:
        if isinstance(tc, dict) and "function" in tc:
            fn = tc["function"]; norm.append({"name": fn.get("name"), "arguments": fn.get("arguments", {})})
        elif isinstance(tc, dict):
            norm.append({"name": tc.get("name"), "arguments": tc.get("arguments", {})})
    d["tool_calls"] = norm or d.get("tool_calls") or []
    d.setdefault("thinking", "")
    d.setdefault("content", "")
    return d


def parse(proc, raw):
    f = getattr(proc, "parse_response", None)
    if f:
        try:
            r = f(raw)
            d = r if isinstance(r, dict) else {"thinking": getattr(r, "thinking", ""),
                "tool_calls": getattr(r, "tool_calls", []), "content": getattr(r, "content", "")}
            if d.get("tool_calls") or d.get("content") or d.get("thinking"):
                return _norm_calls(d)
        except Exception as e:
            print("  (parse_response failed, using regex):", e)
    th = re.search(r"<\|channel>thought\n(.*?)<channel\|>", raw, re.S)
    raw_calls = re.findall(r"<\|tool_call>(.*?)<tool_call\|>", raw, re.S)
    content = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", raw, flags=re.S)
    content = re.sub(r"<\|tool_call>.*?<tool_call\|>", "", content, flags=re.S)
    content = re.sub(r"<\|?[a-z_]+\|?>|<turn\|>|<\|turn>", "", content).strip()
    return _norm_calls({"thinking": th.group(1).strip() if th else "",
                        "tool_calls": _calls_from(raw_calls), "content": content})


_RENDER_KW = [{"enable_thinking": True}, {}]


def render(tok, messages, tools):
    last = None
    for kw in _RENDER_KW:
        try:
            return tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                           tokenize=False, **kw), kw
        except (TypeError, ValueError) as e:
            last = e
    raise RuntimeError(f"apply_chat_template failed for every kwarg combo: {last}")


def gen(tok, model, messages, tools):
    text, kw = render(tok, messages, tools)
    enc = tok(text, return_tensors="pt").to(model.device)
    n = enc["input_ids"].shape[1]
    out = model.generate(**enc, max_new_tokens=640, do_sample=False)
    return text, tok.decode(out[0][n:], skip_special_tokens=False), kw


def run_turn(tok, proc, model, tools, canned, messages, label, show_prompt=False):
    print("\n" + "=" * 80 + f"\n{label}")
    for step in range(5):
        text, raw, kw = gen(tok, model, messages, tools)
        if show_prompt and step == 0:
            print(f"  kwargs accepted: {kw} | tools in prompt? {'declaration:' in text} | "
                  f"think marker? {('<|think' in text) or ('channel' in text)}")
        p = parse(proc, raw)
        print(f"[step {step}] THINK : {(p['thinking'] or '(none)')[:300]}")
        print(f"[step {step}] CALLS : {p['tool_calls']}")
        print(f"[step {step}] ANSWER: {(p['content'] or '(none)')[:300]}")
        if not p["tool_calls"]:
            messages.append({"role": "assistant", "content": p["content"]})
            return
        messages.append({"role": "assistant", "content": p.get("content", "") or "",
                         "tool_calls": [{"type": "function", "function": {"name": tc["name"],
                          "arguments": tc.get("arguments", {})}} for tc in p["tool_calls"]]})
        for tc in p["tool_calls"]:
            messages.append({"role": "tool", "name": tc["name"],
                             "content": canned.get(tc["name"], f"(no canned result for {tc['name']})")})


def main():
    tok, proc, model = load()
    print("\n>>> Judge: does NATIVE thinking REASON about each tool result and answer GROUNDED?")
    for si, sc in enumerate(SCENARIOS):
        msgs = [{"role": "system", "content": sc["sys"]}]
        for ti, user in enumerate(sc["turns"]):
            msgs.append({"role": "user", "content": user})
            run_turn(tok, proc, model, sc["tools"], sc["canned"], msgs,
                     f"{sc['label']}  — turn {ti + 1}", show_prompt=(si == 0 and ti == 0))
    print("\n" + "=" * 80)
    print("VERDICT cues: real native THINK block? right tool call? GROUNDED 'why' (mate AND positional)?")
    print("correct 'no bishop'? AND does the COOKING turn ground in scale_recipe (generalizes off-chess)?")


if __name__ == "__main__":
    main()
