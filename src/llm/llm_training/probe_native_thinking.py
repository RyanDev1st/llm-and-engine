"""PROBE: does BASE Gemma 4 E4B's NATIVE thinking + native tool-calling reason and
ground correctly out of the box — WITHOUT our custom-XML SFT?

This answers the v5 pivot question: our v1-v4 replaced the model's native trained
reasoning (`<|think|>` / `<|channel>thought`, up to ~4k-token CoT) with custom
`<think>` XML + shallow templated traces. If the BASE, using its native thinking
and native `<|tool_call|>` schema, reasons through a tool result and answers
GROUNDED, that's the green light to build v5 native. If it confabulates too, the
format was never the lever and we focus on content.

Loads the BASE only (no adapter), native path: AutoProcessor.apply_chat_template
with tools=... and enable_thinking=True, then parse_response (with a regex
fallback from the tokenizer's x-regex). Feeds canned, CORRECT tool results for a
known position so the only question is whether the model's reasoning USES them.

RUN ON A FRESH COLAB RUNTIME (stop the serve first — two E4B won't fit one T4):
  # point at the base you already downloaded to skip a re-download:
  PROBE_BASE=/content/llm-and-engine/src/llm/models/gemma4_e4b \
    PYTHONPATH=src/llm python -m llm_training.probe_native_thinking
Defaults to the HF repo 'unsloth/gemma-4-E4B-it' if PROBE_BASE is unset.
"""
from __future__ import annotations

import json
import os
import re

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

BASE = os.environ.get("PROBE_BASE", "unsloth/gemma-4-E4B-it")
FEN = "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"  # White: Re8# back-rank mate. NO bishops.

SYS = ("You are a chess coach. The board may be hidden — use the tools to ground every "
       "claim about the position, and never invent pieces, moves, or evaluations.")

def _fn(name, desc, props=None):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props or {}, "required": []}}}

TOOLS = [
    _fn("board_state", "Return the pieces, side to move, and legal moves of the current board."),
    _fn("best_move", "Return the engine's best move and evaluation.", {"depth": {"type": "integer"}}),
    _fn("eval", "Return who stands better and by how much.", {"depth": {"type": "integer"}}),
    _fn("threats", "Return the opponent's strongest threat."),
]

# Canned CORRECT results for FEN above — the only question is whether reasoning USES them.
CANNED = {
    "board_state": ("board_state: turn=white, check=no. pieces: White Re1, Kg1, pawns f2 g2 h2; "
                    "Black Kg8, pawns f7 g7 h7. There are NO bishops, knights, or queens on the board. "
                    "legal moves include Re8 (delivers mate)."),
    "best_move": "best_move: Re8#  — mate in 1 (back-rank). score: White is winning (forced mate).",
    "eval": "eval: White has a forced mate (M1). Decisively winning.",
    "threats": "threats: White threatens Re8# (back-rank mate). Black has no threat.",
}


def load():
    print(f"loading BASE (4-bit): {BASE}", flush=True)
    proc = AutoProcessor.from_pretrained(BASE)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, quantization_config=quant, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": 0})
    return proc, model


def parse(proc, raw):
    """thinking / tool_calls / content from native output. Prefer processor.parse_response;
    fall back to the tokenizer x-regex (`<|channel>thought…<channel|>`, `<|tool_call>…<tool_call|>`)."""
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
    return d


def gen(proc, model, messages):
    text = proc.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True,
                                    tokenize=False, enable_thinking=True)
    enc = proc(text=text, return_tensors="pt").to(model.device)
    n = enc["input_ids"].shape[1]
    out = model.generate(**enc, max_new_tokens=640, do_sample=False)
    raw = proc.decode(out[0][n:], skip_special_tokens=False)
    return text, raw


def run_turn(proc, model, messages, label, show_prompt=False):
    print("\n" + "=" * 80 + f"\n{label}")
    for step in range(5):
        text, raw = gen(proc, model, messages)
        if show_prompt and step == 0:
            print("--- RENDERED NATIVE PROMPT (tail) ---\n" + text[-1100:] + "\n--- end prompt ---")
        p = parse(proc, raw)
        print(f"[step {step}] THINK : {(p['thinking'] or '(none)')[:320]}")
        print(f"[step {step}] CALLS : {p['tool_calls']}")
        print(f"[step {step}] ANSWER: {(p['content'] or '(none)')[:320]}")
        if not p["tool_calls"]:
            messages.append({"role": "assistant", "content": p["content"]})
            return p
        messages.append({"role": "assistant", "content": p.get("content", "") or "",
                         "tool_calls": [{"type": "function", "function": {"name": tc["name"],
                          "arguments": tc.get("arguments", {})}} for tc in p["tool_calls"]]})
        for tc in p["tool_calls"]:
            messages.append({"role": "tool", "name": tc["name"],
                             "content": CANNED.get(tc["name"], f"(no canned result for {tc['name']})")})
    return None


def main():
    proc, model = load()
    print("\n>>> Judge: does NATIVE thinking REASON about the tool result and answer GROUNDED?")
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"Here's my position (FEN {FEN}). What's the best move here?"}]
    run_turn(proc, model, msgs, "SCENARIO 1a — best move (does it call a tool + reason?)", show_prompt=True)
    msgs.append({"role": "user", "content": "why is that a good move?"})
    run_turn(proc, model, msgs, "SCENARIO 1b — WHY is it good? (the key test: grounded reason?)")

    msgs2 = [{"role": "system", "content": SYS},
             {"role": "user", "content": f"In this position (FEN {FEN}), is there a bishop on the board?"}]
    run_turn(proc, model, msgs2, "SCENARIO 2 — grounding: there is NO bishop (does it confabulate one?)")
    print("\n" + "=" * 80)
    print("VERDICT cues: (1) native THINK block present + on-topic? (2) calls the right tool?")
    print("(3) WHY answer cites the real line/mate, not a canned phrase? (4) says NO bishop?")


if __name__ == "__main__":
    main()
