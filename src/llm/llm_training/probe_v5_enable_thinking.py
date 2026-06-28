"""PROBE (v5 one-shot insurance): does a v5-TRAINED E4B adapter still emit NATIVE CoT at
serve (enable_thinking=True), ROUTE + GROUND, and stay DIRECT in fast mode?

The single unvalidated bet of the v5 retrain: training renders fast/think/auto with
enable_thinking=False and NO thinking content (only plan rows train the channel), betting the
FROZEN base's native CoT re-activates at serve via enable_thinking=True. Phase-0
(`probe_native_thinking`) proved the BASE thinks + grounds. The OPEN risk is whether the LoRA
SUPPRESSES native CoT (it's trained to go direct). A base-model probe cannot answer that — only
a TRAINED adapter can. So: micro-train a tiny real-config v5 LoRA, then run this.

RUN (Kaggle, AFTER a micro-train writes runs/v5_probe/checkpoint):
  PROBE_BASE=<repo>/src/llm/models/gemma4_e4b \
  PROBE_ADAPTER=<repo>/runs/v5_probe/checkpoint \
    PYTHONPATH=src/llm python -m llm_training.probe_v5_enable_thinking

Greedy decode, canned-CORRECT tool results (so the only question is whether the adapter's
reasoning USES them). Prints a PASS/FAIL table + an overall GREEN / AMBER / RED verdict.
"""
from __future__ import annotations

import os
import re

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

from .probe_native_thinking import (CHESS_SYS, CHESS_TOOLS, MATE, MATE_CANNED, parse)

BASE = os.environ.get("PROBE_BASE", "unsloth/gemma-4-E4B-it")
ADAPTER = os.environ.get("PROBE_ADAPTER", "")
KNOWN_TOOLS = {"board_state", "best_move", "eval", "threats"}
_SOUP = (r"<thinking>", r"function\s+name\s*=", r"<start_of_turn>", r"```tool")


def load():
    print(f"loading BASE (4-bit): {BASE}", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, quantization_config=quant, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": 0})
    if ADAPTER:
        from peft import PeftModel
        print(f"attaching ADAPTER: {ADAPTER}", flush=True)
        model = PeftModel.from_pretrained(model, ADAPTER)
    else:
        print("WARNING: no PROBE_ADAPTER set — probing the BASE (this only re-runs Phase-0).", flush=True)
    model.eval()
    return tok, model


def render(tok, messages, tools, thinking: bool) -> str:
    for kw in ({"enable_thinking": thinking}, {}):
        try:
            return tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                           tokenize=False, **kw)
        except (TypeError, ValueError):
            continue
    raise RuntimeError("apply_chat_template failed for every kwarg combo")


def gen(tok, model, messages, tools, thinking: bool) -> str:
    text = render(tok, messages, tools, thinking)
    enc = tok(text, return_tensors="pt").to(model.device)
    n = enc["input_ids"].shape[1]
    out = model.generate(**enc, max_new_tokens=512, do_sample=False)
    return tok.decode(out[0][n:], skip_special_tokens=False)


def run_turn(tok, model, tools, canned, messages, thinking: bool) -> dict:
    """One model turn (auto-feeds canned tool results for up to 4 calls). Returns the LAST
    step's parsed {thinking, tool_calls, content} plus a union of every tool called."""
    called, last = [], {"thinking": "", "tool_calls": [], "content": ""}
    for _ in range(4):
        raw = gen(tok, model, messages, tools, thinking)
        p = parse(None, raw)
        last = p
        if not p["tool_calls"]:
            messages.append({"role": "assistant", "content": p["content"]})
            break
        called += [tc["name"] for tc in p["tool_calls"]]
        messages.append({"role": "assistant", "content": p.get("content", "") or "",
                         "tool_calls": [{"type": "function", "function": {
                             "name": tc["name"], "arguments": tc.get("arguments", {})}}
                             for tc in p["tool_calls"]]})
        for tc in p["tool_calls"]:
            messages.append({"role": "tool", "name": tc["name"],
                             "content": canned.get(tc["name"], f"(no canned result for {tc['name']})")})
    last["_called"] = called
    return last


def _soup(text: str) -> bool:
    return any(re.search(p, text or "", re.I) for p in _SOUP)


def main() -> None:
    tok, model = load()
    res = {}

    # --- THINK mode (enable_thinking=True): expect native CoT + routing + grounded answer ---
    msgs = [{"role": "system", "content": CHESS_SYS}]
    think_blocks, answers, called = [], [], []
    for user in (f"Here's my position (FEN {MATE}). What's the best move here?",
                 "why is that a good move?",
                 "is there a bishop on the board?"):
        msgs.append({"role": "user", "content": user})
        p = run_turn(tok, model, CHESS_TOOLS, MATE_CANNED, msgs, thinking=True)
        think_blocks.append(p["thinking"] or "")
        answers.append((p["content"] or "").lower())
        called += p["_called"]
        print(f"\n[THINK] U: {user}\n  THINK : {(p['thinking'] or '(none)')[:240]}"
              f"\n  CALLS : {p['_called']}\n  ANSWER: {(p['content'] or '(none)')[:240]}")

    res["NATIVE_COT"] = any(len(t.strip()) > 20 for t in think_blocks)
    res["ROUTES"] = any(c in KNOWN_TOOLS for c in called)
    why = answers[1] if len(answers) > 1 else ""
    bishop = answers[2] if len(answers) > 2 else ""
    res["GROUNDED"] = (("re8" in why or "mate" in why or "back" in why)
                       and ("no" in bishop and "yes" not in bishop[:40]))
    res["NO_SOUP"] = not any(_soup(a) for a in answers)

    # --- FAST mode (enable_thinking=False): expect NO CoT block ---
    fmsgs = [{"role": "system", "content": CHESS_SYS},
             {"role": "user", "content": f"Position FEN {MATE}. Best move? One line."}]
    fp = run_turn(tok, model, CHESS_TOOLS, MATE_CANNED, fmsgs, thinking=False)
    print(f"\n[FAST]  THINK: {(fp['thinking'] or '(none)')[:160]}\n  ANSWER: {(fp['content'] or '(none)')[:160]}")
    res["FAST_DIRECT"] = len((fp["thinking"] or "").strip()) <= 20

    crit = ["NATIVE_COT", "ROUTES", "GROUNDED", "FAST_DIRECT", "NO_SOUP"]
    print("\n" + "=" * 60 + "\nv5 enable_thinking PROBE — verdict")
    for k in crit:
        print(f"  {'PASS' if res.get(k) else 'FAIL'}  {k}")
    core = res.get("NATIVE_COT") and res.get("ROUTES") and res.get("GROUNDED") and res.get("NO_SOUP")
    if core and res.get("FAST_DIRECT"):
        verdict = "GREEN  — native CoT survived the LoRA; routes + grounds; fast stays direct. Launch the full run."
    elif core:
        verdict = "AMBER  — core OK but FAST_DIRECT failed (fast still thinks). Launchable; check the fast-mode signal in the live eval."
    elif res.get("NATIVE_COT"):
        verdict = "AMBER  — CoT survives but routing/grounding weak at this micro step count; the full 1000-step run likely fixes it (watch step-100 eval)."
    else:
        verdict = "RED    — the LoRA SUPPRESSED native CoT. Do NOT spend 11h: keep <think> traces in v5 thinking rows (mask from loss) before the full run."
    print("\nVERDICT:", verdict)
    print("(SEQ-1920 fit is reported by the micro-train cell: it OOMs there if 1920 doesn't fit a T4.)")


if __name__ == "__main__":
    main()
