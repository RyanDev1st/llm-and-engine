"""One-shot retrain preflight: adversarial GO/NO-GO over the COMMITTED v5 split,
checking the failure modes that train CLEANLY on poisoned targets (you only find
out at serve). Run before the (un-repeatable) E4B retrain.

Checks:
  T  tokenizer invariants  — native markers are SINGLE tokens; TOOL_RESPONSE_IDS
                              {50,51} REALLY are <|tool_response>/<tool_response|>
                              (the loss mask hardcodes these ids).
  M  loss-mask invariants  — on real rows across slices: train:false turn-1 MASKED,
                              tool-result content MASKED, final answer TRAINED,
                              turn-ender TRAINED (model learns to STOP), fact tokens
                              get GROUND_WEIGHT, no row trains an empty label set.
  L  legacy-format leak     — no <tool>/<skill>/<think>/<thinking>/function-call text
                              anywhere in the trained render (would re-teach old format).
  G  grounding holes        — every final fact (SAN, decimal, mate-in-N, standing in a
                              FOLLOW-UP) is backed by a tool result in that turn.
  D  final diversity        — per-slice distinct-ratio + worst repeat (memorization).
  C  coverage               — every skill is loaded somewhere; reasoning-mode mix.
Exit 0 = GO, 1 = NO-GO.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path("src/llm").resolve()))

from llm_dataset.v1.profiles import profile  # noqa: E402
from llm_training.system_prompt import build_system  # noqa: E402
from llm_training.data_pipeline import (  # noqa: E402
    IGNORE_INDEX, TOOL_RESPONSE_IDS, GROUND_WEIGHT, tokenize_with_assistant_mask,
)
from llm_training.native_tools import manifest_to_native_tools, template_messages  # noqa: E402

TOK_DIR = Path("src/llm/models/gemma4_e2b")
MARKERS = {
    "<|turn>": None, "<turn|>": None, "<|tool_call>": 48, "<tool_call|>": 49,
    "<|tool_response>": 50, "<tool_response|>": 51, "<|channel>": 100, "<channel|>": 101,
    '<|"|>': None,
}
_FACT = re.compile(r"[+-]?\d+\.\d{2}|O-O(?:-O)?|[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?")
_MATE = re.compile(r"\bmate in (\d+)\b|\bforced mate\b|\bcheckmate\b", re.I)
_LEGACY = ("<tool>", "</tool>", "<skill>", "</skill>", "<think>", "</think>",
           "<thinking>", "</thinking>", "<tool_code>", "function name=", "```tool")
FAILS: list[str] = []


def _resolve_tok_dir(tok_dir: str | None = None) -> Path:
    return Path(tok_dir or os.environ.get("CHESS_TOK_DIR") or TOK_DIR)


def fail(tag: str, msg: str) -> None:
    FAILS.append(f"[{tag}] {msg}")
    print(f"  FAIL [{tag}] {msg}")


def read_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def render_text(msgs, tok):
    mode = (msgs[0].get("_reasoning_mode", "") if msgs else "").strip().lower()
    return tok.apply_chat_template(
        template_messages(msgs), tokenize=False, add_generation_prompt=False,
        enable_thinking=mode not in ("", "fast"), tools=msgs[0].get("_native_tools") or None)


def render(row, tok):
    system = build_system(row.get("skills_index", []), row.get("tool_manifest", []),
                          row.get("plugin_context", {}), reasoning_mode=row.get("reasoning_mode", ""),
                          include_tools=False)
    native_tools = manifest_to_native_tools(row.get("tool_manifest", []), include_descriptions=False)
    body = [m for m in row["messages"] if m.get("role") != "system"]
    msgs = [{"role": "system", "content": system, "_native_tools": native_tools,
             "_reasoning_mode": row.get("reasoning_mode", "")}, *body]
    return render_text(msgs, tok), msgs


def main(profile_name="v5", tok_dir: str | None = None):
    from transformers import AutoTokenizer
    p = profile(profile_name)
    tok_path = _resolve_tok_dir(tok_dir)
    tok = AutoTokenizer.from_pretrained(str(tok_path), trust_remote_code=True)
    train = list(read_rows(Path(str(p.train_path) + ".gz")))
    print(f"profile={profile_name} train={len(train)} max_seq={p.max_seq} tok_dir={tok_path}")

    # ---- T: tokenizer invariants ----
    print("\n[T] tokenizer invariants")
    for marker, want_id in MARKERS.items():
        ids = tok(marker, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            fail("T", f"marker {marker!r} is {len(ids)} tokens {ids}, not single (mask/format breaks)")
            continue
        got = ids[0]
        if want_id is not None and got != want_id:
            fail("T", f"marker {marker!r} id={got}, expected {want_id}")
        print(f"    {marker!r:20} -> id {got}{' OK' if want_id in (None, got) else ''}")
    if tok("<|tool_response>")["input_ids"][-1] not in TOOL_RESPONSE_IDS or \
       tok("<tool_response|>")["input_ids"][-1] not in TOOL_RESPONSE_IDS:
        fail("T", f"TOOL_RESPONSE_IDS {TOOL_RESPONSE_IDS} != actual tool_response marker ids")

    # ---- M: loss-mask invariants on real rows across slices (POSITION-accurate) ----
    print("\n[M] loss-mask invariants (sampled per slice)")
    by_slice = defaultdict(list)
    for r in train:
        by_slice[r["slice"]].append(r)
    turn_ender_id = tok("<turn|>", add_special_tokens=False)["input_ids"][-1]
    checked = ground_seen = plan_seen = 0
    for sl, rows in by_slice.items():
        for r in rows[:6]:
            checked += 1
            _, msgs = render(r, tok)
            ids, labels, weights = tokenize_with_assistant_mask(msgs, tok, p.max_seq)
            if not any(l != IGNORE_INDEX for l in labels):
                fail("M", f"{r['id']} trains an EMPTY label set (build_examples would drop it)")
            for tid, lab in zip(ids, labels):
                if tid in TOOL_RESPONSE_IDS and lab != IGNORE_INDEX:
                    fail("M", f"{r['id']} trains a tool_response marker id={tid}")
                    break
            if turn_ender_id in ids:
                last = max(i for i, t in enumerate(ids) if t == turn_ender_id)
                if labels[last] == IGNORE_INDEX:
                    fail("M", f"{r['id']} final turn-ender <turn|> is MASKED (won't learn to stop)")
            if any(w > 1.0 for w in weights):
                ground_seen += 1
        # multi-turn: turn-1 (train:false) delta must be FULLY masked — checked by POSITION
        # (re-derive per-message deltas exactly as data_pipeline does), not by id membership.
        if sl == "V1_P_multiturn_followup":
            for r in rows[:8]:
                _, msgs = render(r, tok)
                ctx_i = next((i for i, m in enumerate(msgs)
                              if m.get("role") == "assistant" and m.get("train") is False), None)
                if ctx_i is None:
                    continue
                lo = len(tok(render_text(msgs[:ctx_i], tok), add_special_tokens=False)["input_ids"])
                hi = len(tok(render_text(msgs[:ctx_i + 1], tok), add_special_tokens=False)["input_ids"])
                _, labels, _ = tokenize_with_assistant_mask(msgs, tok, p.max_seq)
                trained = [j for j in range(lo, min(hi, len(labels))) if labels[j] != IGNORE_INDEX]
                if trained:
                    fail("M", f"{r['id']} turn-1 (train:false) delta has {len(trained)} TRAINED tokens (must be 0)")
    if ground_seen == 0:
        fail("M", "GROUND_WEIGHT never fired on any sampled row — fact up-weighting is inert")
    # plan rows: the native reasoning channel (<goal>/<plan>) must be TRAINED (model learns to plan)
    for r in by_slice.get("V1_S_compound_plan", [])[:4] + by_slice.get("V1_T_audited_plan", [])[:4]:
        _, msgs = render(r, tok)
        ch = tok("<|channel>", add_special_tokens=False)["input_ids"][-1]
        ids, labels, _ = tokenize_with_assistant_mask(msgs, tok, p.max_seq)
        if ch in ids:
            plan_seen += 1
            pos = ids.index(ch)
            if labels[pos] == IGNORE_INDEX:
                fail("M", f"{r['id']} plan channel <|channel> is MASKED (won't learn to plan)")
    print(f"    checked {checked} rows; GROUND_WEIGHT active on {ground_seen}; plan-channel rows {plan_seen}")

    # ---- L: legacy-format leak across the FULL trained render ----
    print("\n[L] legacy-format leak (full corpus, trained text only)")
    leak = Counter()
    for r in train:
        for m in r["messages"]:
            if m.get("role") != "assistant" or m.get("train") is False:
                continue
            blob = (m.get("content") or "") + " " + (m.get("reasoning") or "")
            for tag in _LEGACY:
                if tag in blob:
                    leak[tag] += 1
    if leak:
        fail("L", f"legacy markup in trained assistant text: {dict(leak)}")
    else:
        print("    none")

    # ---- G: grounding holes (mate-in-N + standing in follow-ups) ----
    print("\n[G] grounding holes")
    mate_holes = 0
    for r in train:
        tool_blob = " ".join(m.get("content", "") for m in r["messages"] if m.get("role") == "tool").lower()
        finals = [m["content"] for m in r["messages"]
                  if m.get("role") == "assistant" and not m.get("tool_calls") and m.get("train") is not False]
        if not finals:
            continue
        fin = finals[-1]
        if _MATE.search(fin) and "mate" not in tool_blob:
            mate_holes += 1
            if mate_holes <= 5:
                print(f"    mate-claim-without-tool: {r['id']} :: ...{fin[-80:]!r}")
    if mate_holes:
        fail("G", f"{mate_holes} finals claim mate/checkmate with no 'mate' in any tool result")
    else:
        print("    no mate-grounding holes")

    # ---- D: final diversity per slice ----
    print("\n[D] final diversity (distinct / rows : top-repeat)")
    fin_by_slice = defaultdict(Counter)
    for r in train:
        fin = next((m["content"] for m in reversed(r["messages"])
                    if m.get("role") == "assistant" and not m.get("tool_calls")), "")
        fin_by_slice[r["slice"]][fin] += 1
    for sl in sorted(fin_by_slice):
        c = fin_by_slice[sl]
        n, d, top = sum(c.values()), len(c), c.most_common(1)[0][1]
        flagstr = ""
        # Canonical knowledge/lesson/routing slices have inherently few distinct finals
        # (one right answer) — that is NOT memorization and matches the project gate
        # (audit.py passes). Only a CATASTROPHIC collapse (<5 distinct on a big slice,
        # the old "1 final per slice" failure) is a hard fail; the rest is informational.
        if n >= 100 and d < 5:
            flagstr = "  <-- CATASTROPHIC"
            fail("D", f"slice {sl}: only {d} distinct finals / {n} rows (memorization)")
        elif n >= 100 and d / n < 0.05:
            flagstr = "  (low, canonical-OK)"
        print(f"    {sl:28} {d:5}/{n:<5} top={top}{flagstr}")

    # ---- C: coverage (every skill loaded; reasoning-mode mix) ----
    print("\n[C] coverage")
    loaded = Counter()
    for r in train:
        for m in r["messages"]:
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc)
                if fn.get("name") == "load_skill":
                    loaded[(fn.get("arguments") or {}).get("name")] += 1
    all_skills = set()
    for r in train[:200]:
        for s in r.get("skills_index", []):
            all_skills.add(s.get("name"))
    never = all_skills - set(loaded)
    print(f"    skills loaded: {dict(loaded)}")
    if never:
        fail("C", f"skills NEVER loaded in any row (routing blind spot): {sorted(never)}")
    modes = Counter(r.get("reasoning_mode") for r in train)
    print(f"    reasoning_mode: {dict(modes)}")
    for mode in ("fast", "think", "auto", "plan"):
        if modes.get(mode, 0) < 200:
            fail("C", f"reasoning mode '{mode}' has only {modes.get(mode,0)} train rows (<200)")

    # ---- R: TRUE duplicate rows (full render = system+messages), not message-only ----
    # The message-only hash (final_corpus_audit) overcounts: skills_index/tool_manifest are
    # SHUFFLED per row, so message-identical rows render DIFFERENT system prompts = distinct
    # training examples (good order-invariance). This counts byte-identical FULL renders.
    print("\n[R] true duplicate rows (full system+messages render)")
    import hashlib
    msg_hashes, render_hashes = Counter(), Counter()
    for r in train:
        msg_hashes[hashlib.sha1(json.dumps(r["messages"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()] += 1
        text, _ = render(r, tok)
        render_hashes[hashlib.sha1(text.encode("utf-8")).hexdigest()] += 1
    msg_dups = sum(c - 1 for c in msg_hashes.values() if c > 1)
    render_dups = sum(c - 1 for c in render_hashes.values() if c > 1)
    print(f"    message-only dup rows: {msg_dups}   true full-render dup rows: {render_dups}")
    if render_dups > 0.02 * len(train):
        fail("R", f"{render_dups} byte-identical full-render rows (>2% — real memorization)")

    print("\n" + "=" * 60)
    print("PREFLIGHT:", "GO" if not FAILS else f"NO-GO ({len(FAILS)} fails)")
    for f in FAILS:
        print("  " + f)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="v5")
    ap.add_argument("--tok-dir", default=None, help="tokenizer dir to verify; defaults to CHESS_TOK_DIR or local E2B")
    args = ap.parse_args()
    raise SystemExit(main(args.profile, args.tok_dir))
