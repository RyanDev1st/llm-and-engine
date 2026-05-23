"""Assemble the spec-v3 SFT set from the human slices.

Steps: load 10 slices (keep slice/slice C, drop the colder dup) -> clean
(artifact strip + canonical system prompt + warm slice-C errors) -> author
slice D with real Stockfish scores -> validate (schema, grammar incl. free-text
ask_chessbot, mode-2 discipline, routing) -> exact-dedup -> stratified 90/10
split -> write train/val jsonl + findings report.

Run:  python -m llm_dataset.build.assemble
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "llm"))

from llm_dataset.build.clean import clean_record  # noqa: E402
from llm_dataset.build import slice_d, slice_jk  # noqa: E402
from llm_dataset.contracts.schemas import validate_record_shape  # noqa: E402
from llm_dataset.contracts.tool_grammar import parse_tool_name  # noqa: E402

DATA = ROOT / "data" / "sft"
SF = ROOT / "src/llm/runtime/stockfish/stockfish/stockfish-windows-x86-64-avx2.exe"
SLICES = {
    "A": DATA / "slices/slices/slice A.json", "E": DATA / "slices/slices/slice E.json",
    "F": DATA / "slices/slices/slice F.json", "G": DATA / "slices/slices/slice G.json",
    "B": DATA / "slice/slice/slice_B_385.json", "C": DATA / "slice/slice/slice_C_280.json",
    "H": DATA / "slice/slice/slice_H_210.json", "I": DATA / "slice/slice/slice_I_420.json",
    "J": DATA / "slice/slice/slice_J_280.json", "K": DATA / "slice/slice/slice_K_175.json",
}
# Board tools must NOT fire in J/K (negatives). ask_chessbot IS allowed in K
# (spec 6.1 K-1: chess-flavored knowledge routes to ask_chessbot). This resolves
# a contradiction in the spec: 7.4 says "zero <tool>" but 6.1-K-1 routes to a
# tool -- we follow 6.1's behavioural intent.
BOARD_TOOLS = {"move", "eval", "best_move", "review_move", "threats", "legal_moves", "undo", "list_pieces"}
_GRAMMAR = re.compile(r"^<tool>([a-z_]+)((?:\s+[a-z_]+=.+?)*)</tool>$")


def _grammar_ok(content: str) -> bool:
    """Spec grammar, but final arg value may carry free text (ask_chessbot query)."""
    return bool(_GRAMMAR.match(content.strip()))


def validate(rec: dict) -> list[str]:
    errs = [v.reason for v in validate_record_shape(rec)]
    msgs = rec.get("messages", [])
    tool_calls = board_calls = 0
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            c = m["content"].lstrip()
            if c.startswith("<tool>"):
                tool_calls += 1
                if not _grammar_ok(m["content"]):
                    errs.append(f"bad grammar: {m['content'][:60]}")
                if parse_tool_name(m["content"]) in BOARD_TOOLS:
                    board_calls += 1
            if i > 0 and msgs[i - 1]["role"] == "tool" and "<tool>" in m["content"]:
                errs.append("mode-2 tool leak")
    sl = rec.get("slice")
    if sl == "J" and tool_calls:
        errs.append("J must have zero tool calls")
    if sl == "K" and board_calls:
        errs.append("K must have zero board-tool calls")
    if sl not in {"J", "K"} and sl and tool_calls == 0:
        errs.append("A-I must have >=1 tool call")
    return errs


def make_score_fn(engine):
    import chess

    def score_fn(fen: str):
        board = chess.Board(fen)
        if not board.is_valid() or board.is_game_over():
            return ("invalid", None)
        info = engine.analyse(board, chess.engine.Limit(depth=15))
        s = info["score"].white()
        if s.is_mate():
            m = s.mate()
            return ("mate", ("white" if m > 0 else "black", abs(m)))
        return ("cp", s.score())

    return score_fn


def load_clean() -> tuple[list[dict], Counter]:
    rng = random.Random(13)
    recs, counts = [], Counter()
    for sl, fp in SLICES.items():
        for r in json.loads(fp.read_text(encoding="utf-8")):
            recs.append(clean_record(r, rng))
            counts[sl] += 1
    return recs, counts


def stratified_split(recs: list[dict], val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    by: dict[str, list[dict]] = {}
    for r in recs:
        by.setdefault(r["slice"], []).append(r)
    train, val = [], []
    for sl in sorted(by):
        items = by[sl][:]
        rng.shuffle(items)
        k = max(1, int(len(items) * val_ratio))
        val += items[:k]
        train += items[k:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main() -> None:
    import chess.engine
    recs, counts = load_clean()
    print(f"loaded+cleaned {len(recs)} human records {dict(counts)}", flush=True)

    engine = chess.engine.SimpleEngine.popen_uci(str(SF))
    try:
        d = slice_d.generate(make_score_fn(engine), count=315)
    finally:
        engine.quit()
    print(f"slice D authored: {len(d)}", flush=True)
    recs += [clean_record(r, random.Random(1)) for r in d]

    # Augment routing-negative slices J/K (human ones collapsed to ~15 distinct).
    jk_rng = random.Random(23)
    jk = slice_jk.generate_j(300, jk_rng) + slice_jk.generate_k(200, 130, jk_rng)
    recs += [clean_record(r, jk_rng) for r in jk]
    print(f"slice J/K authored (pre-dedup): {len(jk)}", flush=True)

    seen, deduped, dups, dropped = {}, [], 0, Counter()
    for r in recs:
        key = tuple((m["role"], m["content"].strip()) for m in r["messages"] if m["role"] != "system")
        if key in seen:
            dups += 1
            dropped[r["slice"]] += 1
            if seen[key] != r["slice"]:
                print(f"  WARN cross-slice dup: {r['slice']} == {seen[key]}", flush=True)
            continue
        seen[key] = r["slice"]
        deduped.append(r)
    print(f"exact-dedup removed {dups} {dict(dropped)}; kept {len(deduped)}", flush=True)

    bad = [(r["id"], validate(r)) for r in deduped if validate(r)]
    good = [r for r in deduped if not validate(r)]
    print(f"validation: {len(good)} ok, {len(bad)} rejected", flush=True)

    train, val = stratified_split(good)
    _write(DATA / "chess_assistant_v3_train.jsonl", train)
    _write(DATA / "chess_assistant_v3_val.jsonl", val)
    (DATA / "slice/slice/slice_D_315.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    _report(counts, len(d), dups, bad, train, val)
    print(f"WROTE train={len(train)} val={len(val)} total={len(train)+len(val)}", flush=True)


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _report(counts, nd, dups, bad, train, val) -> None:
    sc = Counter(r["slice"] for r in train + val)
    lines = [
        "Parent: docs/superpowers/specs/2026-05-23-chess-coach-sft-design.md", "",
        "# Dataset rebuild findings (v3)", "", "## Status", "Complete.", "",
        "## Scope", "Rebuilt chess_assistant_v3 train/val from human slices.", "",
        "## Evidence",
        f"- Human records cleaned: {sum(counts.values())} {dict(counts)}",
        "- Dropped duplicate `slices/slices/slice C.json` (colder tone); kept ex_C variant.",
        "- Stripped ' (x_N)' artifact from 1,750 user turns (slices B,C,H,I,J,K).",
        "- Warmed slice-C error narration; canonicalised system prompt (dash mojibake fixed).",
        f"- Authored slice D (implicit eval, real Stockfish depth-15 scores): {nd}.",
        f"- Exact-dedup removed: {dups}.",
        f"- Validation rejects: {len(bad)}.",
        f"- Final per-slice: {dict(sorted(sc.items()))}",
        f"- Split: train={len(train)} val={len(val)} total={len(train)+len(val)}.", "",
        "## Next", "1. Smoke train, then full 3-epoch QLoRA on gemma4_e2b.",
        "2. Routing-accuracy audit on val.",
    ]
    if bad:
        lines += ["", "## Rejected (sample)"] + [f"- {i}: {e}" for i, e in bad[:20]]
    (ROOT / "docs" / "2026-05-23-dataset-rebuild-findings.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
