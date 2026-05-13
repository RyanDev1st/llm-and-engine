from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass

from chess_tool_demo import Board, best_move, run_tool_turn


@dataclass(frozen=True)
class FenRow:
    fen: str
    label: str


def read_fens(path: str, limit: int | None) -> list[FenRow]:
    rows: list[FenRow] = []
    with open(path, newline="", encoding="utf-8") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        has_header = "fen" in first_line.lower()
        if has_header:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                fen = row.get("fen") or row.get("FEN") or row.get("position") or row.get("Position")
                if fen:
                    rows.append(FenRow(fen.strip(), (row.get("label") or row.get("Label") or f"row_{index}").strip()))
                if limit and len(rows) >= limit:
                    break
        else:
            reader = csv.reader(handle)
            for index, row in enumerate(reader):
                if row:
                    rows.append(FenRow(row[0].strip(), f"row_{index}"))
                if limit and len(rows) >= limit:
                    break
    return rows


def make_router_record(prompt: str, tool_name: str, arguments: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a FEN-blind chess coach router. Use tools for board-specific claims."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "tool_call": {"tool_name": tool_name, "arguments": arguments}},
        ]
    }


def make_narrator_record(tool_name: str, tool_result: dict, narration: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You narrate only validated chess tool evidence. Never mention internal FEN."},
            {"role": "tool", "name": tool_name, "content": tool_result},
            {"role": "assistant", "content": narration},
        ]
    }


def records_for_position(row: FenRow) -> list[dict]:
    board = Board.from_fen(row.fen)
    records: list[dict] = []
    eval_turn = run_tool_turn(board.clone(), "eval", {})
    records.append(make_router_record("Evaluate the current position.", "eval", {}))
    records.append(make_narrator_record("eval", eval_turn["tool_result"], eval_turn["narration"]))

    legal = board.legal_moves()
    if legal:
        move = legal[0].uci()
        review_turn = run_tool_turn(board.clone(), "review_move", {"uci": move})
        records.append(make_router_record(f"Was my move {move} good?", "review_move", {"uci": move}))
        records.append(make_narrator_record("review_move", review_turn["tool_result"], review_turn["narration"]))

    best = best_move(board.clone())
    if best["status"] == "ok":
        records.append(make_router_record("What move should I consider?", "best_move", {}))
        records.append(make_narrator_record("best_move", best, json.dumps(best, sort_keys=True)))
    return records


def convert(input_path: str, out_dir: str, limit: int | None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train_sft.jsonl")
    eval_path = os.path.join(out_dir, "eval_sft.jsonl")
    manifest_path = os.path.join(out_dir, "manifest.json")
    rows = read_fens(input_path, limit)
    accepted = 0
    rejected = 0
    train_count = 0
    eval_count = 0
    with open(train_path, "w", encoding="utf-8") as train, open(eval_path, "w", encoding="utf-8") as eval_file:
        for index, row in enumerate(rows):
            try:
                records = records_for_position(row)
            except (IndexError, KeyError, ValueError):
                rejected += 1
                continue
            accepted += 1
            target = eval_file if index % 5 == 0 else train
            for record in records:
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                if target is eval_file:
                    eval_count += 1
                else:
                    train_count += 1
    manifest = {
        "input_path": input_path,
        "accepted_positions": accepted,
        "rejected_positions": rejected,
        "train_records": train_count,
        "eval_records": eval_count,
        "fen_policy": "FEN read internally only; SFT user/model-visible content stays FEN-blind.",
        "oracle": "demo-material-mobility-v1",
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Kaggle-style FEN CSV into FEN-blind SFT JSONL")
    parser.add_argument("--input", default="product_demo/sample_kaggle_fens.csv")
    parser.add_argument("--out-dir", default="product_demo/training_data")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(convert(args.input, args.out_dir, args.limit), indent=2))


if __name__ == "__main__":
    main()
