from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from dataclasses import dataclass

from chess_tool_demo import Board, best_move, run_tool_turn

FEN_POLICY = "FEN read internally only; SFT user/model-visible content stays FEN-blind."
FEN_COLUMNS = ("fen", "FEN", "position", "Position")
LABEL_COLUMNS = ("label", "Label", "id", "Id")


@dataclass(frozen=True)
class FenRow:
    row_number: int
    fen: str
    label: str


@dataclass(frozen=True)
class LoadedRows:
    rows: list[FenRow]
    total_rows: int
    missing_fen_rows: int
    source_columns: list[str]


def input_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_value(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = row.get(name)
        if value:
            return value.strip()
    return None


def read_fens(path: str, limit: int | None) -> LoadedRows:
    rows: list[FenRow] = []
    total_rows = 0
    missing_fen_rows = 0
    source_columns: list[str] = []
    with open(path, newline="", encoding="utf-8") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        has_header = any(column.lower() == "fen" for column in first_line.replace(";", ",").split(",")) or "position" in first_line.lower()
        if has_header:
            reader = csv.DictReader(handle)
            source_columns = list(reader.fieldnames or [])
            for index, row in enumerate(reader, start=2):
                total_rows += 1
                fen = choose_value(row, FEN_COLUMNS)
                if not fen:
                    missing_fen_rows += 1
                    continue
                label = choose_value(row, LABEL_COLUMNS) or f"row_{index}"
                rows.append(FenRow(index, fen, label))
                if limit and len(rows) >= limit:
                    break
        else:
            reader = csv.reader(handle)
            source_columns = ["column_0"]
            for index, row in enumerate(reader, start=1):
                total_rows += 1
                if not row or not row[0].strip():
                    missing_fen_rows += 1
                    continue
                rows.append(FenRow(index, row[0].strip(), f"row_{index}"))
                if limit and len(rows) >= limit:
                    break
    return LoadedRows(rows, total_rows, missing_fen_rows, source_columns)


def make_router_record(prompt: str, tool_name: str, arguments: dict, internal_fen: str) -> dict:
    return {
        "internal_fen": internal_fen,
        "messages": [
            {"role": "system", "content": "You are a FEN-blind chess coach router. Use tools for board-specific claims."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "tool_call": {"tool_name": tool_name, "arguments": arguments}},
        ]
    }


def make_narrator_record(tool_name: str, tool_result: dict, narration: str, internal_fen: str) -> dict:
    return {
        "internal_fen": internal_fen,
        "messages": [
            {"role": "system", "content": "You narrate only validated chess tool evidence. Never mention internal FEN."},
            {"role": "tool", "name": tool_name, "content": tool_result},
            {"role": "assistant", "content": narration},
        ]
    }


def router_prompts(move: str | None) -> list[tuple[str, str, dict]]:
    prompts: list[tuple[str, str, dict]] = [
        ("Evaluate the current position.", "eval", {}),
        ("How is this board looking?", "eval", {}),
        ("What move should I consider?", "best_move", {}),
        ("Find a practical candidate move here.", "best_move", {}),
    ]
    if move:
        prompts.extend(
            [
                (f"Was my move {move} good?", "review_move", {"uci": move}),
                (f"Please review {move} in this position.", "review_move", {"uci": move}),
                (f"Is {move} legal and useful?", "review_move", {"uci": move}),
            ]
        )
    return prompts


def records_for_position(row: FenRow) -> list[dict]:
    board = Board.from_fen(row.fen)
    records: list[dict] = []
    eval_turn = run_tool_turn(board.clone(), "eval", {})
    records.append(make_narrator_record("eval", eval_turn["tool_result"], eval_turn["narration"], row.fen))

    legal = board.legal_moves()
    move = legal[0].uci() if legal else None
    if move:
        review_turn = run_tool_turn(board.clone(), "review_move", {"uci": move})
        records.append(make_narrator_record("review_move", review_turn["tool_result"], review_turn["narration"], row.fen))

    best = best_move(board.clone())
    if best["status"] == "ok":
        records.append(make_narrator_record("best_move", best, json.dumps(best, sort_keys=True), row.fen))

    for prompt, tool_name, arguments in router_prompts(move):
        records.append(make_router_record(prompt, tool_name, arguments, row.fen))
    return records


def visible_record_text(record: dict) -> str:
    visible = []
    for message in record.get("messages", []):
        if not isinstance(message, dict):
            continue
        if "content" in message:
            visible.append(str(message["content"]))
        tool_call = message.get("tool_call")
        if isinstance(tool_call, dict):
            visible.append(json.dumps(tool_call, sort_keys=True))
    return "\n".join(visible)


def record_contains_internal_fen(record: dict, fen: str) -> bool:
    return record.get("internal_fen") == fen


def fen_visible_tokens(fen: str) -> list[str]:
    parts = fen.split()
    tokens = [fen]
    if parts:
        tokens.append(parts[0])
    if len(parts) > 2 and parts[2] != "-" and len(parts[2]) > 1:
        tokens.append(parts[2])
    if len(parts) > 3 and parts[3] != "-":
        tokens.append(parts[3])
    return [token for token in tokens if token]


def leaks_fen(record: dict, fen: str) -> bool:
    visible = visible_record_text(record)
    return any(token in visible for token in fen_visible_tokens(fen))


def validate_rows(rows: list[FenRow]) -> tuple[list[FenRow], Counter[str]]:
    accepted: list[FenRow] = []
    reasons: Counter[str] = Counter()
    seen: set[str] = set()
    for row in rows:
        if not row.fen:
            reasons["blank_fen"] += 1
            continue
        if row.fen in seen:
            reasons["duplicate_fen"] += 1
            continue
        try:
            Board.from_fen(row.fen)
        except ValueError:
            reasons["invalid_fen"] += 1
            continue
        seen.add(row.fen)
        accepted.append(row)
    return accepted, reasons


def production_mode(input_path: str, data_mode: str, accepted_positions: int) -> str:
    if data_mode != "auto":
        return data_mode
    if accepted_positions >= 300000:
        return "real_kaggle"
    return "sample"


def split_rows(rows: list[FenRow], eval_ratio: float, seed: int) -> tuple[set[int], set[int]]:
    indexes = list(range(len(rows)))
    random.Random(seed).shuffle(indexes)
    eval_size = max(1, int(len(indexes) * eval_ratio)) if len(indexes) > 1 else len(indexes)
    eval_indexes = set(indexes[:eval_size])
    train_indexes = set(indexes[eval_size:])
    return train_indexes, eval_indexes


def convert(input_path: str, out_dir: str, limit: int | None, split_seed: int, eval_ratio: float, data_mode: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train_sft.jsonl")
    eval_path = os.path.join(out_dir, "eval_sft.jsonl")
    manifest_path = os.path.join(out_dir, "manifest.json")
    loaded = read_fens(input_path, limit)
    accepted_rows, rejection_reasons = validate_rows(loaded.rows)
    rejection_reasons["missing_fen"] += loaded.missing_fen_rows
    train_indexes, eval_indexes = split_rows(accepted_rows, eval_ratio, split_seed)
    train_count = 0
    eval_count = 0
    leakage_count = 0
    record_errors: Counter[str] = Counter()
    with open(train_path, "w", encoding="utf-8") as train, open(eval_path, "w", encoding="utf-8") as eval_file:
        for index, row in enumerate(accepted_rows):
            try:
                records = records_for_position(row)
            except (IndexError, KeyError, ValueError):
                record_errors["record_generation_failed"] += 1
                continue
            target = eval_file if index in eval_indexes else train
            for record in records:
                if leaks_fen(record, row.fen) or not record_contains_internal_fen(record, row.fen):
                    leakage_count += 1
                    continue
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                if target is eval_file:
                    eval_count += 1
                else:
                    train_count += 1
    mode = production_mode(input_path, data_mode, len(accepted_rows))
    rejected_positions = sum(rejection_reasons.values()) + sum(record_errors.values())
    manifest = {
        "input_path": input_path,
        "input_sha256": input_sha256(input_path),
        "source_columns": loaded.source_columns,
        "source_dataset": "kaggle_350k_fen_csv" if mode == "real_kaggle" else "sample_or_subset_fen_csv",
        "data_mode": mode,
        "input_rows_total": loaded.total_rows,
        "accepted_positions": len(accepted_rows),
        "rejected_positions": rejected_positions,
        "rejection_reasons": dict(sorted((rejection_reasons + record_errors).items())),
        "duplicate_positions": rejection_reasons.get("duplicate_fen", 0),
        "train_records": train_count,
        "eval_records": eval_count,
        "split_seed": split_seed,
        "eval_ratio": eval_ratio,
        "fen_policy": FEN_POLICY,
        "fen_leakage_audit_fields": ["full_fen", "piece_placement", "castling", "en_passant"],
        "fen_leakage_records_dropped": leakage_count,
        "fen_internal_position_records": train_count + eval_count,
        "fen_leakage_audit_passed": leakage_count == 0,
        "oracle": "python-chess-static-eval-v1",
        "is_production_valid": mode == "real_kaggle" and len(accepted_rows) >= 300000 and leakage_count == 0 and eval_count > 0 and train_count > 0,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Kaggle-style FEN CSV into FEN-blind SFT JSONL")
    parser.add_argument("--input", default="product_demo/sample_kaggle_fens.csv")
    parser.add_argument("--out-dir", default="product_demo/training_data")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split-seed", type=int, default=20260513)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--data-mode", choices=["auto", "sample", "real_kaggle"], default="auto")
    args = parser.parse_args()
    print(json.dumps(convert(args.input, args.out_dir, args.limit, args.split_seed, args.eval_ratio, args.data_mode), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
