from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

from chess_tool_demo import Board, Move, PIECE_VALUE, best_move, evaluate
from prepare_kaggle_sft import read_fens

FEATURES = ["material", "mobility", "capture", "promotion"]


def features_for_move(board: Board, move: Move) -> dict[str, float]:
    piece = board.piece_at(move.source) or "P"
    captured = board.piece_at(move.target)
    probe = board.clone()
    before_mobility = len(board.legal_moves())
    probe.move(move)
    after_mobility = len(probe.legal_moves())
    sign = 1.0 if board.turn == "w" else -1.0
    return {
        "material": sign * PIECE_VALUE[piece.upper()] / 900.0,
        "mobility": (after_mobility - before_mobility) / 40.0,
        "capture": (PIECE_VALUE[captured.upper()] / 900.0 if captured else 0.0),
        "promotion": 1.0 if move.promotion else 0.0,
    }


def score_move(weights: dict[str, float], features: dict[str, float]) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in features.items())


def expert_move(board: Board) -> str | None:
    result = best_move(board.clone())
    return result.get("best_move") if result["status"] == "ok" else None


def train_from_positions(fens: list[str], epochs: int, learning_rate: float) -> tuple[dict[str, float], list[dict]]:
    weights = {name: 0.0 for name in FEATURES}
    progress = []
    for epoch in range(1, epochs + 1):
        mistakes = 0
        examples = 0
        for fen in fens:
            board = Board.from_fen(fen)
            legal = board.legal_moves()
            target_uci = expert_move(board)
            if not legal or not target_uci:
                continue
            predicted = max(legal, key=lambda move: (score_move(weights, features_for_move(board, move)), move.uci()))
            if predicted.uci() != target_uci:
                mistakes += 1
                target = Move.parse(target_uci)
                target_features = features_for_move(board, target)
                predicted_features = features_for_move(board, predicted)
                for name in FEATURES:
                    weights[name] += learning_rate * (target_features[name] - predicted_features[name])
            examples += 1
        progress.append({"epoch": epoch, "examples": examples, "mistakes": mistakes, "accuracy": 1 - mistakes / examples if examples else 0.0, "weights": dict(weights)})
    return weights, progress


def trained_move(board: Board, weights: dict[str, float]) -> Move | None:
    legal = board.legal_moves()
    if not legal:
        return None
    return max(legal, key=lambda move: (score_move(weights, features_for_move(board, move)), move.uci()))


def weak_bot_move(board: Board, rng: random.Random) -> Move | None:
    legal = board.legal_moves()
    if not legal:
        return None
    captures = [move for move in legal if board.piece_at(move.target)]
    if captures and rng.random() < 0.65:
        return rng.choice(captures)
    return rng.choice(legal)


def play(weights: dict[str, float], game: int, engine_color: str, max_plies: int) -> dict:
    board = Board.from_fen()
    rng = random.Random(5000 + game)
    plies = 0
    for _ in range(max_plies):
        move = trained_move(board, weights) if board.turn == engine_color else weak_bot_move(board, rng)
        if move is None:
            break
        result = board.move(move)
        if result["status"] != "ok":
            break
        plies += 1
    final_eval = evaluate(board)
    score = final_eval["score_cp_white"]
    if abs(score) < 80:
        outcome = "drawish"
    else:
        outcome = "engine_win" if (score > 0) == (engine_color == "w") else "engine_loss"
    return {"game": game, "engine_color": engine_color, "plies": plies, "outcome": outcome, "final_eval": final_eval}


def match_suite(weights: dict[str, float], games: int, max_plies: int) -> dict:
    rows = [play(weights, index + 1, "w" if index % 2 == 0 else "b", max_plies) for index in range(games)]
    counts = defaultdict(int)
    for row in rows:
        counts[row["outcome"]] += 1
    return {
        "games": games,
        "max_plies": max_plies,
        "engine_wins": counts["engine_win"],
        "engine_losses": counts["engine_loss"],
        "drawish": counts["drawish"],
        "score_rate": (counts["engine_win"] + 0.5 * counts["drawish"]) / games,
        "rows": rows,
    }


def save_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a basic custom chess evaluator from FEN positions")
    parser.add_argument("--input", default="product_demo/sample_kaggle_fens.csv")
    parser.add_argument("--out-dir", default="product_demo/poc_models")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-plies", type=int, default=80)
    args = parser.parse_args()

    fens = [row.fen for row in read_fens(args.input, None)]
    weights, progress = train_from_positions(fens, args.epochs, args.learning_rate)
    matches = match_suite(weights, args.games, args.max_plies)
    payload = {"model_type": "basic-linear-chess-evaluator-v1", "features": FEATURES, "weights": weights, "training_positions": len(fens), "progress": progress, "match": matches}
    save_json(os.path.join(args.out_dir, "chess_engine_model.json"), payload)
    print(json.dumps({"out_dir": args.out_dir, "training_positions": len(fens), "final_training_accuracy": progress[-1]["accuracy"] if progress else 0.0, "match_score_rate": matches["score_rate"], "engine_wins": matches["engine_wins"], "engine_losses": matches["engine_losses"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
