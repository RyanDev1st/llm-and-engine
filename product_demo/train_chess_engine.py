from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict

import chess

from prepare_kaggle_sft import read_fens

FEATURES = [
    "bias",
    "material_delta",
    "piece_square_delta",
    "mobility_delta",
    "capture_value",
    "gives_check",
    "promotion_value",
    "castle",
    "center_control_delta",
]

PIECE_VALUES = {
    chess.PAWN: 100.0,
    chess.KNIGHT: 320.0,
    chess.BISHOP: 330.0,
    chess.ROOK: 500.0,
    chess.QUEEN: 900.0,
    chess.KING: 0.0,
}

CENTER = {chess.D4, chess.E4, chess.D5, chess.E5}


def board_from_fen(fen: str) -> chess.Board:
    return chess.Board(fen)


def material(board: chess.Board) -> float:
    score = 0.0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return score


def piece_square(board: chess.Board) -> float:
    score = 0.0
    for square, piece in board.piece_map().items():
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        centrality = 3.5 - (abs(file_index - 3.5) + abs(rank_index - 3.5)) / 2.0
        value = centrality * (0.08 if piece.piece_type == chess.PAWN else 0.16) * PIECE_VALUES[piece.piece_type]
        score += value if piece.color == chess.WHITE else -value
    return score


def legal_mobility(board: chess.Board) -> float:
    turn = board.turn
    white_moves = len(list(board.legal_moves)) if turn == chess.WHITE else None
    board.turn = not turn
    black_moves = len(list(board.legal_moves))
    board.turn = turn
    if white_moves is None:
        black_moves = len(list(board.legal_moves))
        board.turn = chess.WHITE
        white_moves = len(list(board.legal_moves))
        board.turn = turn
    return float(white_moves - black_moves)


def center_control(board: chess.Board) -> float:
    white = sum(1 for sq in CENTER for _ in board.attackers(chess.WHITE, sq))
    black = sum(1 for sq in CENTER for _ in board.attackers(chess.BLACK, sq))
    return float(white - black)


def static_eval(board: chess.Board) -> float:
    if board.is_checkmate():
        return -100000.0 if board.turn == chess.WHITE else 100000.0
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0
    return material(board) + piece_square(board) + 3.0 * legal_mobility(board) + 8.0 * center_control(board)


def move_features(board: chess.Board, move: chess.Move) -> dict[str, float]:
    before_eval = static_eval(board)
    before_mobility = len(list(board.legal_moves))
    before_center = center_control(board)
    captured = board.piece_at(move.to_square)
    promotion_value = PIECE_VALUES.get(move.promotion, 0.0) if move.promotion else 0.0
    is_castle = board.is_castling(move)
    color_sign = 1.0 if board.turn == chess.WHITE else -1.0
    probe = board.copy(stack=False)
    probe.push(move)
    after_eval = static_eval(probe)
    after_mobility = len(list(probe.legal_moves))
    after_center = center_control(probe)
    return {
        "bias": 1.0,
        "material_delta": color_sign * (after_eval - before_eval) / 1000.0,
        "piece_square_delta": color_sign * (piece_square(probe) - piece_square(board)) / 1000.0,
        "mobility_delta": (after_mobility - before_mobility) / 80.0,
        "capture_value": (PIECE_VALUES[captured.piece_type] / 900.0 if captured else 0.0),
        "gives_check": 1.0 if probe.is_check() else 0.0,
        "promotion_value": promotion_value / 900.0,
        "castle": 1.0 if is_castle else 0.0,
        "center_control_delta": color_sign * (after_center - before_center) / 16.0,
    }


def score_move(weights: dict[str, float], features: dict[str, float]) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in features.items())


def expert_move(board: chess.Board) -> chess.Move | None:
    legal = list(board.legal_moves)
    if not legal:
        return None
    color_sign = 1.0 if board.turn == chess.WHITE else -1.0
    return max(legal, key=lambda move: (color_sign * eval_after(board, move), move.uci()))


def eval_after(board: chess.Board, move: chess.Move) -> float:
    probe = board.copy(stack=False)
    probe.push(move)
    return static_eval(probe)


def predict_move(board: chess.Board, weights: dict[str, float]) -> chess.Move | None:
    legal = list(board.legal_moves)
    if not legal:
        return None
    return max(legal, key=lambda move: (score_move(weights, move_features(board, move)), move.uci()))


def train_from_positions(fens: list[str], epochs: int, learning_rate: float, seed: int) -> tuple[dict[str, float], list[dict]]:
    rng = random.Random(seed)
    weights = {name: 0.0 for name in FEATURES}
    progress = []
    for epoch in range(1, epochs + 1):
        shuffled = list(fens)
        rng.shuffle(shuffled)
        mistakes = 0
        examples = 0
        for fen in shuffled:
            board = board_from_fen(fen)
            target = expert_move(board)
            predicted = predict_move(board, weights)
            if target is None or predicted is None:
                continue
            if predicted != target:
                mistakes += 1
                target_features = move_features(board, target)
                predicted_features = move_features(board, predicted)
                for name in FEATURES:
                    weights[name] += learning_rate * (target_features[name] - predicted_features[name])
            examples += 1
        progress.append({"epoch": epoch, "examples": examples, "mistakes": mistakes, "accuracy": 1 - mistakes / examples if examples else 0.0, "weights": dict(weights)})
    return weights, progress


def evaluate_positions(fens: list[str], weights: dict[str, float]) -> dict:
    rows = []
    correct = 0
    legal_predictions = 0
    for fen in fens:
        board = board_from_fen(fen)
        predicted = predict_move(board, weights)
        target = expert_move(board)
        if predicted is not None:
            legal_predictions += 1
        ok = predicted == target and predicted is not None
        correct += int(ok)
        rows.append({"predicted": predicted.uci() if predicted else None, "expert": target.uci() if target else None, "ok": ok})
    return {
        "positions": len(rows),
        "legal_prediction_rate": legal_predictions / len(rows) if rows else 0.0,
        "top1_accuracy": correct / len(rows) if rows else 0.0,
        "rows": rows[:50],
    }


def engine_move(board: chess.Board, weights: dict[str, float]) -> chess.Move | None:
    return predict_move(board, weights)


def baseline_move(board: chess.Board) -> chess.Move | None:
    legal = list(board.legal_moves)
    if not legal:
        return None
    color_sign = 1.0 if board.turn == chess.WHITE else -1.0
    return max(legal, key=lambda move: (color_sign * eval_after(board, move), move.uci()))


def play(weights: dict[str, float], game: int, engine_color: chess.Color, max_plies: int) -> dict:
    board = chess.Board()
    plies = 0
    termination = "max_plies"
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            termination = board.outcome(claim_draw=True).termination.name.lower()
            break
        move = engine_move(board, weights) if board.turn == engine_color else baseline_move(board)
        if move is None:
            termination = "no_legal_moves"
            break
        board.push(move)
        plies += 1
    score = static_eval(board)
    if board.is_checkmate():
        outcome = "engine_win" if board.turn != engine_color else "engine_loss"
    elif abs(score) < 80:
        outcome = "drawish"
    else:
        outcome = "engine_win" if (score > 0) == (engine_color == chess.WHITE) else "engine_loss"
    return {"game": game, "engine_color": "white" if engine_color == chess.WHITE else "black", "plies": plies, "outcome": outcome, "termination": termination, "final_eval_cp_white": round(score, 3)}


def match_suite(weights: dict[str, float], games: int, max_plies: int) -> dict:
    rows = [play(weights, index + 1, chess.WHITE if index % 2 == 0 else chess.BLACK, max_plies) for index in range(games)]
    counts = defaultdict(int)
    for row in rows:
        counts[row["outcome"]] += 1
    return {
        "opponent": "static-eval baseline using python-chess legal move generation",
        "games": games,
        "max_plies": max_plies,
        "engine_wins": counts["engine_win"],
        "engine_losses": counts["engine_loss"],
        "drawish": counts["drawish"],
        "score_rate": (counts["engine_win"] + 0.5 * counts["drawish"]) / games if games else 0.0,
        "rows": rows,
    }


def split_fens(fens: list[str], eval_fraction: float) -> tuple[list[str], list[str]]:
    if len(fens) < 2:
        return fens, fens
    eval_count = max(1, math.ceil(len(fens) * eval_fraction))
    return fens[eval_count:], fens[:eval_count]


def save_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a python-chess legal move evaluator from FEN positions")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", default="product_demo/poc_models")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    fens = [row.fen for row in read_fens(args.input, None)]
    train_fens, eval_fens = split_fens(fens, args.eval_fraction)
    weights, progress = train_from_positions(train_fens, args.epochs, args.learning_rate, args.seed)
    train_eval = evaluate_positions(train_fens, weights)
    eval_result = evaluate_positions(eval_fens, weights)
    matches = match_suite(weights, args.games, args.max_plies)
    payload = {
        "model_type": "basic-linear-python-chess-evaluator-v1",
        "legality_backend": "python-chess",
        "features": FEATURES,
        "weights": weights,
        "training_positions": len(train_fens),
        "eval_positions": len(eval_fens),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "progress": progress,
        "train_eval": train_eval,
        "eval": eval_result,
        "match": matches,
    }
    save_json(os.path.join(args.out_dir, "chess_engine_model.json"), payload)
    print(json.dumps({
        "out_dir": args.out_dir,
        "legality_backend": "python-chess",
        "training_positions": len(train_fens),
        "eval_positions": len(eval_fens),
        "final_train_accuracy": train_eval["top1_accuracy"],
        "eval_accuracy": eval_result["top1_accuracy"],
        "legal_prediction_rate": eval_result["legal_prediction_rate"],
        "match_score_rate": matches["score_rate"],
        "engine_wins": matches["engine_wins"],
        "engine_losses": matches["engine_losses"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
