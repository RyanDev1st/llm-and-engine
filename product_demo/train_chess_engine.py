from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
from collections import defaultdict

import chess

from prepare_kaggle_sft import read_fens, validate_rows

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
    white_probe = board.copy(stack=False)
    white_probe.turn = chess.WHITE
    black_probe = board.copy(stack=False)
    black_probe.turn = chess.BLACK
    return float(len(list(white_probe.legal_moves)) - len(list(black_probe.legal_moves)))


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


class StockfishOpponent:
    def __init__(self, path: str, movetime_ms: int, skill_level: int | None) -> None:
        self.path = path
        self.movetime_ms = movetime_ms
        self.process = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok")
        if skill_level is not None:
            self._send(f"setoption name Skill Level value {skill_level}")
        self._send("isready")
        self._read_until("readyok")

    def close(self) -> None:
        if self.process.poll() is None:
            self._send("quit")
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def move(self, board: chess.Board) -> chess.Move | None:
        if board.is_game_over(claim_draw=True):
            return None
        self._send(f"position fen {board.fen()}")
        self._send(f"go movetime {self.movetime_ms}")
        line = self._read_until("bestmove")
        parts = line.split()
        if len(parts) < 2 or parts[1] == "(none)":
            return None
        move = chess.Move.from_uci(parts[1])
        if move not in board.legal_moves:
            raise ValueError(f"Stockfish returned illegal move {move.uci()} for {board.fen()}")
        return move

    def _send(self, command: str) -> None:
        if self.process.stdin is None:
            raise RuntimeError("Stockfish stdin unavailable")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _read_until(self, marker: str) -> str:
        if self.process.stdout is None:
            raise RuntimeError("Stockfish stdout unavailable")
        while True:
            line = self.process.stdout.readline()
            if line == "":
                stderr = self.process.stderr.read() if self.process.stderr else ""
                raise RuntimeError(f"Stockfish exited before {marker}: {stderr.strip()}")
            if line.startswith(marker) or line.strip() == marker:
                return line.strip()


def choose_opponent_move(board: chess.Board, opponent: StockfishOpponent | None) -> chess.Move | None:
    if opponent is not None:
        return opponent.move(board)
    return baseline_move(board)


def play(weights: dict[str, float], game: int, engine_color: chess.Color, max_plies: int, opponent: StockfishOpponent | None = None) -> dict:
    board = chess.Board()
    plies = 0
    termination = "max_plies"
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            termination = board.outcome(claim_draw=True).termination.name.lower()
            break
        move = engine_move(board, weights) if board.turn == engine_color else choose_opponent_move(board, opponent)
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


def match_suite(weights: dict[str, float], games: int, max_plies: int, stockfish_path: str | None = None, stockfish_movetime_ms: int = 50, stockfish_skill_level: int | None = None) -> dict:
    opponent = None
    opponent_name = "same-heuristic static-eval baseline using python-chess legal move generation"
    if stockfish_path:
        opponent = StockfishOpponent(stockfish_path, stockfish_movetime_ms, stockfish_skill_level)
        opponent_name = f"Stockfish UCI opponent at {stockfish_movetime_ms}ms/move"
        if stockfish_skill_level is not None:
            opponent_name += f", skill_level={stockfish_skill_level}"
    try:
        rows = [play(weights, index + 1, chess.WHITE if index % 2 == 0 else chess.BLACK, max_plies, opponent) for index in range(games)]
    finally:
        if opponent is not None:
            opponent.close()
    counts = defaultdict(int)
    for row in rows:
        counts[row["outcome"]] += 1
    return {
        "opponent": opponent_name,
        "stockfish_path": stockfish_path,
        "stockfish_movetime_ms": stockfish_movetime_ms if stockfish_path else None,
        "stockfish_skill_level": stockfish_skill_level if stockfish_path else None,
        "games": games,
        "max_plies": max_plies,
        "engine_wins": counts["engine_win"],
        "engine_losses": counts["engine_loss"],
        "drawish": counts["drawish"],
        "score_rate": (counts["engine_win"] + 0.5 * counts["drawish"]) / games if games else 0.0,
        "rows": rows,
    }


def play_stockfish_engine_game(game: int, engine_color: chess.Color, max_plies: int, product_engine: StockfishOpponent, opponent: StockfishOpponent) -> dict:
    board = chess.Board()
    plies = 0
    termination = "max_plies"
    moves = []
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            termination = outcome.termination.name.lower() if outcome else "game_over"
            break
        move = product_engine.move(board) if board.turn == engine_color else opponent.move(board)
        if move is None:
            termination = "no_legal_moves"
            break
        moves.append(move.uci())
        board.push(move)
        plies += 1
    score = static_eval(board)
    if board.is_checkmate():
        outcome = "engine_win" if board.turn != engine_color else "engine_loss"
    elif abs(score) < 80:
        outcome = "drawish"
    else:
        outcome = "engine_win" if (score > 0) == (engine_color == chess.WHITE) else "engine_loss"
    return {
        "game": game,
        "engine_color": "white" if engine_color == chess.WHITE else "black",
        "plies": plies,
        "outcome": outcome,
        "termination": termination,
        "final_eval_cp_white": round(score, 3),
        "moves": moves[:80],
    }


def stockfish_engine_match(engine_path: str, opponent_path: str, games: int, max_plies: int, engine_movetime_ms: int, opponent_movetime_ms: int, engine_skill_level: int | None, opponent_skill_level: int | None) -> dict:
    product_engine = StockfishOpponent(engine_path, engine_movetime_ms, engine_skill_level)
    opponent = StockfishOpponent(opponent_path, opponent_movetime_ms, opponent_skill_level)
    try:
        rows = [play_stockfish_engine_game(index + 1, chess.WHITE if index % 2 == 0 else chess.BLACK, max_plies, product_engine, opponent) for index in range(games)]
    finally:
        product_engine.close()
        opponent.close()
    counts = defaultdict(int)
    for row in rows:
        counts[row["outcome"]] += 1
    return {
        "engine": f"Stockfish UCI engine at {engine_movetime_ms}ms/move, skill_level={engine_skill_level}",
        "opponent": f"Stockfish UCI opponent at {opponent_movetime_ms}ms/move, skill_level={opponent_skill_level}",
        "engine_path": engine_path,
        "opponent_path": opponent_path,
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
    parser.add_argument("--stockfish-path", default=None)
    parser.add_argument("--stockfish-movetime-ms", type=int, default=50)
    parser.add_argument("--stockfish-skill-level", type=int, default=None)
    parser.add_argument("--use-stockfish-engine", action="store_true")
    parser.add_argument("--stockfish-engine-movetime-ms", type=int, default=50)
    parser.add_argument("--stockfish-engine-skill-level", type=int, default=1)
    args = parser.parse_args()

    loaded = read_fens(args.input, None)
    valid_rows, rejection_reasons = validate_rows(loaded.rows)
    if not valid_rows:
        raise ValueError(f"No valid FEN rows found in {args.input}; rejected={dict(sorted(rejection_reasons.items()))}")
    fens = [row.fen for row in valid_rows]
    train_fens, eval_fens = split_fens(fens, args.eval_fraction)
    weights, progress = train_from_positions(train_fens, args.epochs, args.learning_rate, args.seed)
    train_eval = evaluate_positions(train_fens, weights)
    eval_result = evaluate_positions(eval_fens, weights)
    matches = match_suite(weights, args.games, args.max_plies, args.stockfish_path, args.stockfish_movetime_ms, args.stockfish_skill_level)
    stockfish_engine_matches = None
    if args.use_stockfish_engine:
        if not args.stockfish_path:
            raise ValueError("--use-stockfish-engine requires --stockfish-path")
        stockfish_engine_matches = stockfish_engine_match(
            args.stockfish_path,
            args.stockfish_path,
            args.games,
            args.max_plies,
            args.stockfish_engine_movetime_ms,
            args.stockfish_movetime_ms,
            args.stockfish_engine_skill_level,
            args.stockfish_skill_level,
        )
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
        "stockfish_engine_match": stockfish_engine_matches,
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
        "match_opponent": matches["opponent"],
        "stockfish_engine_score_rate": stockfish_engine_matches["score_rate"] if stockfish_engine_matches else None,
        "stockfish_engine_wins": stockfish_engine_matches["engine_wins"] if stockfish_engine_matches else None,
        "stockfish_engine_losses": stockfish_engine_matches["engine_losses"] if stockfish_engine_matches else None,
        "engine_wins": matches["engine_wins"],
        "engine_losses": matches["engine_losses"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
