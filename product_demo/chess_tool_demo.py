from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import chess

ENGINE_ID = "python-chess-static-eval-v1"
PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


class ToolError(ValueError):
    def __init__(self, error_code: str, safe_user_message: str):
        super().__init__(safe_user_message)
        self.error_code = error_code
        self.safe_user_message = safe_user_message


@dataclass(frozen=True)
class Move:
    value: str

    @classmethod
    def parse(cls, text: str) -> "Move":
        value = text.strip().lower()
        try:
            chess.Move.from_uci(value)
        except ValueError as exc:
            raise ToolError("invalid_move", "Move must be UCI like e2e4 or e7e8q.") from exc
        return cls(value)

    def uci(self) -> str:
        return self.value


@dataclass
class Board:
    board: chess.Board

    @classmethod
    def from_fen(cls, fen: str | None = None) -> "Board":
        try:
            return cls(chess.Board(fen) if fen else chess.Board())
        except ValueError as exc:
            raise ToolError("invalid_fen", "Position data is invalid.") from exc

    def clone(self) -> "Board":
        return Board(self.board.copy(stack=True))

    def piece_at(self, square: str) -> str | None:
        try:
            piece = self.board.piece_at(chess.parse_square(square))
        except ValueError:
            return None
        return piece.symbol() if piece else None

    def legal_moves(self) -> list[Move]:
        return [Move(move.uci()) for move in self.board.legal_moves]

    def move(self, move: Move) -> dict:
        chess_move = chess.Move.from_uci(move.uci())
        if chess_move not in self.board.legal_moves:
            return {
                "status": "error",
                "error_code": "illegal_move",
                "safe_user_message": f"{move.uci()} is not legal in this position.",
                "state_changed": False,
            }
        captured = self.board.piece_at(chess_move.to_square)
        self.board.push(chess_move)
        return {
            "status": "ok",
            "move": move.uci(),
            "captured": captured.symbol() if captured else None,
            "state_changed": True,
            "side_to_move": "w" if self.board.turn == chess.WHITE else "b",
            "evaluation": evaluate(self),
        }

    def undo(self) -> dict:
        if not self.board.move_stack:
            return {"status": "error", "error_code": "empty_stack", "safe_user_message": "No move to undo."}
        move = self.board.pop()
        return {"status": "ok", "undone": move.uci(), "side_to_move": "w" if self.board.turn == chess.WHITE else "b", "evaluation": evaluate(self)}

    def snapshot(self) -> dict:
        return {
            "board": [[(piece.symbol() if (piece := self.board.piece_at(chess.square(file, rank))) else "") for file in range(8)] for rank in range(7, -1, -1)],
            "turn": "w" if self.board.turn == chess.WHITE else "b",
            "legal_moves": [move.uci() for move in self.board.legal_moves],
            "evaluation": evaluate(self),
            "move_count": len(self.board.move_stack),
        }

    def ascii(self) -> str:
        return str(self.board)


def material(board: chess.Board) -> int:
    score = 0
    for piece_type, value in PIECE_VALUE.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return score


def mobility(board: chess.Board) -> int:
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
    return white_moves - black_moves


def evaluate(board: Board) -> dict:
    raw = board.board
    if raw.is_checkmate():
        score = -100000 if raw.turn == chess.WHITE else 100000
    elif raw.is_stalemate() or raw.is_insufficient_material() or raw.can_claim_draw():
        score = 0
    else:
        score = material(raw) + 3 * mobility(raw)
    return {"score_cp_white": score, "bucket": bucket(score), "engine": ENGINE_ID}


def bucket(score: int) -> str:
    if score >= 300:
        return "white winning"
    if score <= -300:
        return "black winning"
    if score >= 80:
        return "white better"
    if score <= -80:
        return "black better"
    return "balanced"


def best_move(board: Board) -> dict:
    legal = list(board.board.legal_moves)
    if not legal:
        return {"status": "error", "error_code": "no_legal_moves"}
    color_sign = 1 if board.board.turn == chess.WHITE else -1
    scored = []
    for move in legal:
        probe = board.board.copy(stack=False)
        probe.push(move)
        score = evaluate(Board(probe))["score_cp_white"]
        scored.append((color_sign * score, move))
    score, move = max(scored, key=lambda item: (item[0], item[1].uci()))
    return {"status": "ok", "best_move": move.uci(), "relative_score": score, "engine": ENGINE_ID}


def review_move(board: Board, move: Move) -> dict:
    before = board.clone()
    best_before = best_move(before)
    result = before.move(move)
    if result["status"] != "ok":
        return result
    played_eval = evaluate(before)["score_cp_white"]
    if best_before["status"] == "ok":
        probe = board.clone()
        probe.move(Move.parse(best_before["best_move"]))
        best_eval = evaluate(probe)["score_cp_white"]
        loss = max(0, best_eval - played_eval) if board.board.turn == chess.WHITE else max(0, played_eval - best_eval)
    else:
        loss = 0
    quality = "good" if loss < 80 else "inaccuracy" if loss < 200 else "blunder"
    return {
        "status": "ok",
        "played": move.uci(),
        "best_known": best_before.get("best_move"),
        "quality": quality,
        "centipawn_loss": loss,
        "evidence": {"before": evaluate(board), "after": evaluate(before)},
        "state_changed": False,
    }


def tool_error_result(exc: ToolError) -> dict:
    return {"status": "error", "error_code": exc.error_code, "safe_user_message": exc.safe_user_message, "state_changed": False}


def run_tool_turn(board: Board, name: str, args: dict) -> dict:
    result = tool_call(board, name, args)
    return {"tool_result": result, "narration": narrator(name, result), "state": board.snapshot()}


def tool_call(board: Board, name: str, args: dict) -> dict:
    if name == "legal_moves":
        return {"status": "ok", "legal_moves": [move.uci() for move in board.board.legal_moves]}
    if name == "eval":
        return {"status": "ok", **evaluate(board)}
    if name == "best_move":
        return best_move(board)
    if name == "move":
        try:
            return board.move(Move.parse(args.get("uci", "")))
        except ToolError as exc:
            return tool_error_result(exc)
    if name == "review_move":
        try:
            return review_move(board, Move.parse(args.get("uci", "")))
        except ToolError as exc:
            return tool_error_result(exc)
    if name == "undo":
        return board.undo()
    return {"status": "error", "error_code": "unknown_tool"}


def narrator(tool_name: str, result: dict) -> str:
    if result["status"] != "ok":
        return result.get("safe_user_message", "Tool failed safely.")
    if tool_name == "review_move":
        return f"Engine-backed review: {result['played']} is {result['quality']}. Best known alternative: {result['best_known']}. Evidence bucket moved from {result['evidence']['before']['bucket']} to {result['evidence']['after']['bucket']}."
    if tool_name == "move":
        return f"Move accepted. Side to move: {result['side_to_move']}. Position is {result['evaluation']['bucket']}."
    if tool_name == "eval":
        return f"Current engine bucket: {result['bucket']} ({result['score_cp_white']} cp from White perspective)."
    return json.dumps(result, sort_keys=True)


def demo() -> None:
    board = Board.from_fen()
    script = [
        ("move", {"uci": "e2e4"}),
        ("move", {"uci": "e7e5"}),
        ("review_move", {"uci": "g1f3"}),
        ("move", {"uci": "g1f3"}),
        ("eval", {}),
    ]
    print("=== Board ===")
    print(board.ascii())
    for name, args in script:
        print(f"\nrouter -> {json.dumps({'tool_name': name, 'arguments': args})}")
        turn = run_tool_turn(board, name, args)
        print(f"tool <- {json.dumps(turn['tool_result'], sort_keys=True)}")
        print(f"narrator -> {turn['narration']}")
    print("\n=== Final Board ===")
    print(board.ascii())


def sft_records() -> list[dict]:
    return [
        {
            "messages": [
                {"role": "system", "content": "You are a chess coach router. Use tools for board-specific claims."},
                {"role": "user", "content": "I played e4. Put it on the board."},
                {"role": "assistant", "tool_call": {"tool_name": "move", "arguments": {"uci": "e2e4"}}},
                {"role": "tool", "name": "move", "content": {"status": "ok", "move": "e2e4", "side_to_move": "b"}},
                {"role": "assistant", "content": "Move accepted. It is Black to move."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You narrate only validated tool evidence. Do not invent board facts."},
                {"role": "user", "content": "Was Nf3 a good move here?"},
                {"role": "assistant", "tool_call": {"tool_name": "review_move", "arguments": {"uci": "g1f3"}}},
                {"role": "tool", "name": "review_move", "content": {"status": "ok", "played": "g1f3", "quality": "good", "best_known": "g1f3"}},
                {"role": "assistant", "content": "Engine-backed review: g1f3 is good. It develops a knight and matches validated engine evidence."},
            ]
        },
    ]


def write_sft(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in sft_records():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chess-engine-backed tool-call and SFT demo")
    parser.add_argument("command", choices=["demo", "sft"])
    parser.add_argument("--out", default="product_demo/sample_sft.jsonl")
    args = parser.parse_args()
    if args.command == "demo":
        demo()
    else:
        write_sft(args.out)
        print(args.out)


if __name__ == "__main__":
    main()
