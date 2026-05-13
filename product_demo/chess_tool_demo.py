from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from typing import Iterable

FILES = "abcdefgh"
RANKS = "12345678"
PIECE_VALUE = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0}
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class ToolError(ValueError):
    def __init__(self, error_code: str, safe_user_message: str):
        super().__init__(safe_user_message)
        self.error_code = error_code
        self.safe_user_message = safe_user_message


@dataclass(frozen=True)
class Move:
    source: str
    target: str
    promotion: str | None = None

    @classmethod
    def parse(cls, text: str) -> "Move":
        value = text.strip()
        if len(value) not in (4, 5):
            raise ToolError("invalid_move", "Move must be UCI like e2e4 or e7e8q.")
        source, target = value[:2], value[2:4]
        promotion = value[4].upper() if len(value) == 5 else None
        if not is_square(source) or not is_square(target):
            raise ToolError("invalid_square", "Move contains an invalid square.")
        if promotion and promotion not in {"Q", "R", "B", "N"}:
            raise ToolError("invalid_promotion", "Promotion must be q, r, b, or n.")
        return cls(source, target, promotion)

    def uci(self) -> str:
        return f"{self.source}{self.target}{(self.promotion or '').lower()}"


@dataclass
class Board:
    pieces: dict[str, str]
    turn: str
    history: list[tuple[Move, str | None, str]]

    @classmethod
    def from_fen(cls, fen: str = START_FEN) -> "Board":
        placement, turn, *_ = fen.split()
        pieces: dict[str, str] = {}
        for rank_index, row in enumerate(placement.split("/")):
            rank = str(8 - rank_index)
            file_index = 0
            for char in row:
                if char.isdigit():
                    file_index += int(char)
                else:
                    pieces[f"{FILES[file_index]}{rank}"] = char
                    file_index += 1
        return cls(pieces=pieces, turn=turn, history=[])

    def clone(self) -> "Board":
        return Board(copy.deepcopy(self.pieces), self.turn, list(self.history))

    def piece_at(self, square: str) -> str | None:
        return self.pieces.get(square)

    def move(self, move: Move) -> dict:
        legal = {item.uci(): item for item in self.legal_moves()}
        if move.uci() not in legal:
            return {
                "status": "error",
                "error_code": "illegal_move",
                "safe_user_message": f"{move.uci()} is not legal in this position.",
                "state_changed": False,
            }
        applied = legal[move.uci()]
        piece = self.pieces.pop(applied.source)
        captured = self.pieces.pop(applied.target, None)
        if applied.promotion and piece.upper() == "P":
            piece = applied.promotion if piece.isupper() else applied.promotion.lower()
        self.pieces[applied.target] = piece
        previous_turn = self.turn
        self.turn = other(self.turn)
        self.history.append((applied, captured, previous_turn))
        return {
            "status": "ok",
            "move": applied.uci(),
            "captured": captured,
            "state_changed": True,
            "side_to_move": self.turn,
            "evaluation": evaluate(self),
        }

    def undo(self) -> dict:
        if not self.history:
            return {"status": "error", "error_code": "empty_stack", "safe_user_message": "No move to undo."}
        move, captured, previous_turn = self.history.pop()
        piece = self.pieces.pop(move.target)
        if move.promotion:
            piece = "P" if piece.isupper() else "p"
        self.pieces[move.source] = piece
        if captured:
            self.pieces[move.target] = captured
        self.turn = previous_turn
        return {"status": "ok", "undone": move.uci(), "side_to_move": self.turn, "evaluation": evaluate(self)}

    def legal_moves(self) -> list[Move]:
        moves: list[Move] = []
        for square, piece in sorted(self.pieces.items()):
            if color_of(piece) != self.turn:
                continue
            for move in pseudo_moves(self, square, piece):
                probe = self.clone()
                probe.apply_unchecked(move)
                if not probe.king_in_check(self.turn):
                    moves.append(move)
        return moves

    def apply_unchecked(self, move: Move) -> None:
        piece = self.pieces.pop(move.source)
        self.pieces.pop(move.target, None)
        if move.promotion and piece.upper() == "P":
            piece = move.promotion if piece.isupper() else move.promotion.lower()
        self.pieces[move.target] = piece
        self.turn = other(self.turn)

    def king_in_check(self, color: str) -> bool:
        king = "K" if color == "w" else "k"
        king_square = next((square for square, piece in self.pieces.items() if piece == king), None)
        if king_square is None:
            return True
        enemy = other(color)
        for square, piece in self.pieces.items():
            if color_of(piece) != enemy:
                continue
            for target in attack_squares(self, square, piece):
                if target == king_square:
                    return True
        return False

    def snapshot(self) -> dict:
        return {
            "board": [[self.pieces.get(f"{file}{rank}", "") for file in FILES] for rank in reversed(RANKS)],
            "turn": self.turn,
            "legal_moves": [move.uci() for move in self.legal_moves()],
            "evaluation": evaluate(self),
            "move_count": len(self.history),
        }

    def ascii(self) -> str:
        rows = []
        for rank in reversed(RANKS):
            row = [self.pieces.get(f"{file}{rank}", ".") for file in FILES]
            rows.append(f"{rank} " + " ".join(row))
        rows.append("  a b c d e f g h")
        return "\n".join(rows)


def is_square(square: str) -> bool:
    return len(square) == 2 and square[0] in FILES and square[1] in RANKS


def color_of(piece: str) -> str:
    return "w" if piece.isupper() else "b"


def other(color: str) -> str:
    return "b" if color == "w" else "w"


def offset(square: str, df: int, dr: int) -> str | None:
    file_index = FILES.index(square[0]) + df
    rank_index = RANKS.index(square[1]) + dr
    if 0 <= file_index < 8 and 0 <= rank_index < 8:
        return f"{FILES[file_index]}{RANKS[rank_index]}"
    return None


def pseudo_moves(board: Board, square: str, piece: str) -> Iterable[Move]:
    kind = piece.upper()
    if kind == "P":
        yield from pawn_moves(board, square, piece)
    elif kind == "N":
        yield from knight_moves(board, square, piece)
    elif kind == "B":
        yield from sliding_moves(board, square, piece, [(-1, -1), (-1, 1), (1, -1), (1, 1)])
    elif kind == "R":
        yield from sliding_moves(board, square, piece, [(-1, 0), (1, 0), (0, -1), (0, 1)])
    elif kind == "Q":
        yield from sliding_moves(board, square, piece, [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)])
    elif kind == "K":
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if df == 0 and dr == 0:
                    continue
                target = offset(square, df, dr)
                if target and can_land(board, piece, target):
                    yield Move(square, target)


def pawn_moves(board: Board, square: str, piece: str) -> Iterable[Move]:
    direction = 1 if piece.isupper() else -1
    start_rank = "2" if piece.isupper() else "7"
    promotion_rank = "8" if piece.isupper() else "1"
    one = offset(square, 0, direction)
    if one and board.piece_at(one) is None:
        yield promote_or_plain(square, one, promotion_rank)
        two = offset(square, 0, direction * 2)
        if square[1] == start_rank and two and board.piece_at(two) is None:
            yield Move(square, two)
    for df in (-1, 1):
        target = offset(square, df, direction)
        if target and board.piece_at(target) and color_of(board.piece_at(target) or "") != color_of(piece):
            yield promote_or_plain(square, target, promotion_rank)


def promote_or_plain(source: str, target: str, promotion_rank: str) -> Move:
    if target[1] == promotion_rank:
        return Move(source, target, "Q")
    return Move(source, target)


def knight_moves(board: Board, square: str, piece: str) -> Iterable[Move]:
    for df, dr in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
        target = offset(square, df, dr)
        if target and can_land(board, piece, target):
            yield Move(square, target)


def sliding_moves(board: Board, square: str, piece: str, directions: list[tuple[int, int]]) -> Iterable[Move]:
    for df, dr in directions:
        target = offset(square, df, dr)
        while target:
            occupant = board.piece_at(target)
            if occupant is None:
                yield Move(square, target)
            else:
                if color_of(occupant) != color_of(piece):
                    yield Move(square, target)
                break
            target = offset(target, df, dr)


def can_land(board: Board, piece: str, target: str) -> bool:
    occupant = board.piece_at(target)
    return occupant is None or color_of(occupant) != color_of(piece)


def attack_squares(board: Board, square: str, piece: str) -> list[str]:
    if piece.upper() == "P":
        direction = 1 if piece.isupper() else -1
        return [target for df in (-1, 1) if (target := offset(square, df, direction))]
    return [move.target for move in pseudo_moves(board, square, piece)]


def evaluate(board: Board) -> dict:
    material = 0
    for piece in board.pieces.values():
        sign = 1 if piece.isupper() else -1
        material += sign * PIECE_VALUE[piece.upper()]
    turn_bonus = 4 * len(board.legal_moves()) if len(board.history) < 80 else 0
    score = material + (turn_bonus if board.turn == "w" else -turn_bonus)
    return {"score_cp_white": score, "bucket": bucket(score)}


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
    legal = board.legal_moves()
    if not legal:
        return {"status": "error", "error_code": "no_legal_moves"}
    scored = []
    for move in legal:
        probe = board.clone()
        probe.apply_unchecked(move)
        score = evaluate(probe)["score_cp_white"]
        scored.append((score if board.turn == "w" else -score, move))
    scored.sort(key=lambda item: (item[0], item[1].uci()), reverse=True)
    score, move = scored[0]
    return {"status": "ok", "best_move": move.uci(), "relative_score": score, "engine": "demo-material-mobility-v1"}


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
        if board.turn == "w":
            loss = max(0, best_eval - played_eval)
        else:
            loss = max(0, played_eval - best_eval)
    else:
        loss = 0
    quality = "good" if loss < 80 else "inaccuracy" if loss < 200 else "blunder"
    return {
        "status": "ok",
        "played": move.uci(),
        "best_known": best_before.get("best_move"),
        "quality": quality,
        "centipawn_loss_demo": loss,
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
        return {"status": "ok", "legal_moves": [move.uci() for move in board.legal_moves()]}
    if name == "eval":
        return {"status": "ok", **evaluate(board), "engine": "demo-material-mobility-v1"}
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
                {"role": "assistant", "content": "Engine-backed review: g1f3 is good. It develops a knight and matches the best known move from the demo oracle."},
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
