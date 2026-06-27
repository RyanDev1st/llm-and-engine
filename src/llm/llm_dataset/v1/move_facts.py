"""Move-effect facts: turn a (FEN, SAN move) into the TRUE, grounded ingredients of
a 'why this move' explanation — derived geometrically from python-chess so a
generated final states only what is real, and the validator can re-check it.

This is the keystone behind the 80%-no-why corpus gap: v1-v4 finals named the move
and the line but never WHY, because no fact about the move's *effect* was ever
extracted. Every field here is a verifiable property of the move (does it check?
capture? fork two pieces? pin one to the king? win material? develop? mate?), so the
composer can justify the move and the gate can reject any claim that isn't true.

NOT a serve-time dependency: this is purely a dataset generation + validation aid.
At serve the trained model produces the 'why' itself."""
from __future__ import annotations

from dataclasses import dataclass

import chess

PIECE_NAMES = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}
_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
_MINORS = (chess.KNIGHT, chess.BISHOP)


@dataclass(frozen=True)
class MoveFacts:
    san: str
    piece: str                          # moved piece (name)
    gives_check: bool
    is_mate: bool
    is_capture: bool
    captured: str | None                # captured piece name
    capture_square: str | None
    wins_material: bool                 # free capture, or capturing up in value
    is_castling: bool
    develops_minor: bool                # a knight/bishop leaving its back rank
    promotes: str | None                # promoted piece name
    forks: tuple[str, ...]              # squares of >=2 valuable enemy pieces the moved piece now hits
    fork_names: tuple[str, ...]         # their piece names, aligned with forks
    pin_to_king: str | None             # square of an enemy piece newly pinned to its king
    pin_name: str | None                # that piece's name
    attacks_queen: bool                 # the moved piece now attacks the enemy queen (a tempo)

    def tactic_terms(self) -> set[str]:
        """The grounding vocabulary this move LICENSES — the only tactical words a final
        may use about it. The validator maps a final's words back to this set."""
        terms: set[str] = set()
        if self.is_mate:
            terms |= {"mate", "checkmate"}
        if self.gives_check:
            terms.add("check")
        if self.is_capture:
            terms.add("capture")
        if self.wins_material:
            terms |= {"wins", "free", "material"}
        if self.forks:
            terms |= {"fork", "double attack", "both"}
        if self.pin_to_king:
            terms.add("pin")
        if self.is_castling:
            terms |= {"castle", "king safety"}
        if self.develops_minor:
            terms.add("develop")
        if self.attacks_queen:
            terms.add("queen")
        return terms


def _occupant_type(board: chess.Board, move: chess.Move) -> int:
    return move.promotion or board.piece_at(move.from_square).piece_type


def _captured(board: chess.Board, move: chess.Move) -> tuple[str | None, str | None]:
    if not board.is_capture(move):
        return None, None
    if board.is_en_passant(move):
        return "pawn", chess.square_name(move.to_square)
    victim = board.piece_at(move.to_square)
    return (PIECE_NAMES[victim.piece_type] if victim else None), chess.square_name(move.to_square)


def _wins_material(board: chess.Board, after: chess.Board, move: chess.Move) -> bool:
    if not board.is_capture(move):
        return False
    victim = board.piece_at(move.to_square)
    cap_value = 1 if board.is_en_passant(move) else (_VALUE[victim.piece_type] if victim else 0)
    mover = not after.turn                                   # side that just moved
    if not after.attackers(after.turn, move.to_square):     # nobody can recapture -> free
        return cap_value >= 1
    return cap_value > _VALUE[_occupant_type(board, move)]   # captured up in value despite recapture


def _multi_attack(after: chess.Board, to_sq: int, enemy: chess.Color) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Valuable enemy pieces (>= knight, incl. king) the piece now on to_sq attacks."""
    sqs, names = [], []
    for sq in after.attacks(to_sq):
        pc = after.piece_at(sq)
        if pc and pc.color == enemy and _VALUE[pc.piece_type] >= 3:
            sqs.append(chess.square_name(sq))
            names.append(PIECE_NAMES[pc.piece_type])
    if len(sqs) >= 2:
        return tuple(sqs), tuple(names)
    return (), ()


def _pin_to_king(after: chess.Board, to_sq: int, enemy: chess.Color) -> tuple[str | None, str | None]:
    """An enemy piece the moved (sliding) piece pins against its king: the king sits on
    the far side of the line with exactly that one piece between."""
    mover_pc = after.piece_at(to_sq)
    if not mover_pc or mover_pc.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        return None, None
    ksq = after.king(enemy)
    if ksq is None:
        return None, None
    for sq in after.attacks(to_sq):
        pc = after.piece_at(sq)
        if not pc or pc.color != enemy or pc.piece_type == chess.KING:
            continue
        between = chess.SquareSet(chess.between(to_sq, ksq))
        if sq in between and (between & chess.SquareSet(after.occupied)) == chess.SquareSet([sq]):
            return chess.square_name(sq), PIECE_NAMES[pc.piece_type]
    return None, None


def move_facts(fen: str, san: str) -> MoveFacts | None:
    """Extract the move-effect facts, or None if the move is illegal in the position."""
    board = chess.Board(fen)
    try:
        move = board.parse_san(san)
    except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError, ValueError):
        return None
    mover = board.turn
    enemy = not mover
    piece = board.piece_at(move.from_square)
    clean = board.san(move)
    captured, cap_sq = _captured(board, move)
    gives_check = board.gives_check(move)
    is_castling = board.is_castling(move)
    back_rank = 0 if mover == chess.WHITE else 7
    develops = (piece.piece_type in _MINORS
                and chess.square_rank(move.from_square) == back_rank
                and chess.square_rank(move.to_square) != back_rank)
    after = board.copy()
    after.push(move)
    forks, fork_names = _multi_attack(after, move.to_square, enemy)
    pin_sq, pin_name = _pin_to_king(after, move.to_square, enemy)
    aq = any(after.piece_at(s) and after.piece_at(s).color == enemy
             and after.piece_at(s).piece_type == chess.QUEEN
             for s in after.attacks(move.to_square))
    return MoveFacts(
        san=clean,
        piece=PIECE_NAMES[piece.piece_type],
        gives_check=gives_check,
        is_mate=after.is_checkmate(),
        is_capture=board.is_capture(move),
        captured=captured,
        capture_square=cap_sq,
        wins_material=_wins_material(board, after, move),
        is_castling=is_castling,
        develops_minor=develops,
        promotes=PIECE_NAMES[move.promotion] if move.promotion else None,
        forks=forks,
        fork_names=fork_names,
        pin_to_king=pin_sq,
        pin_name=pin_name,
        attacks_queen=aq,
    )
