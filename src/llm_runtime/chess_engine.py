from __future__ import annotations

from dataclasses import dataclass, field

from .engine_protocol import ToolContext
from .errors import error_result


@dataclass
class ChessSession:
    moves: list[str] = field(default_factory=list)

    @property
    def state(self) -> str:
        return f"moves:{' '.join(self.moves)}" if self.moves else "startpos"


class ChessToolBackend:
    def __init__(self) -> None:
        self._sessions: dict[str, ChessSession] = {}

    def execute(self, tool: str, args: dict, context: ToolContext) -> dict:
        session = self._sessions.setdefault(context.conversation_id, ChessSession())
        if tool == "move":
            move = str(args.get("uci", ""))
            if not move or move == "illegal":
                return error_result(tool, "invalid_move", "illegal move")
            session.moves.append(move)
            return {"ok": True, "tool": tool, "state": session.state, "applied": move}
        if tool == "undo":
            if session.moves:
                session.moves.pop()
            return {"ok": True, "tool": tool, "state": session.state, "undone": True}
        if tool == "eval":
            return {"ok": True, "tool": tool, "state": session.state, "score_cp": 24}
        if tool == "best_move":
            return {"ok": True, "tool": tool, "state": session.state, "move": "e2e4", "top_k": ["e2e4", "d2d4", "g1f3"]}
        if tool == "review_move":
            san = str(args.get("san", ""))
            if not san:
                return error_result(tool, "invalid_move", "missing san")
            return {"ok": True, "tool": tool, "state": session.state, "san": san, "classification": "good", "delta_cp": 18}
        if tool == "legal_moves":
            return {"ok": True, "tool": tool, "state": session.state, "moves": ["e2e4", "d2d4", "g1f3"]}
        if tool == "list_pieces":
            return {"ok": True, "tool": tool, "state": session.state, "pieces": ["wk:e1", "bk:e8"]}
        if tool == "threats":
            return {"ok": True, "tool": tool, "state": session.state, "threats": ["...Qh4+"]}
        if tool == "ask_chessbot":
            return {"ok": True, "tool": tool, "answer": "Consider developing a knight."}
        return error_result(tool, "invalid_position", f"unsupported tool {tool}")
