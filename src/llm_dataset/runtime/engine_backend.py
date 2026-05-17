from __future__ import annotations

from dataclasses import dataclass, field
import json

from ..contracts.tool_grammar import parse_tool_name

CANONICAL_ERRORS = {"timeout", "engine_unavailable", "invalid_position", "invalid_move"}
STATEFUL_TOOLS = {"move", "undo", "legal_moves", "list_pieces", "eval", "best_move", "review_move", "threats"}


@dataclass
class EngineSession:
    state: str = "startpos"
    moves: list[str] = field(default_factory=list)


class EngineToolBackend:
    def __init__(self) -> None:
        self._sessions: dict[str, EngineSession] = {}

    def execute(self, tool_call: str) -> str:
        tool_name = parse_tool_name(tool_call)
        if not tool_name:
            return _error("invalid_position", "invalid tool call")
        payload = _parse_call(tool_call)
        conv_id = payload.get("conversation_id", "default")
        session = self._sessions.setdefault(conv_id, EngineSession())
        if tool_name not in STATEFUL_TOOLS:
            return _error("invalid_position", f"unsupported tool {tool_name}")
        return self._dispatch(tool_name, payload, session)

    def _dispatch(self, tool_name: str, payload: dict[str, str], session: EngineSession) -> str:
        if payload.get("force_error") in CANONICAL_ERRORS:
            return _error(payload["force_error"], "forced error", detail="test injection")
        if tool_name == "move":
            move = payload.get("uci", "")
            if not move or move == "illegal":
                return _error("invalid_move", "illegal move")
            session.moves.append(move)
            session.state = f"moves:{' '.join(session.moves)}"
            return _ok(tool_name, session, {"applied": move})
        if tool_name == "undo":
            if session.moves:
                session.moves.pop()
            session.state = f"moves:{' '.join(session.moves)}" if session.moves else "startpos"
            return _ok(tool_name, session, {"undone": True})
        if tool_name == "legal_moves":
            return _ok(tool_name, session, {"moves": ["e2e4", "d2d4", "g1f3"]})
        if tool_name == "list_pieces":
            return _ok(tool_name, session, {"pieces": ["wk:e1", "bk:e8"]})
        if tool_name == "eval":
            return _ok(tool_name, session, {"score_cp": 24})
        if tool_name == "best_move":
            return _ok(tool_name, session, {"move": "e2e4", "top_k": ["e2e4", "d2d4", "g1f3"]})
        if tool_name == "review_move":
            return _ok(tool_name, session, {"classification": "good", "delta_cp": 18})
        if tool_name == "threats":
            return _ok(tool_name, session, {"threats": ["...Qh4+"]})
        return _error("invalid_position", f"unsupported tool {tool_name}")


def _parse_call(tool_call: str) -> dict[str, str]:
    body = tool_call.replace("<tool>", "").replace("</tool>", "").strip()
    tokens = body.split()
    args = tokens[1:] if len(tokens) > 1 else []
    out: dict[str, str] = {}
    for item in args:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        out[key] = value
    return out


def _ok(tool: str, session: EngineSession, extra: dict) -> str:
    payload = {"ok": True, "tool": tool, "state": session.state, **extra}
    return json.dumps(payload, sort_keys=True)


def _error(code: str, message: str, detail: str | None = None) -> str:
    payload = {"ok": False, "error_code": code, "message": message}
    if detail:
        payload["detail"] = detail
    return json.dumps(payload, sort_keys=True)
