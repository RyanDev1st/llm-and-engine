from __future__ import annotations

CANONICAL_ERRORS = {"timeout", "engine_unavailable", "invalid_position", "invalid_move"}


def error_result(tool: str, error_code: str, message: str) -> dict:
    if error_code not in CANONICAL_ERRORS:
        raise ValueError(f"unknown canonical error: {error_code}")
    return {"ok": False, "tool": tool, "error_code": error_code, "message": message}
