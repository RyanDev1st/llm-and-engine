from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundingFailure:
    error_id: str
    reason: str


def validate_narration(text: str, tool_result: dict) -> list[GroundingFailure]:
    failures: list[GroundingFailure] = []
    lowered = text.strip().lower()
    if not lowered or lowered.startswith("{"):
        failures.append(GroundingFailure("NARRATOR_TONE_INVALID", "narration must be human text"))
        return failures
    if tool_result.get("ok") is False:
        code = tool_result.get("error_code")
        if any(word in lowered for word in ["success", "applied", "done", "accepted"]):
            failures.append(GroundingFailure("NO_FABRICATED_SUCCESS", "failed tool narrated as success"))
        if code == "timeout" and "illegal" in lowered:
            failures.append(GroundingFailure("CANONICAL_ERROR_MAPPING", "timeout rewritten as illegal move"))
        if code == "invalid_move" and "timeout" in lowered:
            failures.append(GroundingFailure("CANONICAL_ERROR_MAPPING", "invalid_move rewritten as timeout"))
        return failures
    if "score" in lowered and "score_cp" not in tool_result:
        failures.append(GroundingFailure("GROUNDING_MISSING_FIELD", "score claim lacks score_cp"))
    if "best move" in lowered and "move" not in tool_result:
        failures.append(GroundingFailure("GROUNDING_MISSING_FIELD", "best move claim lacks move"))
    if any(word in lowered for word in ["accepted", "applied", "legal"]) and tool_result.get("ok") is not True:
        failures.append(GroundingFailure("MOVE_CLAIM_INVALID", "success claim mismatches tool result"))
    return failures
