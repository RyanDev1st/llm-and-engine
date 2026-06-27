from __future__ import annotations

import random
from typing import Any

OFFICIAL_SKILL = {
    "name": "chess-coach",
    "description": "Analyze chess positions, choose moves, review mistakes, inspect board state, explain plans.",
    "plugin": "chess-official",
    "source": "official_plugin",
    "enabled": True,
}

HUMAN_CHAT_SKILL = {
    "name": "hood-human-chat",
    "description": "Normalize slang, shorthand, typo-heavy, vague, or multilingual-lite user chat like yo, watsup, idk, mb, ic before task routing.",
    "plugin": "user-skills",
    "source": "user_skill",
    "enabled": True,
}

USER_SKILL_TOOLS: list[dict[str, Any]] = [
    {"name": "normalize_human_chat", "description": "Translate slang, shorthand, typos, or vague chat into explicit task intent.", "args": {"text": "required"}, "applies_when": "always"},
]

# Canonical "calculator" template: the model substitutes its expression into this
# known-good snippet (plug-and-play) instead of composing novel code, so a weak
# coder still produces a runnable, two-decimal-grounded script. SINGLE SOURCE —
# surfaced in the tool description here AND used verbatim by the V1_R renderer.
CALC_TEMPLATE = 'print(f"{EXPR:.2f}")'

# Domain-neutral CORE verification tool (no plugin tag -> always callable; the
# plugin-gating validator skips tools without a plugin). The agent RUNS a short
# script and reads stdout to GROUND a computed/checkable claim instead of asserting
# it — the keystone of the Stage-0 "verification as tool-use" test. `code` is a
# free-text arg (captures the rest of the call; see toolfmt/validate). Defined here
# so build_system renders it identically at train and serve; executor is
# backend.sandbox.run_python.
COMPUTE_TOOLS: list[dict[str, Any]] = [
    {"name": "python", "description": "Run a short Python script and return its stdout — write a script that print()s the answer, then read it, to verify a computed or checkable claim instead of asserting it.", "args": {"code": "required"}, "applies_when": "always"},
]


def compute_tools() -> list[dict[str, Any]]:
    return [dict(tool) for tool in COMPUTE_TOOLS]


# v5-native: skill loading is a native TOOL (Gemma has no `<skill>` verb). The
# skills_index is still listed in the prompt; load_skill pulls one skill's body
# into context before the model uses it (progressive disclosure).
HARNESS_TOOLS: list[dict[str, Any]] = [
    {"name": "load_skill", "description": "Load a listed skill's instructions into context before you use it.", "args": {"name": "required"}, "applies_when": "always"},
]


def harness_tools() -> list[dict[str, Any]]:
    return [dict(tool) for tool in HARNESS_TOOLS]

OFFICIAL_TOOLS: list[dict[str, Any]] = [
    {"name": "move", "description": "Play a SAN move on the live board.", "args": {"san": "required"}, "applies_when": "game_in_progress"},
    {"name": "load_fen", "description": "Set the board to a position from a FEN string (e.g. to set up a puzzle).", "args": {"fen": "required"}, "applies_when": "always"},
    {"name": "new_game", "description": "Reset the board to the starting position for a new game.", "args": {}, "applies_when": "always"},
    {"name": "random_position", "description": "Set up a fresh position for a skill to work on: a tactical puzzle, a random scramble, or a random opening.", "args": {"kind": ["puzzle", "scramble", "open"]}, "applies_when": "always"},
    {"name": "fetch_puzzle", "description": "Fetch a real, rated tactical puzzle from Lichess (online) with verified themes and solution, and set it on the board.", "args": {}, "applies_when": "always"},
    {"name": "eval", "description": "Evaluate the current chess position.", "args": {"depth": "required"}, "applies_when": "game_in_progress"},
    {"name": "best_move", "description": "Find the engine's best move or principal variation.", "args": {"depth": "required", "top": ["1", "2", "3", "4", "5"], "series": ["1", "2", "3", "4", "5"]}, "applies_when": "game_in_progress"},
    {"name": "what_if", "description": "Weigh a move you're considering: score the position after it and compare to the engine's best — use it for 'should I play X?' or 'is X or Y better?'.", "args": {"san": "required"}, "applies_when": "game_in_progress"},
    {"name": "review_move", "description": "Judge the last move played.", "args": {"depth": "required"}, "applies_when": "has_history"},
    {"name": "threats", "description": "Show opponent's strongest threat.", "args": {"depth": "required"}, "applies_when": "game_in_progress"},
    {"name": "legal_moves", "description": "List legal moves overall or for one square.", "args": {"square": "required"}, "applies_when": "game_in_progress"},
    {"name": "undo", "description": "Take back the last move.", "args": {}, "applies_when": "has_history"},
    {"name": "list_pieces", "description": "List remaining pieces by color.", "args": {"color": ["white", "black", "mine"]}, "applies_when": "always"},
    {"name": "ask_chessbot", "description": "Answer general chess knowledge from a small KB.", "args": {"query": "required"}, "applies_when": "always"},
    {"name": "board_state", "description": "Snapshot the hidden live board.", "args": {"fields": ["basic", "all", "fen"]}, "applies_when": "always"},
]
def official_tools() -> list[dict[str, Any]]:
    return [
        {**tool, "plugin": "chess-official", "source": "official_plugin", "enabled": True}
        for tool in OFFICIAL_TOOLS
    ]


# --- Flat pure-chess catalog (v5) -------------------------------------------------
# The serve harness aggregates every enabled plugin into ONE flat catalog the model
# sees, with no plugin gating. v5 training mirrors that exactly (train==serve parity):
# the generalist coach + three specialists + the human-chat normalizer, and the core
# tools + the opening/analysis specialist tools + the human-chat tool + python. NO
# plugin_context, NO cross-domain distractors. Specialist skill descriptions and tool
# schemas are VERBATIM from backend/plugins/{openings,analysis,puzzles}.py so routing
# keys on the same text the served model will see.
SPECIALIST_SKILLS: list[dict[str, Any]] = [
    {"name": "game-reviewer", "description": "Use when the user asks how they played overall, their accuracy, or to find blunders across the game."},
    {"name": "opening-advisor", "description": "Use when the user asks what opening this is, or for opening plans, theory, or a repertoire."},
    {"name": "tactical-puzzles", "description": "Use when the user wants a tactical puzzle, to practice or hone tactics, or to be coached through a combination."},
    {"name": "plan-audit", "description": "Verify a plan's checkable claims by running the numbers (the python tool) and reading the output, not by asserting them."},
]

SPECIALIST_TOOLS: list[dict[str, Any]] = [
    {"name": "name_opening", "description": "Identify the opening being played from the move history.", "args": {}, "applies_when": "has_history"},
    {"name": "opening_ideas", "description": "Give the typical plans and ideas for the current opening.", "args": {}, "applies_when": "has_history"},
    {"name": "accuracy_report", "description": "Score how accurately the whole game was played so far (per-side accuracy).", "args": {"depth": "required"}, "applies_when": "has_history"},
    {"name": "find_blunders", "description": "List the blunders made so far in the game with the better move.", "args": {"depth": "required"}, "applies_when": "has_history"},
]


def chess_skills() -> list[dict[str, Any]]:
    """The flat pure-chess skill catalog listed in every row: generalist coach + the
    three specialists + the human-chat normalizer. Flat (no plugin/enabled gating) —
    the model picks by description/context, exactly as it must at serve."""
    coach = {"name": OFFICIAL_SKILL["name"], "description": OFFICIAL_SKILL["description"]}
    chat = {"name": HUMAN_CHAT_SKILL["name"], "description": HUMAN_CHAT_SKILL["description"]}
    return [coach, *[dict(s) for s in SPECIALIST_SKILLS], chat]


def chess_tools() -> list[dict[str, Any]]:
    """The flat pure-chess tool manifest: core coach tools + opening/analysis specialist
    tools + the human-chat helper tool + python. Flat, parity with the served manifest."""
    return (
        harness_tools()
        + [dict(t) for t in OFFICIAL_TOOLS]
        + [dict(t) for t in SPECIALIST_TOOLS]
        + [dict(t) for t in USER_SKILL_TOOLS]
        + compute_tools()
    )


def with_plugin(items: list[dict[str, Any]], plugin: str, source: str, enabled: bool = True) -> list[dict[str, Any]]:
    return [{**item, "plugin": plugin, "source": source, "enabled": enabled} for item in items]


def alt_skills() -> list[dict[str, str]]:
    return [
        {"name": "socratic-tutor", "description": "Teach by asking short guiding questions instead of giving answers."},
        {"name": "endgame-drills", "description": "Drill king-and-pawn, rook, and minor-piece endings with targeted exercises."},
        {"name": "tactic-trainer", "description": "Spot forks, pins, skewers, and discovered attacks in a position."},
        {"name": "opening-prep", "description": "Suggest mainline and sideline ideas for the current opening."},
        {"name": "blunder-coach", "description": "Identify and explain blunders, return a corrective plan."},
        {"name": "rating-coach", "description": "Guide players in a specific rating band on practical priorities."},
        {"name": "cooking-helper", "description": "Plan and adjust recipes, substitutions, and timing."},
        {"name": "math-tutor", "description": "Explain math problems step by step at the user's level."},
        {"name": "code-reviewer", "description": "Comment on code diffs and suggest improvements."},
        {"name": "writer-coach", "description": "Improve writing tone, clarity, and structure."},
    ]


def alt_tools() -> list[dict[str, Any]]:
    return [
        {"name": "search_kb", "description": "Search a knowledge base.", "args": {"query": "required"}, "applies_when": "always"},
        {"name": "recipe_lookup", "description": "Find recipes by ingredient.", "args": {"ingredient": "required"}, "applies_when": "always"},
        {"name": "math_eval", "description": "Evaluate a math expression.", "args": {"expr": "required"}, "applies_when": "always"},
        {"name": "diff_view", "description": "Show a code diff range.", "args": {"path": "required", "range": "required"}, "applies_when": "always"},
        {"name": "style_check", "description": "Run a writing style check.", "args": {"text": "required"}, "applies_when": "always"},
        {"name": "drill_pick", "description": "Pick an endgame drill by theme.", "args": {"theme": "required"}, "applies_when": "always"},
        {"name": "tactic_spot", "description": "Highlight tactical motifs in a chess position.", "args": {"depth": "required"}, "applies_when": "game_in_progress"},
        {"name": "opening_book", "description": "Look up opening moves.", "args": {"fen": "required"}, "applies_when": "game_in_progress"},
    ]


def synthetic_skill_name(seed: int) -> str:
    rng = random.Random(seed)
    prefix = rng.choice(["skill", "ski", "plugin", "ext"])
    body = rng.choice(["pluto", "vega", "kappa", "zen", "orbit", "echo", "lyra"])
    suffix = rng.randint(2, 99)
    return f"{prefix}-{body}-{suffix}"


def synthetic_tool_name(seed: int) -> str:
    rng = random.Random(seed)
    body = rng.choice(["zb", "qx", "vk", "om", "rt", "lp"])
    suffix = rng.randint(10, 999)
    return f"tool_{body}_{suffix}"
