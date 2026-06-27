"""Reasoning-mode gating: fast/think/auto must be a real toggle, and AUTO must
think only on hard-decision steps (interleaved), so the model learns WHEN to
think — not always, not never. Validates the gating helpers + the corpus
invariant (fast rows carry no <think>) + the system-prompt signal."""
from collections import Counter

from llm_dataset.v1.renderer.thinking import (
    MODES, gated_answer, gated_fix, gated_think, pick_mode,
)
from llm_dataset.v1.validate import validate_row
from llm_training.system_prompt import build_system


def test_pick_mode_covers_all_three_and_is_seeded():
    counts = Counter(pick_mode(s) for s in range(2000))
    assert set(counts) == set(MODES)               # all three appear
    assert all(counts[m] > 200 for m in MODES)     # each is well-represented
    assert pick_mode(123) == pick_mode(123)         # deterministic


def test_fast_never_thinks():
    assert gated_think(1, "load_skill", 0, mode="fast", kind="select") == ""
    assert gated_fix(1, "eval", mode="fast") == ""
    assert gated_answer(1, "x", mode="fast") == ""


def test_think_always_thinks():
    assert "<think>" in gated_think(1, "load_skill", 0, mode="think", kind="routine")
    assert "<think>" in gated_answer(1, "x", mode="think")
    assert "<think>" in gated_fix(1, "eval", mode="think")


def test_auto_thinks_only_on_hard_steps():
    # routine step -> no think; judgment steps (select/decide/recover/answer) -> think
    assert gated_think(1, "board_state", 2, mode="auto", kind="routine") == ""
    assert "<think>" in gated_think(1, "load_skill", 0, mode="auto", kind="select")
    assert "<think>" in gated_think(1, "legal_moves", 3, mode="auto", kind="decide")
    assert "<think>" in gated_fix(1, "eval", mode="auto")       # recovery is hard
    assert "<think>" in gated_answer(1, "x", mode="auto")        # goal-met check is hard


def _fast_row(with_think: bool) -> dict:
    content = ("<think>plan.</think>\n" if with_think else "") + "First.\n<tool>eval depth=15</tool>"
    return {
        "id": "x", "slice": "D", "kind": "harness_chess", "reasoning_mode": "fast",
        "intent": "i", "plugin_context": {"enabled": ["chess-official"]},
        "skills_index": [{"name": "chess-coach", "description": "d", "plugin": "chess-official",
                          "source": "official_plugin", "enabled": True}],
        "selected_skills": [], "tool_manifest": [
            {"name": "eval", "description": "e", "args": {"depth": "int"}, "applies_when": "always",
             "plugin": "chess-official", "source": "official_plugin", "enabled": True}],
        "expected_tool_calls": ["eval"], "grounding_sources": [],
        "messages": [{"role": "user", "content": "rate"},
                     {"role": "assistant", "content": content},
                     {"role": "assistant", "content": "About level."}],
        "acceptance_rules": ["final_no_xml", "known_tool_only", "args_match_schema"],
        "position_fen": None, "stockfish_truth": None,
    }


def test_validator_rejects_think_in_fast_row():
    bad = [v for v in validate_row(_fast_row(with_think=True)) if v.rule == "reasoning_mode_fast_no_think"]
    assert bad, "fast row with <think> must be flagged"
    ok = [v for v in validate_row(_fast_row(with_think=False)) if v.rule == "reasoning_mode_fast_no_think"]
    assert not ok, "fast row without <think> must pass the mode check"


def test_system_prompt_renders_mode_line():
    assert "FAST" in build_system([], [], {}, reasoning_mode="fast")
    assert "THINK" in build_system([], [], {}, reasoning_mode="think")
    # v4.1: reasoning is Gemma-native (enable_thinking), so "auto" resolves to a reasoning-ON
    # prompt (the serve router picks fast vs think BEFORE this renders) — it commits a <goal>
    # and reasons, rather than naming a distinct AUTO mode.
    auto = build_system([], [], {}, reasoning_mode="auto")
    assert "THINK" in auto and "<goal>" in auto
    # The mode lines no longer instruct the custom <think> tag (native handles reasoning).
    assert "<think>" not in build_system([], [], {}, reasoning_mode="think")
    assert "Reasoning mode" not in build_system([], [], {})  # unset -> no line (back-compat)
