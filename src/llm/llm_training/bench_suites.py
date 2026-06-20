"""Held-out STRESS suite for the benchmark — hand-written prompts (in NO training row) over a
REAL out-of-domain bundle. The catalog the model routes on is sourced from `backend.plugins.
life_skills` (real skill bodies + real deterministic tool executors), NOT hand-waved dicts — so
these are real capabilities, not scaffolds: a correct route can load a real body and call a real
tool end to end (see bench_transcript.py).

Measures ROBUSTNESS the in-distribution val set can't: messy/slang/typo human phrasing, UNSEEN
out-of-domain catalogs the model must route by reading the in-context DESCRIPTION (not memory),
and DECLINE cases (nothing fits -> answer directly). Gold is UNAMBIGUOUS by construction (one
specialized skill/tool matches, or nothing). Rows are shaped exactly like val rows, so
eval_benchmark consumes them with no special-casing."""
from __future__ import annotations

from backend import plugins

# Enable the real out-of-domain bundle so its manifest + skill catalog are what the model sees.
PC = {"installed": ["life-skills"], "enabled": ["life-skills"], "marketplace": []}
_SKILLS = plugins.plugin_skills(PC)       # real catalog (name + description), from the real bundle
_TOOLS = plugins.plugin_tools(PC)         # real tool manifest (name + args + description)

# (slice tag, user prompt, gold first-action). Gold '' / plain text => the 'none' class.
# Skill requests route to the skill (the domain method); a bare data request routes to the tool.
_CASES = [
    # --- out-of-domain SKILL routing, clean phrasing ---
    ("ood_skill_clean", "I need to file my taxes this year, where do I even start?", "<skill>tax-filing-helper</skill>"),
    ("ood_skill_clean", "can you help me read this guitar tab?", "<skill>guitar-tutor</skill>"),
    ("ood_skill_clean", "scale my cookie recipe from 12 up to 30 servings", "<skill>recipe-scaler</skill>"),
    ("ood_skill_clean", "I'd like to be guided through a short breathing exercise", "<skill>breathing-coach</skill>"),
    # --- out-of-domain SKILL routing, MESSY / slang / typo (robustness to phrasing drift) ---
    ("ood_skill_messy", "ugh taxes r due n i got no clue wtf im doing help", "<skill>tax-filing-helper</skill>"),
    ("ood_skill_messy", "yo can u help me figure out these gutiar chrods", "<skill>guitar-tutor</skill>"),
    ("ood_skill_messy", "wanna make like 3x the cookies lol how do i", "<skill>recipe-scaler</skill>"),
    ("ood_skill_messy", "stressed af rn need to chill n breathe", "<skill>breathing-coach</skill>"),
    ("ood_skill_messy", "my W2 is confusin and i dunno which deductons i can take", "<skill>tax-filing-helper</skill>"),
    ("ood_skill_messy", "halp this tablature thingy makes no sense to me", "<skill>guitar-tutor</skill>"),
    # --- out-of-domain TOOL routing (fill args from the schema) ---
    ("ood_tool_clean", "convert 5 miles into kilometers", "<tool>convert_units value=5 from_unit=miles to_unit=kilometers</tool>"),
    ("ood_tool_clean", "what's 100 celsius in fahrenheit?", "<tool>convert_units value=100 from_unit=celsius to_unit=fahrenheit</tool>"),
    ("ood_tool_clean", "set a metronome to 120 bpm", "<tool>metronome_bpm bpm=120</tool>"),
    ("ood_tool_messy", "how many km is liek 5 miles??", "<tool>convert_units value=5 from_unit=miles to_unit=km</tool>"),
    ("ood_tool_messy", "gimme a 90 sec breathing timer", "<tool>breathing_timer seconds=90</tool>"),
    # --- DECLINE: nothing in the catalog fits -> answer directly, no action (gold = none) ---
    ("decline", "what is the capital of France?", "Paris is the capital of France."),
    ("decline", "tell me a fun fact about octopuses", "Octopuses have three hearts."),
    ("decline", "who won the 2018 world cup", "France won the 2018 World Cup."),
    ("decline", "lol what's 2+2", "2 + 2 = 4."),
    ("decline", "do you like pizza?", "I don't eat, but I can help scale a pizza recipe if you'd like."),
]


def stress_rows() -> list[dict]:
    """The held-out stress rows (drop-in for eval_benchmark / _system / gold_action). Catalog is
    the REAL life-skills bundle, so a routed action can run end to end."""
    rows = []
    for sl, user, gold in _CASES:
        rows.append({
            "slice": f"STRESS_{sl}", "reasoning_mode": "",
            "skills_index": [dict(s) for s in _SKILLS],
            "tool_manifest": [dict(t) for t in _TOOLS],
            "plugin_context": dict(PC),
            "messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": gold}],
        })
    return rows
