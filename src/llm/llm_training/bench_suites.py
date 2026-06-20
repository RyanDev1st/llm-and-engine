"""Held-out STRESS suite for the benchmark — hand-written here, in NO training row, to measure
ROBUSTNESS the in-distribution val set can't: messy/slang/typo human phrasing, and UNSEEN
out-of-domain catalogs (tax / music / cooking / wellness) the model must route by reading the
in-context DESCRIPTION, not memory. Plus DECLINE cases (nothing fits -> answer directly).

Gold is UNAMBIGUOUS by construction (one specialized skill/tool matches each prompt, or nothing
does), so grading is fair. This is the tier where the 3-condition contrast is sharpest: the
product (adapter+harness) should route these unseen catalogs correctly under messy phrasing,
base+harness weaker, and base-no-harness near zero (no concept of the verbs at all).

Rows are shaped exactly like val rows (skills_index / tool_manifest / plugin_context / messages),
so eval_benchmark consumes them with no special-casing.
"""
from __future__ import annotations

# Out-of-domain catalog — none of these domains appear in training.
_SKILLS = [
    {"name": "tax-filing-helper", "description": "Use when the user asks about filing taxes, "
     "deductions, tax forms, brackets, or filing deadlines.", "plugin": "finance", "enabled": True},
    {"name": "guitar-tab-reader", "description": "Use when the user wants to read, interpret, or "
     "play guitar tablature, chords, or fingering.", "plugin": "music", "enabled": True},
    {"name": "recipe-scaler", "description": "Use when the user wants to scale a recipe up or down "
     "for a different number of servings.", "plugin": "cooking", "enabled": True},
    {"name": "meditation-coach", "description": "Use when the user wants to relax, de-stress, or be "
     "guided through a breathing or meditation exercise.", "plugin": "wellness", "enabled": True},
]
_TOOLS = [
    {"name": "convert_units", "description": "Convert a numeric value between measurement units.",
     "args": {"value": "required", "from_unit": "required", "to_unit": "required"}, "applies_when": "always"},
    {"name": "currency_rate", "description": "Get the current exchange rate between two currencies.",
     "args": {"base": "required", "quote": "required"}, "applies_when": "always"},
    {"name": "timer_set", "description": "Start a countdown timer for a number of seconds.",
     "args": {"seconds": "required"}, "applies_when": "always"},
]
_PC = {"installed": ["finance", "music", "cooking", "wellness"],
       "enabled": ["finance", "music", "cooking", "wellness"], "marketplace": []}

# (slice tag, user prompt, gold action). Gold '' / plain text => the 'none' (answer-directly) class.
_CASES = [
    # --- out-of-domain SKILL routing, clean phrasing ---
    ("ood_skill_clean", "I need to file my taxes this year, where do I even start?", "<skill>tax-filing-helper</skill>"),
    ("ood_skill_clean", "can you help me read this guitar tab?", "<skill>guitar-tab-reader</skill>"),
    ("ood_skill_clean", "scale my cookie recipe from 12 up to 30 servings", "<skill>recipe-scaler</skill>"),
    ("ood_skill_clean", "I'd like to be guided through a short meditation", "<skill>meditation-coach</skill>"),
    # --- out-of-domain SKILL routing, MESSY / slang / typo (robustness to phrasing drift) ---
    ("ood_skill_messy", "ugh taxes r due n i got no clue wtf im doing help", "<skill>tax-filing-helper</skill>"),
    ("ood_skill_messy", "yo can u help me figure out these gutiar chrods", "<skill>guitar-tab-reader</skill>"),
    ("ood_skill_messy", "wanna make like 3x the cookies lol how do i", "<skill>recipe-scaler</skill>"),
    ("ood_skill_messy", "stressed af rn need to chill n breathe", "<skill>meditation-coach</skill>"),
    ("ood_skill_messy", "my W2 is confusin and i dunno which deductons i can take", "<skill>tax-filing-helper</skill>"),
    ("ood_skill_messy", "halp this tablature thingy makes no sense to me", "<skill>guitar-tab-reader</skill>"),
    # --- out-of-domain TOOL routing (fill args from the schema) ---
    ("ood_tool_clean", "convert 5 miles into kilometers", "<tool>convert_units value=5 from_unit=miles to_unit=kilometers</tool>"),
    ("ood_tool_clean", "what is the USD to EUR exchange rate right now?", "<tool>currency_rate base=USD quote=EUR</tool>"),
    ("ood_tool_clean", "start a 300 second timer", "<tool>timer_set seconds=300</tool>"),
    ("ood_tool_messy", "how many km is liek 5 miles?? ", "<tool>convert_units value=5 from_unit=miles to_unit=km</tool>"),
    ("ood_tool_messy", "whats a pound in dollars these days", "<tool>currency_rate base=GBP quote=USD</tool>"),
    # --- DECLINE: nothing in the catalog fits -> answer directly, no action (gold = none) ---
    ("decline", "what is the capital of France?", "Paris is the capital of France."),
    ("decline", "tell me a fun fact about octopuses", "Octopuses have three hearts."),
    ("decline", "who won the 2018 world cup", "France won the 2018 World Cup."),
    ("decline", "lol what's 2+2", "2 + 2 = 4."),
    ("decline", "do you like pizza?", "I don't eat, but I can help with cooking tasks if you'd like."),
]


def stress_rows() -> list[dict]:
    """The held-out stress rows (drop-in for eval_benchmark / _system / gold_action)."""
    rows = []
    for sl, user, gold in _CASES:
        rows.append({
            "slice": f"STRESS_{sl}", "reasoning_mode": "",
            "skills_index": [dict(s) for s in _SKILLS],
            "tool_manifest": [dict(t) for t in _TOOLS],
            "plugin_context": dict(_PC),
            "messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": gold}],
        })
    return rows
