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

    # ===== EXPANSION (P1-c): broaden each slice for a real robustness claim (n>=60). Gold stays
    # UNAMBIGUOUS — one catalog skill/tool matches, or nothing (decline). Tool args are illustrative
    # (the routing score keys off verb+name only), but kept realistic so the transcript runs. =====

    # --- more SKILL routing, clean phrasing ---
    ("ood_skill_clean", "I'm filing my tax return for the first time and feel lost", "<skill>tax-filing-helper</skill>"),
    ("ood_skill_clean", "teach me how to read guitar chord diagrams", "<skill>guitar-tutor</skill>"),
    ("ood_skill_clean", "I need to double my pancake recipe", "<skill>recipe-scaler</skill>"),
    ("ood_skill_clean", "walk me through a calming breathing routine", "<skill>breathing-coach</skill>"),
    ("ood_skill_clean", "what's the standard deduction and how do I claim it?", "<skill>tax-filing-helper</skill>"),
    ("ood_skill_clean", "I can't make sense of this guitar tablature", "<skill>guitar-tutor</skill>"),

    # --- more SKILL routing, messy / slang / typo ---
    ("ood_skill_messy", "taxez due soon n i still havent started ughhh", "<skill>tax-filing-helper</skill>"),
    ("ood_skill_messy", "how do i even read gutair tabs man", "<skill>guitar-tutor</skill>"),
    ("ood_skill_messy", "need 2 make half the recipe, like 4 servings not 8", "<skill>recipe-scaler</skill>"),
    ("ood_skill_messy", "rly anxious rn can u help me calm down w breathing", "<skill>breathing-coach</skill>"),
    ("ood_skill_messy", "which deductionz do i even qualify 4 idk", "<skill>tax-filing-helper</skill>"),
    ("ood_skill_messy", "explain this chord chart thingo pls", "<skill>guitar-tutor</skill>"),

    # --- more TOOL routing, clean phrasing (length / mass / temp / tempo / timer) ---
    ("ood_tool_clean", "convert 10 kilometers to miles", "<tool>convert_units value=10 from_unit=kilometers to_unit=miles</tool>"),
    ("ood_tool_clean", "how many pounds is 5 kg?", "<tool>convert_units value=5 from_unit=kg to_unit=pounds</tool>"),
    ("ood_tool_clean", "turn 32 fahrenheit into celsius", "<tool>convert_units value=32 from_unit=fahrenheit to_unit=celsius</tool>"),
    ("ood_tool_clean", "what's 2 feet in centimeters?", "<tool>convert_units value=2 from_unit=feet to_unit=cm</tool>"),
    ("ood_tool_clean", "convert 500 grams to ounces", "<tool>convert_units value=500 from_unit=grams to_unit=ounces</tool>"),
    ("ood_tool_clean", "how many meters is 100 feet?", "<tool>convert_units value=100 from_unit=feet to_unit=meters</tool>"),
    ("ood_tool_clean", "set a metronome to 90 bpm", "<tool>metronome_bpm bpm=90</tool>"),
    ("ood_tool_clean", "give me a click track at 140 beats per minute", "<tool>metronome_bpm bpm=140</tool>"),
    ("ood_tool_clean", "start a 120 second breathing timer", "<tool>breathing_timer seconds=120</tool>"),

    # --- more TOOL routing, messy / slang / typo ---
    ("ood_tool_messy", "how many lbs in 20 kgs??", "<tool>convert_units value=20 from_unit=kgs to_unit=lbs</tool>"),
    ("ood_tool_messy", "convrt 6 ft to meters plz", "<tool>convert_units value=6 from_unit=ft to_unit=meters</tool>"),
    ("ood_tool_messy", "wuts 0 celcius in F", "<tool>convert_units value=0 from_unit=celsius to_unit=fahrenheit</tool>"),
    ("ood_tool_messy", "metronme at 100bpm thx", "<tool>metronome_bpm bpm=100</tool>"),
    ("ood_tool_messy", "gimme a click @ 75 bpm", "<tool>metronome_bpm bpm=75</tool>"),
    ("ood_tool_messy", "120sec breathin timer go", "<tool>breathing_timer seconds=120</tool>"),
    ("ood_tool_messy", "how many km in 26.2 miles lol marathon", "<tool>convert_units value=26.2 from_unit=miles to_unit=km</tool>"),
    ("ood_tool_messy", "set tempo 60bpm", "<tool>metronome_bpm bpm=60</tool>"),

    # --- more DECLINE: nothing in THIS catalog fits (no calc/translate/search tool) -> answer ---
    ("decline", "what's the tallest mountain on earth?", "Mount Everest is the tallest mountain above sea level."),
    ("decline", "who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
    ("decline", "how far is the moon from earth?", "The Moon is about 384,400 km from Earth on average."),
    ("decline", "what's 12 times 13?", "12 x 13 = 156."),
    ("decline", "tell me a joke", "Why did the scarecrow win an award? He was outstanding in his field."),
    ("decline", "what year did world war 2 end?", "World War II ended in 1945."),
    ("decline", "how do i say hello in spanish?", "You say 'hola'."),
    ("decline", "whats your favorite color", "I don't have one, but I'm happy to help with a task."),
    ("decline", "recommend a good sci-fi movie", "Blade Runner is a classic sci-fi film."),
    ("decline", "how many continents are there?", "There are seven continents."),
    ("decline", "spell the word necessary", "It's spelled n-e-c-e-s-s-a-r-y."),
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
