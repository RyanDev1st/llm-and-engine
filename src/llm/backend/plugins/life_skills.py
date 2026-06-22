"""life-skills plugin: a REAL out-of-domain bundle (cooking / music / wellness / tax) whose
domains are ABSENT from the chess training corpus. It exists to prove the product claim end to
end on unseen domains: the model routes by reading the in-context DESCRIPTION (not memory), loads
a real SKILL.md body, calls a real tool, and narrates a real result. NOT a scaffold — every tool
has a deterministic executor (real math, no external API, so it's reproducible) and every skill
has a real instructional body. The benchmark's STRESS suite sources its catalog from here.

Not enabled in the default chess serve (kept out of PLUGIN_CONTEXT); the benchmark/transcript
turn it on per-request via plugin_context. Registered in plugins.REGISTRY so its tools dispatch
and its skill bodies load when enabled."""
from __future__ import annotations

NAME = "life-skills"

# --- real, deterministic unit conversions (factor TO a canonical base, per dimension) ---
_LENGTH = {"mile": 1609.34, "miles": 1609.34, "mi": 1609.34, "km": 1000.0, "kilometer": 1000.0,
           "kilometers": 1000.0, "kilometre": 1000.0, "m": 1.0, "meter": 1.0, "meters": 1.0,
           "ft": 0.3048, "feet": 0.3048, "foot": 0.3048, "cm": 0.01, "in": 0.0254, "inch": 0.0254}
_MASS = {"lb": 0.453592, "lbs": 0.453592, "pound": 0.453592, "pounds": 0.453592, "kg": 1.0,
         "kilogram": 1.0, "kilograms": 1.0, "g": 0.001, "gram": 0.001, "grams": 0.001,
         "oz": 0.0283495, "ounce": 0.0283495, "ounces": 0.0283495}


def _convert(value: str, frm: str, to: str) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "error: convert_units needs a numeric value"
    f, t = (frm or "").lower().strip(), (to or "").lower().strip()
    if f in {"c", "celsius"} and t in {"f", "fahrenheit"}:
        return f"convert: {v:g} C = {v * 9 / 5 + 32:g} F"
    if f in {"f", "fahrenheit"} and t in {"c", "celsius"}:
        return f"convert: {v:g} F = {(v - 32) * 5 / 9:g} C"
    for table, dim in ((_LENGTH, "length"), (_MASS, "mass")):
        if f in table and t in table:
            return f"convert: {v:g} {frm} = {v * table[f] / table[t]:.4g} {to} ({dim})"
    return f"error: convert_units can't convert {frm!r} to {to!r}"


def _scale(frm: str, to: str) -> str:
    try:
        a, b = float(frm), float(to)
    except (TypeError, ValueError):
        return "error: scale_recipe needs numeric from_servings and to_servings"
    if a <= 0:
        return "error: from_servings must be > 0"
    return f"scale_recipe: multiply every ingredient by {b / a:.3g}x (from {a:g} to {b:g} servings)"


def _metronome(bpm: str) -> str:
    try:
        b = float(bpm)
    except (TypeError, ValueError):
        return "error: metronome_bpm needs a numeric bpm"
    if b <= 0:
        return "error: bpm must be > 0"
    return f"metronome_bpm: {b:g} bpm = {60000 / b:.1f} ms per beat"


def _breathing(seconds: str) -> str:
    try:
        s = int(float(seconds))
    except (TypeError, ValueError):
        return "error: breathing_timer needs a number of seconds"
    if s <= 0:
        return "error: seconds must be > 0"
    cycles = max(1, s // 19)            # a 4-7-8 breath cycle is ~19s
    return f"breathing_timer: {s}s set — about {cycles} slow 4-7-8 breath cycle(s)."


TOOLS = [
    {"name": "convert_units", "description": "Convert a numeric value between measurement units "
     "(length, mass, or temperature).", "args": {"value": "required", "from_unit": "required",
     "to_unit": "required"}, "applies_when": "always"},
    {"name": "scale_recipe", "description": "Get the multiplier to scale a recipe from one number "
     "of servings to another.", "args": {"from_servings": "required", "to_servings": "required"},
     "applies_when": "always"},
    {"name": "metronome_bpm", "description": "Convert a musical tempo in beats-per-minute to "
     "milliseconds per beat.", "args": {"bpm": "required"}, "applies_when": "always"},
    {"name": "breathing_timer", "description": "Start a guided breathing timer for a number of "
     "seconds.", "args": {"seconds": "required"}, "applies_when": "always"},
]


def _body(name: str, desc: str, steps: str) -> str:
    return f"---\nname: {name}\ndescription: {desc}\n---\n\n# {name}\n\n{steps}"


SKILLS = [
    {"name": "recipe-scaler",
     "description": "Use when the user wants to scale a recipe up or down for a different number of servings.",
     "body": _body("recipe-scaler", "Scale a recipe to a new serving count.",
                   "Read the serving counts from the user's message (e.g. 'from 12 up to 30' -> "
                   "from_servings=12, to_servings=30) and call `scale_recipe from_servings=<a> "
                   "to_servings=<b>` right away. Do NOT ask for numbers the user already gave — only "
                   "ask if a count is genuinely missing. Then tell the user the multiplier to apply "
                   "to every ingredient. Never guess the factor — use the tool.")},
    {"name": "guitar-tutor",
     "description": "Use when the user wants to read guitar tablature, chords, fingering, or set a practice tempo.",
     "body": _body("guitar-tutor", "Read tablature and set a practice tempo.",
                   "Explain the tab/chord plainly. For tempo, use the BPM the user named (e.g. "
                   "'120 bpm' -> bpm=120) and call `metronome_bpm bpm=<n>` right away — don't ask for "
                   "a number they gave. Report the ms-per-beat so they can set a metronome. Ground "
                   "tempo in the tool's number.")},
    {"name": "breathing-coach",
     "description": "Use when the user wants to relax, de-stress, or be guided through a breathing exercise.",
     "body": _body("breathing-coach", "Guide a short breathing exercise.",
                   "Reassure briefly, then call `breathing_timer seconds=<n>` — use the duration the "
                   "user gave if any, otherwise default to 60 (don't ask for a number they already "
                   "provided). Then walk them through slow 4-7-8 breaths for that duration.")},
    {"name": "tax-filing-helper",
     "description": "Use when the user asks about filing taxes, deductions, tax forms, brackets, or filing deadlines.",
     "body": _body("tax-filing-helper", "General tax-filing guidance (not advice).",
                   "Give plain, general filing steps (gather forms, check the standard vs itemized "
                   "deduction, note the deadline). State you are not a tax professional. There is no "
                   "calculation tool here — answer from the loaded guidance.")},
]

_DISPATCH = {"convert_units": lambda a: _convert(a.get("value", ""), a.get("from_unit", ""), a.get("to_unit", "")),
             "scale_recipe": lambda a: _scale(a.get("from_servings", ""), a.get("to_servings", "")),
             "metronome_bpm": lambda a: _metronome(a.get("bpm", "")),
             "breathing_timer": lambda a: _breathing(a.get("seconds", ""))}


def handle(name: str, args: dict, executor) -> str | None:
    """Execute one of this bundle's tools; None if the name isn't ours (registry routes on)."""
    fn = _DISPATCH.get(name)
    return fn(args) if fn else None
