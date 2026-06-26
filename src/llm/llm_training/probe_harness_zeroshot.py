"""PROBE: can the BASE Gemma 4 E4B operate OUR harness ZERO-SHOT (no fine-tune)?

The honest question raised this session: the base already reasons + calls tools, so does
our fine-tuning earn its place, or is the real product "base + harness + prompting"? This
gives the base the FULL multi-skill catalog + a native `load_skill` tool (progressive
disclosure) across domains, and checks whether it ROUTES to the right skill, LOADS it,
calls the right domain tool, and GROUNDS — reliably, not just on 3 easy cases.

Verdict logic: each scenario has an expected skill (or None = answer directly). High routing
accuracy zero-shot → "base + harness" may be the honest product (no fine-tune, and it dodges
the reasoning-suppression we found). Mis-routes (wrong skill among many, skips loading,
over-loads a direct question) → fine-tuning to drive the harness reliably is justified.

RUN ON A FRESH COLAB RUNTIME (stop the serve; one E4B per T4):
  PROBE_BASE=/content/llm-and-engine/src/llm/models/gemma4_e4b \
    PYTHONPATH=src/llm python -m llm_training.probe_harness_zeroshot
"""
from __future__ import annotations

from llm_training.probe_native_thinking import _fn, gen, load, parse

# load_skill = the progressive-disclosure verb, modeled as a native TOOL.
TOOLS = [
    _fn("load_skill", "Load a listed skill's instructions by name, then follow them.", {"name": {"type": "string"}}),
    _fn("board_state", "Pieces, side to move, legal moves of the current board."),
    _fn("best_move", "Engine's best move + evaluation.", {"depth": {"type": "integer"}}),
    _fn("threats", "Opponent's strongest threat."),
    _fn("name_opening", "Name the opening from the moves played."),
    _fn("random_position", "Set up a fresh position.", {"kind": {"type": "string"}}),
    _fn("scale_recipe", "Scale recipe quantities by a factor.", {"factor": {"type": "number"}}),
    _fn("convert_units", "Convert a quantity between units.", {"qty": {"type": "string"}, "to": {"type": "string"}}),
]

SYS = (
    "You operate a tool/skill harness. You have callable TOOLS (provided) and a catalog of "
    "SKILLS — each skill is on-demand instructions. To handle a request: if a skill fits, call "
    "load_skill{name=...} to read its instructions, then follow them (calling the right tools). "
    "Ground every claim in a tool result; never invent facts. If NO skill fits, just answer directly.\n"
    "SKILLS (load by name):\n"
    "- chess-coach: analyze a position, choose/explain a move, inspect the board.\n"
    "- opening-advisor: name the opening, give opening plans/theory.\n"
    "- game-reviewer: review a whole game — accuracy, blunders.\n"
    "- tactical-puzzles: set or coach a tactical puzzle.\n"
    "- recipe-scaler: scale recipe quantities, convert cooking units.\n"
    "- tax-filing-helper: basic personal tax-filing questions."
)

SKILL_BODIES = {
    "chess-coach": ("For any board claim, call board_state first; for the best move call best_move; "
                    "for danger call threats. Never invent pieces. Name concrete SAN moves and say why."),
    "opening-advisor": ("Call name_opening, then give the main plans for both sides. Don't dive into tactics."),
    "tactical-puzzles": ("Call random_position kind=puzzle, then coach the user to find the idea — don't spoil it."),
    "recipe-scaler": ("Call scale_recipe with the factor; for unit changes call convert_units. Never guess amounts."),
    "game-reviewer": ("Call the review tools to summarize accuracy and the key blunders."),
    "tax-filing-helper": ("Answer basic filing questions plainly; if a figure is needed, ask for it."),
}

CANNED = {
    "board_state": ("board_state: turn=white. White Re1, Kg1, pawns f2 g2 h2; Black Kg8, pawns f7 g7 h7. "
                    "No bishops/knights/queens. legal incl. Re8 (mate)."),
    "best_move": "best_move: Re8#  — mate in 1 (back-rank). White winning.",
    "threats": "threats: White threatens Re8# back-rank mate; Black has no threat.",
    "name_opening": "name_opening: Ruy Lopez (Spanish), Morphy Defence after 3...a6. White pressures e5/c6; Black hits the bishop.",
    "random_position": "random_position: puzzle set — mate in 1 (back-rank), fen=6k1/5ppp/8/8/8/8/5PPP/4R1K1 w.",
    "scale_recipe": "scale_recipe(factor=3): flour 600 g, sugar 300 g.",
    "convert_units": "convert_units: 600 g flour = 4.8 cups.",
}

# (label, expected_skill or None, [user turns])
SCENARIOS = [
    ("OPENING (specialized skill vs general)", "opening-advisor",
     ["What opening am I playing — 1.e4 e5 2.Nf3 Nc6 3.Bb5? Name it and the plans."]),
    ("CHESS best move + why", "chess-coach",
     ["FEN 6k1/5ppp/8/8/8/8/5PPP/4R1K1 — what's the best move here, and why?"]),
    ("PUZZLE", "tactical-puzzles", ["Can you give me a tactical puzzle to solve?"]),
    ("COOKING (cross-domain)", "recipe-scaler",
     ["I'm tripling this recipe (flour 200 g, sugar 100 g) — how much flour do I need?"]),
    ("NO-SKILL direct (must NOT over-route)", None, ["Hey — what kinds of things can you help me with?"]),
]


def run_turn(tok, proc, model, messages, label):
    print("\n" + "=" * 80 + f"\n{label}")
    loaded = []
    for step in range(6):
        _text, raw, _kw = gen(tok, model, messages, TOOLS)
        p = parse(proc, raw)
        print(f"[step {step}] CALLS : {p['tool_calls']}")
        if (p["thinking"] or "").strip():
            print(f"[step {step}] THINK : {p['thinking'][:180]}")
        if (p["content"] or "").strip():
            print(f"[step {step}] ANSWER: {p['content'][:240]}")
        if not p["tool_calls"]:
            messages.append({"role": "assistant", "content": p["content"]})
            return loaded
        messages.append({"role": "assistant", "content": p.get("content", "") or "",
                         "tool_calls": [{"type": "function", "function": {"name": tc["name"],
                          "arguments": tc.get("arguments", {})}} for tc in p["tool_calls"]]})
        for tc in p["tool_calls"]:
            if tc["name"] == "load_skill":
                nm = (tc.get("arguments") or {}).get("name", "")
                loaded.append(nm)
                res = SKILL_BODIES.get(nm, f"(unknown skill: {nm})")
            else:
                res = CANNED.get(tc["name"], f"(no canned result for {tc['name']})")
            messages.append({"role": "tool", "name": tc["name"], "content": res})
    return loaded


def main():
    tok, proc, model = load()
    print("\n>>> Can the BASE operate the harness zero-shot? (route -> load -> tool -> ground)")
    results = []
    for label, expected, turns in SCENARIOS:
        msgs = [{"role": "system", "content": SYS}]
        loaded = []
        for user in turns:
            msgs.append({"role": "user", "content": user})
            loaded += run_turn(tok, proc, model, msgs, f"{label}  [expected skill: {expected}]")
        if expected is None:
            ok = (len(loaded) == 0)
        else:
            ok = (expected in loaded)
        results.append((label, expected, loaded, ok))
    print("\n" + "=" * 80 + "\nROUTING SUMMARY (did it load the EXPECTED skill?):")
    hits = 0
    for label, expected, loaded, ok in results:
        hits += ok
        print(f"  {'OK  ' if ok else 'MISS'}  expected={expected!s:18} loaded={loaded}  ({label})")
    print(f"\nrouting: {hits}/{len(results)}. Also judge each final: grounded + correct?")
    print("High zero-shot routing => 'base + harness' is viable (no fine-tune). Mis-routes => fine-tune earns its place.")


if __name__ == "__main__":
    main()
