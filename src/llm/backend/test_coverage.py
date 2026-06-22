"""The coverage layer on the single loop: the model must not finish a turn while a
detected intent is ungathered. It is steered once (s1-style "Wait"), then the tool
is force-routed as a backstop. coverage=False disables it (the ablation)."""
from backend.game import Game
from backend.inference import CoachLoop
from backend.tools import ToolExecutor
from backend.toolfmt import parse_call


class ScriptedModel:
    def __init__(self, steps):
        self.steps = list(steps)
        self.i = 0

    def generate(self, messages, max_new_tokens, stop):
        out = self.steps[min(self.i, len(self.steps) - 1)]
        self.i += 1
        return out


def _names(out):
    # tool_calls now DISPLAYS skill loads as the trained verb <skill>NAME</skill>
    # (execution still uses load_skill); map it back so these behavior assertions hold.
    names = []
    for c in out["tool_calls"]:
        n = parse_call(c)[0]
        if n is None and "<skill>" in c:
            n = "load_skill"
        names.append(n)
    return names


def _loop(steps, game=None):
    return CoachLoop(ScriptedModel(steps), ToolExecutor(game or Game(), None))


def test_model_proactive_multi_tool_then_reply():
    # "best move and the evaluation" -> required {best_move, eval}.
    # Model proactively gathers BOTH itself, then replies — no forcing needed.
    out = _loop([
        "<tool>best_move top=3",      # gather best_move
        "<tool>eval depth=18",        # gather eval (model-driven)
        "Final summary.",              # all covered -> final reply (mentions no fact)
    ]).respond([], "give me the best move and the evaluation")
    assert _names(out) == ["best_move", "eval"]
    # the vague reply mentions neither fact -> answer-coverage appends both, grounded
    assert out["reply"].startswith("Final summary.")


def test_force_routes_outstanding_when_model_stops_early():
    # required {eval}. Model stops without it -> force-routed directly (no nudge round-trip).
    out = _loop([
        "Just play e4.",               # stops; eval outstanding -> force-route eval
        "Okay, evaluation noted.",     # all covered -> final reply (states no number)
    ]).respond([], "how am I doing?")
    assert "eval" in _names(out)
    assert out["reply"].startswith("Okay, evaluation noted.")  # eval fact appended after


def test_on_event_fires_per_tool_for_streaming():
    # Streaming progress: on_event must fire once per executed tool, in order, each
    # carrying the tool name + result — so the UI can show steps live.
    events = []
    out = _loop([
        "<tool>eval depth=18",
        "<tool>best_move top=3",
        "Done.",
    ]).respond([], "give me the best move and the evaluation", on_event=events.append)
    assert [e["type"] for e in events] == ["tool", "tool"]
    assert [e["name"] for e in events] == ["eval", "best_move"]
    assert all(e["result"] for e in events)        # each event carries the tool result
    assert _names(out) == ["eval", "best_move"]     # non-streaming return shape unchanged


def test_plural_best_moves_force_routed_after_eval():
    # The screenshot: "eval and the 5 next best moves" — model evals then stops to ask.
    # Coverage must force-route best_move (top=5) so both are gathered.
    out = _loop([
        "<tool>eval depth=18",                       # eval gathered
        "Want me to suggest the top 5 moves?",        # tries to answer; best_move outstanding -> Wait
        "Sure, here they are.",                        # still no best_move -> backstop force-routes it
    ]).respond([], "can you eval and give me the 5 next best moves?")
    assert set(_names(out)) == {"eval", "best_move"}
    assert any("top=5" in c for c in out["tool_calls"])   # honored the requested count


def test_coverage_off_lets_the_model_stop_early():
    out = _loop(["Just play e4."]).respond([], "how am I doing?", coverage=False)
    assert out["tool_calls"] == [] and out["reply"] == "Just play e4."


def test_no_required_intent_returns_first_reply():
    out = _loop(["Hello there!"]).respond([], "hi")
    assert out["tool_calls"] == [] and out["reply"] == "Hello there!"


def test_leadin_only_terminal_reply_narrates_tool_result():
    # the live "it didn't finish" bug: model ran ask_chessbot, then its final reply was
    # just "Loading the chess-coach skill." (a dangling lead-in) -> narrate the real result.
    out = _loop([
        "<tool>ask_chessbot query=explain chess",          # runs ask_chessbot
        "Loading the chess-coach skill.",                   # terminal lead-in, no tool -> non-answer
    ]).respond([], "explain chess in 8 sentences")
    assert "ask_chessbot" in _names(out)
    assert "Loading the chess-coach skill" not in out["reply"]      # the dangling lead-in is replaced
    assert out["reply"]                                             # with a real narrated answer


def test_leadin_only_kept_when_no_tools_ran():
    # without tools, a lead-in is the model's actual (if weak) reply — don't fabricate.
    out = _loop(["Let me check that for you."]).respond([], "hi")
    assert out["reply"] == "Let me check that for you."


def test_skill_load_then_whiff_retries_real_answer():
    # The live bug: "explain chess" -> model loads chess-coach skill -> then gives an
    # empty/greeting non-answer -> the generic "What would you like to look at?" replaced
    # the real reply. A context-only (skill) load must be FOLLOWED by the answer: retry
    # once with an explicit "answer now" nudge instead of greeting.
    out = _loop([
        "<tool>load_skill name=chess-coach",                       # loads context
        "",                                                         # whiff: empty final reply
        "Chess is a strategic board game between two players.",     # forced retry answers
    ]).respond([], "explain chess in 8 sentences")
    assert _names(out) == ["load_skill"]
    assert out["reply"].startswith("Chess is a strategic")
    assert "What would you like to look at" not in out["reply"]


def test_skill_load_then_whiff_falls_back_if_retry_also_whiffs():
    # If even the forced retry produces nothing usable, fall back to the greeting (last
    # resort) — never hang or emit empty.
    out = _loop([
        "<tool>load_skill name=chess-coach",
        "",          # whiff
        "",          # forced retry also whiffs -> greeting fallback
    ]).respond([], "explain chess in 8 sentences")
    assert _names(out) == ["load_skill"]
    assert out["reply"]                                            # never empty
    assert out["reply"] == "What would you like to look at on the board?"


def test_skill_load_deflection_is_verified_then_fulfilled():
    # the live bug: "what should I go next?" -> model loads chess-coach -> then DEFLECTS
    # ("The skill is loaded. What would you like to do?") instead of answering. That reply
    # isn't empty/lead-in so the whiff guard misses it. Design B self-verify: the model is
    # asked if it fulfilled the request; it returns the next tool, which runs and is narrated.
    out = _loop([
        "<tool>load_skill name=chess-coach",                       # loads context
        "The skill is loaded. What would you like to do with this position?",  # DEFLECTION
        "<tool>best_move depth=18",                                # verify -> next tool
        "e4 is the strongest move here.",                          # narrate the real result
    ]).respond([], "whats my plan here")
    assert "best_move" in _names(out)                              # the deflection was repaired
    assert "What would you like to do" not in out["reply"]         # not the deflection
    assert "e4" in out["reply"]


def test_skill_load_genuine_answer_passes_verification():
    # if the model DID answer after loading the skill, verification returns DONE and the
    # answer stands — no spurious extra tool.
    out = _loop([
        "<tool>load_skill name=chess-coach",
        "Develop your pieces and castle early; the center is balanced.",  # a real answer
        "DONE",                                                            # verify: fulfilled
    ]).respond([], "whats my plan here")
    assert _names(out) == ["load_skill"]
    assert out["reply"].startswith("Develop your pieces")


def test_plain_chat_no_tools_is_not_verified():
    # verify must NOT fire when no tool ran (pure chat) — only on skill-load-without-fact.
    out = _loop(["Hi! Ask me to read the board or suggest a move."]).respond([], "hello")
    assert out["tool_calls"] == []
    assert out["reply"].startswith("Hi!")


def test_fact_tool_whiff_keeps_grounded_fallback_not_retry():
    # When a FACT tool ran (not a skill), the grounded fallback narrates the real result;
    # the answer-retry must NOT fire (we have a fact, don't risk a re-gen).
    out = _loop([
        "<tool>ask_chessbot query=explain chess",
        "Loading the chess-coach skill.",   # leadin non-answer, but ask_chessbot has a result
    ]).respond([], "explain chess in 8 sentences")
    assert "ask_chessbot" in _names(out)
    assert "Loading the chess-coach skill" not in out["reply"]


def test_game_over_skips_coverage():
    g = Game()
    for san in ["f3", "e5", "g4", "Qh4#"]:
        g.move(san)
    out = _loop(["That's checkmate — Black wins."], game=g).respond([], "how am I doing?")
    assert out["tool_calls"] == [] and "checkmate" in out["reply"].lower()


def test_answer_coverage_appends_dropped_eval():
    # The screenshot bug: model gathers eval AND best_move, but the final reply only
    # narrates the moves. Answer-coverage must append the eval fact (grounded).
    out = _loop([
        "<tool>eval depth=18",          # required eval gathered
        "<tool>best_move top=3",        # required best_move gathered
        "The engine suggests e4, then d4 and c4. Want me to play e4?",  # drops the eval
    ]).respond([], "suggest 3 next best moves and the eval")
    assert set(_names(out)) == {"eval", "best_move"}
    assert "e4" in out["reply"]                      # moves still there
    # eval result at the start position is "score: 0.00 ... equal" -> appended fact
    assert "0.00" in out["reply"] or "equal" in out["reply"].lower()


def test_answer_coverage_no_double_when_already_mentioned():
    # If the reply already states the eval number, don't append it again.
    out = _loop([
        "<tool>eval depth=18",
        "<tool>best_move top=3",
        "Position is 0.00 (equal); best is e4, then d4, c4.",
    ]).respond([], "best moves and the eval")
    assert out["reply"].count("0.00") == 1           # not duplicated


# --- consumer C: grounding is plugin-aware (not chess-required-only) ---

_LIFE_PC = {"installed": ["life-skills"], "enabled": ["life-skills"], "marketplace": []}


def _life_loop(steps):
    return CoachLoop(ScriptedModel(steps), ToolExecutor(Game(), None, _LIFE_PC),
                     plugin_context=_LIFE_PC)


def test_dropped_plugin_result_is_grounded():
    # The breathing/convert transcript bug: a plugin tool returns a real result, the final
    # answer DROPS its number. matched_calls is chess-only so `required` is empty here — the
    # grounding must still fire for the executed plugin tool (domain-neutral).
    out = _life_loop([
        "<tool>convert_units value=5 from_unit=miles to_unit=km",
        "I converted that for you.",                  # drops the 8.047 result
    ]).respond([], "how many km is 5 miles?")
    assert _names(out) == ["convert_units"]
    assert "8.047" in out["reply"]                    # grounded fact appended


def test_plugin_result_not_doubled_when_narrated():
    # If the model already cites the plugin result, don't append it again.
    out = _life_loop([
        "<tool>convert_units value=5 from_unit=miles to_unit=km",
        "5 miles is about 8.047 kilometers.",         # already grounded
    ]).respond([], "how many km is 5 miles?")
    assert out["reply"].count("8.047") == 1


def test_result_signal_delegates_to_generic_for_plugin_results():
    from backend.inference import _result_signal
    assert _result_signal("score: +0.44 pawns from white POV, depth=18") == "+0.44"  # chess unchanged
    assert _result_signal("convert: 5 miles = 8.047 kilometers (length)") == "8.047"  # delegated
    assert _result_signal("metronome_bpm: 120 bpm = 500.0 ms per beat") == "500.0"


def test_grounding_does_not_double_append_when_model_paraphrases_or_rounds():
    # Transcript bug: the model grounded NATURALLY ("60 seconds", "8.05 km") but the brittle
    # exact-substring check ("60s" not in "60 seconds"; "8.047" not in the rounded "8.05") thought
    # the fact was dropped and appended the RAW tool line -> "...8.05 kilometers. convert: 5 miles =
    # 8.047 kilometers (length)". A numeric-aware check must treat these as already-grounded.
    from backend.inference import _ensure_required_narrated
    reply = "You're set for 60 seconds — about 3 slow 4-7-8 breath cycles."
    res = "breathing_timer: 60s set — about 3 slow 4-7-8 breath cycle(s)."
    out = _ensure_required_narrated(reply, {}, ["<tool>breathing_timer seconds=60</tool>"], [res])
    assert out == reply and "breathing_timer:" not in out          # "60" present as "60 seconds"
    reply2 = "5 miles is about 8.05 kilometers."                    # model rounded 8.047 -> 8.05
    res2 = "convert: 5 miles = 8.047 kilometers (length)"
    out2 = _ensure_required_narrated(reply2, {}, ["<tool>convert_units value=5 from_unit=miles to_unit=km</tool>"], [res2])
    assert out2 == reply2 and "convert:" not in out2


def test_grounding_still_appends_a_genuinely_dropped_fact():
    # The leniency must NOT swallow a real drop: a reply with NO matching number still gets grounded.
    from backend.inference import _ensure_required_narrated
    reply = "Ready to try it?"
    res = "breathing_timer: 60s set — about 3 slow 4-7-8 breath cycle(s)."
    out = _ensure_required_narrated(reply, {}, ["<tool>breathing_timer seconds=60</tool>"], [res])
    assert out != reply and "60" in out


# --- consumer D: the chess deterministic layer is scoped to the chess domain ---

def test_chess_coverage_suppressed_for_ood_context():
    # "how am I doing with my taxes?" matches the chess `eval` trigger ("how am i doing"), but
    # with an OOD (life-skills) context the chess deterministic layer must NOT force-route eval
    # onto a non-chess turn. The trained model routes any domain on its own here.
    out = _life_loop([
        "To check your taxes, gather your forms and compare standard vs itemized deductions.",
    ]).respond([], "how am I doing with my taxes?")
    assert "eval" not in _names(out)
    assert out["tool_calls"] == []                 # nothing force-routed on the OOD turn


def test_chess_coverage_still_fires_in_chess_context():
    # Same trigger phrase, DEFAULT (chess) context -> eval IS force-routed. No regression: the
    # gate only suppresses the crutch when a non-chess bundle is enabled.
    out = _loop([
        "Let me see.",                             # stops without eval -> coverage backstop
        "Position noted.",
    ]).respond([], "how am I doing?")
    assert "eval" in _names(out)


def test_tool_as_skill_miss_is_coerced_and_grounded_without_model_recovery():
    # The dominant E4B miss mode = a tool emitted as a skill (<skill>list_pieces</skill>). BEFORE
    # coercion, a model that doesn't recover from the off-distribution corrective dead-ended on
    # duplicate_tool_call -> ungrounded non-answer. With verb-coercion, the tool RUNS on the first
    # step regardless of whether the (frozen) model recovers. list_pieces needs no engine.
    out = _loop([
        "<skill>list_pieces</skill>",          # tool-as-skill miss
        "You still have your pieces.",          # model does NOT re-emit the tool (non-compliant)
    ]).respond([], "which pieces do I have left?")
    assert _names(out) == ["list_pieces"]                       # coerced + executed (not load_skill)
    assert any(r.startswith("pieces:") for r in out["tool_results"])   # real grounded result
    assert "duplicate_tool_call" not in out["tool_results"]     # no dead-end


def test_chess_coverage_on_for_training_catalog_bundles_with_no_runtime_tools():
    # The gate keys off the RUNTIME tool surface, not bundle names. A val-style context lists
    # training catalog bundles (user-skills/synthetic-pack) that register NO runtime tools, so
    # the live surface is still pure-chess -> coverage MUST stay on (else every chess val row
    # would lose its backstop in the completion eval). Regression guard for the D refinement.
    syn_pc = {"installed": ["chess-official", "user-skills", "synthetic-pack"],
              "enabled": ["chess-official", "user-skills", "synthetic-pack"], "marketplace": []}
    out = CoachLoop(ScriptedModel(["Let me see.", "Position noted."]),
                    ToolExecutor(Game(), None, syn_pc), plugin_context=syn_pc).respond([], "how am I doing?")
    assert "eval" in _names(out)                    # crutch stays on for chess-only runtime surface


def test_dedup_is_by_full_call_not_name():
    # best_move with different args must BOTH run (name-dedup would have blocked the 2nd).
    out = _loop([
        "<tool>best_move depth=1",
        "<tool>best_move top=3 series=2",
        "Done.",
    ]).respond([], "show me moves")     # no required intent; pure model-driven
    assert _names(out) == ["best_move", "best_move"]
