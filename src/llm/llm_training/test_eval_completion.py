"""The completion-grading rubric (eval_completion.grade) — scored offline with SYNTHETIC loop
results (no model/GPU). It converts the loop's output into the task-completion metrics the peer
reviews said routing accuracy misses: completed, grounded, and especially `recovered` (a wrong
first route that the loop self-corrects to a grounded answer — a WIN strict first-action scoring
records as a loss)."""
from llm_training.eval_completion import grade, run_completion


def _row(user, gold, expected=None, slice="X", mode=""):
    r = {"slice": slice, "reasoning_mode": mode,
         "messages": [{"role": "user", "content": user},
                      {"role": "assistant", "content": gold}]}
    if expected is not None:
        r["expected_tool_calls"] = expected
    return r


def _res(calls, results, reply):
    return {"tool_calls": calls, "tool_results": results, "reply": reply}


def test_perfect_chess_completion():
    row = _row("play e4", "<tool>board_state fields=all</tool>", expected=["board_state", "move"])
    res = _res(["<tool>board_state fields=all</tool>", "<tool>move san=e4</tool>"],
               ["board_state: turn=white, legal_count=20", "success: played e4"], "I played e4 for you.")
    g = grade(row, res)
    assert g["first_ok"] and g["completed"] and g["exec_ok"] and g["args_ok"]
    assert not g["recovered"]                       # routed right first try -> not a recovery


def test_plugin_tool_completion_grounded():
    row = _row("how many km is 5 miles?", "<tool>convert_units value=5 from_unit=miles to_unit=km</tool>")
    res = _res(["<tool>convert_units value=5 from_unit=miles to_unit=km</tool>"],
               ["convert: 5 miles = 8.047 kilometers (length)"], "5 miles is about 8.047 km.")
    g = grade(row, res)
    assert g["first_ok"] and g["completed"] and g["grounded"]


def test_dropped_grounding_is_caught():
    row = _row("how many km is 5 miles?", "<tool>convert_units value=5 from_unit=miles to_unit=km</tool>")
    res = _res(["<tool>convert_units value=5 from_unit=miles to_unit=km</tool>"],
               ["convert: 5 miles = 8.047 kilometers (length)"], "I converted that for you.")
    assert grade(row, res)["grounded"] is False     # reply dropped the 8.047 -> not grounded


def test_recovered_counts_a_corrected_first_route():
    # model emitted <skill>metronome_bpm> (wrong verb), got the corrective error, then called the
    # tool and grounded the answer. first_ok False, but the TURN succeeded -> recovered True.
    row = _row("set a metronome to 120 bpm", "<tool>metronome_bpm bpm=120</tool>")
    res = _res(["<skill>metronome_bpm</skill>", "<tool>metronome_bpm bpm=120</tool>"],
               ["error: 'metronome_bpm' is a tool, not a skill — call it with <tool>metronome_bpm bpm=<bpm></tool>",
                "metronome_bpm: 120 bpm = 500.0 ms per beat"], "120 bpm is 500.0 ms per beat.")
    g = grade(row, res)
    assert g["first_ok"] is False                   # first action was the wrong verb
    assert g["completed"] and g["grounded"] and g["recovered"]
    assert g["exec_ok"]                             # the LAST call for that tool succeeded


def test_decline_row_is_complete_when_answered_directly():
    row = _row("what is the capital of France?", "Paris is the capital of France.")
    g = grade(row, _res([], [], "Paris is the capital of France."))
    assert g["first_ok"] and g["completed"] and g["grounded"] and not g["recovered"]


def test_arg_error_fails_args_and_exec_ok():
    row = _row("make a move", "<tool>move san=e4</tool>", expected=["move"])
    res = _res(["<tool>move</tool>"], ["error: tool 'move' needs 'san=...' — e.g. <tool>move san=...</tool>"],
               "Tell me which move to play.")
    g = grade(row, res)
    assert g["args_ok"] is False and g["exec_ok"] is False


class _Scripted:
    """Per-call scripted model (resets nothing across one row's loop)."""
    def __init__(self, steps):
        self.steps = list(steps)
        self.i = 0

    def generate(self, messages, max_new_tokens, stop):
        out = self.steps[min(self.i, len(self.steps) - 1)]
        self.i += 1
        return out


def test_run_completion_wires_loop_to_rubric_on_an_ood_row():
    # End-to-end on CPU: a real CoachLoop runs over a life-skills row (engine=None), the executed
    # plugin tool returns a real result, and run_completion aggregates the rubric. Proves the
    # runner wiring, not just grade().
    pc = {"installed": ["life-skills"], "enabled": ["life-skills"], "marketplace": []}
    row = {"slice": "STRESS_ood_tool_clean", "reasoning_mode": "", "plugin_context": pc,
           "messages": [{"role": "user", "content": "convert 5 miles to km"},
                        {"role": "assistant",
                         "content": "<tool>convert_units value=5 from_unit=miles to_unit=km</tool>"}]}
    model = _Scripted(["<tool>convert_units value=5 from_unit=miles to_unit=km",
                       "5 miles is about 8.047 km."])
    res = run_completion(model, [row], engine=None, progress_every=0)
    assert res["n"] == 1
    assert res["totals"]["completed"] == 1 and res["totals"]["grounded"] == 1
    assert res["by_slice"]["STRESS_ood_tool_clean"]["n"] == 1
