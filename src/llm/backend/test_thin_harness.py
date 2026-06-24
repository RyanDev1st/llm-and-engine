"""S1 — the CHESS_THIN_HARNESS flag. When ON, the loop drops the deterministic RESCUE layer
built for the weak E2B (coverage force-routing, the per-turn self-verify probe, the ask-back
re-gen) and trusts the model like a top harness. These tests prove the flag flips exactly those
three behaviors and nothing else, using scripted models so no GPU is needed.

See docs/findings/2026-06-24-harness-vs-claude-code-codex.md (S1)."""
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
    names = []
    for c in out["tool_calls"]:
        n = parse_call(c)[0]
        if n is None and "<skill>" in c:
            n = "load_skill"
        names.append(n)
    return names


def _loop(steps):
    return CoachLoop(ScriptedModel(steps), ToolExecutor(Game(), None))


# --- #17 coverage force-routing -------------------------------------------------

def test_coverage_force_routes_when_thin_off():
    # Baseline (flag default OFF): "how am I doing?" -> required {eval}; the model stops without
    # it, so coverage force-routes eval. This is the rescue behavior the flag will disable.
    out = _loop(["Just play e4.", "Okay, noted."]).respond([], "how am I doing?")
    assert "eval" in _names(out)


def test_coverage_not_forced_when_thin_on(monkeypatch):
    # Flag ON: the same stop-early model is TRUSTED — no eval is force-routed, the reply stands.
    monkeypatch.setattr("backend.inference._THIN_HARNESS", True)
    out = _loop(["Just play e4."]).respond([], "how am I doing?")
    assert out["tool_calls"] == [] and out["reply"] == "Just play e4."


# --- #19 per-turn self-verify probe + #23 deflection re-gen ----------------------

_DEFLECTION_RUN = [
    "<skill>chess-coach",                                    # load context only
    "I can help with tactics and openings — what would you like to work on?",  # deflect
    "<tool>eval depth=18",                                    # what the self-verify probe pulls in
    "All set — the position is equal.",
]


def test_self_verify_pulls_in_a_tool_when_thin_off():
    # Flag OFF: after a skill-load-only turn that deflects, the self-verify probe runs and the
    # model's "next tool" (eval) gets executed — the rescue behavior.
    out = _loop(_DEFLECTION_RUN).respond([], "coach me")
    assert _names(out) == ["load_skill", "eval"]


def test_deflection_accepted_when_thin_on(monkeypatch):
    # Flag ON: the self-verify probe is gated off, so the deflection is accepted as the reply and
    # no hidden extra generation pulls in a tool (lower latency, trust the model).
    monkeypatch.setattr("backend.inference._THIN_HARNESS", True)
    out = _loop(_DEFLECTION_RUN).respond([], "coach me")
    assert _names(out) == ["load_skill"]
    assert out["reply"].startswith("I can help with tactics")


# --- guards that must SURVIVE thin mode -----------------------------------------

def test_thin_keeps_reload_nudge_and_does_not_crash(monkeypatch):
    # Thin mode must still break a reload loop (a kept genuine-4B guard): the model loads the same
    # skill twice, gets nudged, then answers — it does not spin or error.
    monkeypatch.setattr("backend.inference._THIN_HARNESS", True)
    out = _loop([
        "<skill>chess-coach",
        "<skill>chess-coach",            # repeat -> reload nudge, not a re-execution
        "Here's the plan: develop and castle.",
    ]).respond([], "coach me")
    assert _names(out) == ["load_skill"]            # the duplicate load was not recorded again
    assert "develop" in out["reply"]
