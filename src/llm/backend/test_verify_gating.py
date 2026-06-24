"""LATENCY: the self-verify probe (_verify_fulfilled) is a FULL extra generation that was firing on
EVERY skill-load turn. It now only fires when the draft LOOKS like a non-answer (deflection / ask-back
/ very short). A confident substantive answer skips it — one fewer decode per well-behaved coach turn.
These tests pin BOTH halves: the fast path skips the probe, and a real deflection still gets caught."""
from backend.game import Game
from backend.inference import CoachLoop
from backend.tools import ToolExecutor


def test_good_substantive_answer_skips_the_verify_probe():
    # coverage=True (the live default), but the prompt maps to NO required tool, so this is a genuine
    # skill-load-only turn. A long, specific answer must NOT trigger the Self-check probe — the
    # optimization: trust it, save the decode.
    GOOD = ("Good coaching starts with king safety, then piece activity, then pawn structure. "
            "Castle early, develop toward the centre, and avoid moving the same piece twice in the opening.")
    probed = {"n": 0}

    class M:
        def __init__(self): self.i = 0
        def generate(self, messages, mx, stop):
            last = messages[-1]["content"] if messages else ""
            if "Self-check" in last:
                probed["n"] += 1
                return "DONE"
            self.i += 1
            return "<skill>chess-coach</skill>" if self.i == 1 else GOOD

    out = CoachLoop(M(), ToolExecutor(Game(), None)).respond([], "coach me on my general strategy")
    assert out["reply"] == GOOD
    assert probed["n"] == 0          # the probe (an extra generation) never ran


def test_deflection_still_triggers_the_probe_and_is_caught():
    # A capability blurb after a skill load IS suspicious -> the probe still runs and the deflection
    # guard still forces a real answer. The optimization must not weaken this.
    DEFLECT = "I'm here to help with your game — tactics, positions, or planning. What's on your mind?"
    REAL = "You are playing White. Develop your knights and claim the centre with a pawn."
    probed = {"n": 0}

    class M:
        def __init__(self): self.i = 0
        def generate(self, messages, mx, stop):
            last = messages[-1]["content"] if messages else ""
            if "answer my question directly" in last.lower():
                return REAL
            if "Self-check" in last:
                probed["n"] += 1
                return "DONE"
            self.i += 1
            return "<skill>chess-coach</skill>" if self.i == 1 else DEFLECT

    out = CoachLoop(M(), ToolExecutor(Game(), None)).respond([], "what can you do for me?")
    assert out["reply"] == REAL          # deflection caught + replaced
    assert probed["n"] == 1              # the probe DID run on the suspicious draft
