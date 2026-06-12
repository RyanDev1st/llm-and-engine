"""True token streaming: when on_event is given and the model's generate accepts
on_token, respond() streams each generation's tokens out as `token` events so the UI
fills live (overlapping generation), instead of revealing a finished block."""
from backend.game import Game
from backend.inference import CoachLoop
from backend.tools import ToolExecutor


class StreamModel:
    """generate() accepts on_token (so inference detects streaming support) and emits
    the output word-by-word through it, mimicking llama.cpp token streaming."""
    def __init__(self, steps):
        self.steps = list(steps)
        self.i = 0

    def generate(self, messages, max_new_tokens, stop, on_token=None):
        out = self.steps[min(self.i, len(self.steps) - 1)]
        self.i += 1
        if on_token:
            for w in out.split(" "):
                on_token(w + " ")
        return out


def test_reply_streams_as_token_events():
    events = []
    loop = CoachLoop(StreamModel(["Hello there, ask me anything."]), ToolExecutor(Game(), None))
    out = loop.respond([], "hi", on_event=events.append)          # no required tools -> first gen is the reply
    toks = [e for e in events if e["type"] == "token"]
    assert toks, "expected live token events"
    assert "".join(e["text"] for e in toks).strip() == "Hello there, ask me anything."
    assert out["reply"] == "Hello there, ask me anything."


def test_tool_turn_streams_then_emits_tool_event():
    events = []
    # eval intent -> first gen is a tool decision (streamed), then the reply gen.
    loop = CoachLoop(StreamModel(["<tool>eval depth=18", "Roughly equal."]), ToolExecutor(Game(), None))
    loop.respond([], "what's the eval?", on_event=events.append)
    kinds = [e["type"] for e in events]
    assert "token" in kinds and "tool" in kinds          # both streams present
    # a tool event follows the streamed tool-decision tokens (frontend clears them then)
    assert kinds.index("tool") > kinds.index("token")


def test_no_on_event_means_no_streaming():
    loop = CoachLoop(StreamModel(["Hi."]), ToolExecutor(Game(), None))
    out = loop.respond([], "hi")                          # no on_event -> plain, no token plumbing
    assert out["reply"] == "Hi." and "trace" not in out
