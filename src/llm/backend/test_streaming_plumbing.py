"""Streaming plumbing: tokens must reach the UI live during the long T4 decode. The chain was broken
because AdapterView (the wrapper in the service path) dropped on_token, so CoachLoop.respond saw a
non-streaming model and the whole reply landed at once. These CPU tests (scripted model, no GPU) prove
on_token is forwarded and that respond() emits live `token` events when the backend supports it."""
from backend.game import Game
from backend.inference import AdapterView, CoachLoop
from backend.tools import ToolExecutor


class StreamModel:
    """A fake backend that streams: generate() emits the reply char-by-char via on_token."""
    def __init__(self, steps):
        self.steps = list(steps); self.i = 0

    def generate(self, messages, max_new_tokens, stop, use_adapter=True, on_token=None):
        out = self.steps[min(self.i, len(self.steps) - 1)]; self.i += 1
        if on_token is not None:
            for ch in out:
                on_token(ch)
        return out

    def count_tokens(self, text):
        return len(text)

    def context_limit(self):
        return 8192


def test_adapterview_forwards_on_token():
    seen = []
    av = AdapterView(StreamModel(["hello there"]), use_adapter=True)
    out = av.generate([{"role": "user", "content": "hi"}], 32, [], on_token=lambda t: seen.append(t))
    assert out == "hello there"
    assert "".join(seen) == "hello there"          # every token reached the callback


def test_respond_emits_live_token_events_when_backend_streams():
    events = []
    loop = CoachLoop(AdapterView(StreamModel(["You're doing well; develop a piece."]), True),
                     ToolExecutor(Game(), None))
    loop.respond([], "how am I doing?", coverage=False, on_event=events.append)
    tokens = [e for e in events if e.get("type") == "token"]
    assert tokens and "".join(t["text"] for t in tokens).startswith("You're doing well")


def test_respond_without_streaming_backend_still_works():
    # A model whose generate has no on_token param -> can_stream False -> no token events, still replies.
    class Plain:
        def generate(self, messages, max_new_tokens, stop):
            return "Plain reply."
    out = CoachLoop(Plain(), ToolExecutor(Game(), None)).respond([], "hi", coverage=False)
    assert out["reply"] == "Plain reply."
