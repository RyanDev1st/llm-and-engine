"""The spec section-4 turn loop: decide -> (execute tool) -> narrate.

The model uses the role of the latest message to pick Mode 1 vs Mode 2; we keep
the same system prompt across phases. A `ModelBackend` just needs generate()."""
from __future__ import annotations

from typing import Protocol

from llm_dataset.v1.catalog import official_tools
from llm_training.system_prompt import build_system

from .context_window import ContextWindow, WindowConfig, estimate_tokens
from .skills import load_skills
from .tool_hints import routing_hints, skill_hints, matched_calls
from .toolfmt import parse_call
from .tools import ToolExecutor

MAX_TOOL_CALLS = 8  # headroom for the coverage "Wait" nudges (each can cost a step)
DEFAULT_N_CTX = 4096  # used only if a backend can't report its own context limit
# Serve == train: the catalog of installed skills + the official tool manifest are
# rendered by the SAME build_system() the loader uses. Skills appear by name +
# description only (progressive disclosure); load_skill pulls the body on demand.
PLUGIN_CONTEXT = {"installed": ["chess-official"], "enabled": ["chess-official"], "marketplace": []}


class ModelBackend(Protocol):
    def generate(self, messages: list[dict], max_new_tokens: int, stop: list[str]) -> str:
        ...


class AdapterView:
    """Wraps an HFModel so a CoachLoop runs it with the LoRA adapter on (our SFT)
    or off (the untrained base) — same weights, for the side-by-side demo."""

    def __init__(self, model, use_adapter: bool) -> None:
        self.model = model
        self.use_adapter = use_adapter

    def generate(self, messages: list[dict], max_new_tokens: int, stop: list[str]) -> str:
        return self.model.generate(messages, max_new_tokens, stop, use_adapter=self.use_adapter)

    def count_tokens(self, text: str) -> int:
        return self.model.count_tokens(text)

    def context_limit(self) -> int:
        return self.model.context_limit()


def _build_window(model: "ModelBackend") -> ContextWindow:
    """Wire the real tokenizer into the budget when the backend exposes one;
    fall back to a chars/4 estimate so fakes/tests still work."""
    count = getattr(model, "count_tokens", None) or estimate_tokens
    limit_fn = getattr(model, "context_limit", None)
    n_ctx = limit_fn() if callable(limit_fn) else DEFAULT_N_CTX
    return ContextWindow(count, WindowConfig(n_ctx=n_ctx))


def contains_tool_call(text: str) -> bool:
    # Gemma natively emits <tool_code>…</tool_code>; treat it as a tool call too.
    return any(t in text for t in ("<tool>", "</tool>", "<tool_code>", "</tool_code>"))


def narrate_tool_result(tool_result: str) -> str:
    text = tool_result.strip()
    if text.startswith("error: illegal"):
        reason = text.partition("reason=")[2] or "that move is not legal here"
        return f"I can't make that move: {reason}. Try a legal move from the current position."
    if text.startswith("error: ambiguous"):
        return "That move is ambiguous. Please include the piece or square so I can tell which move you mean."
    if text.startswith("error: invalid_syntax"):
        return "I couldn't read that request as a valid chess action. Try phrasing it with a move or square."
    if text.startswith("error: engine_unavailable") or text.startswith("error: timeout"):
        return "I can't analyze that position right now — analysis is unavailable. Try again in a moment."
    if text.startswith("error: duplicate_tool_call"):
        return "I already asked for that exact tool result, so I'll answer from what I have instead."
    if "is a skill, not a tool" in text:
        return "Let me load the right skill and try again."
    if text.startswith("error: invalid_fen"):
        return "That FEN doesn't look valid — check the layout and side-to-move and try again."
    if text.startswith("error: no moves to review"):
        return "There's no move to review yet — make a move first."
    if text.startswith("error: unknown_skill"):
        return "I couldn't find that skill. Let me work with what I have."
    if text.startswith("error:"):  # never leak a raw internal error to the user
        return "That didn't work — let me try a different approach."
    if text.startswith("board_state:"):
        return f"Current board snapshot: {text.removeprefix('board_state:').strip()}."
    if text.startswith("best_line:"):
        line = text.removeprefix("best_line:").strip()
        return f"The line to play is {line}. That gives you the clearest plan from here."
    if text.startswith("best_moves:"):
        moves = text.removeprefix("best_moves:").strip()
        return f"The strongest tries here are {moves}."
    if text.startswith("best:"):
        move = text.removeprefix("best:").strip()
        return f"{move} is the move here."
    if text.startswith("score:"):
        score = text.removeprefix("score:").strip()
        return f"The position stands at {score} (positive favours White, negative Black)."
    if text.startswith("review:"):
        return f"Move review: {text.removeprefix('review:').strip()}."
    if text.startswith("threats:"):
        return f"Threat check: {text.removeprefix('threats:').strip()}."
    if text.startswith("legal_moves:"):
        return f"Legal moves: {text.removeprefix('legal_moves:').strip()}."
    if text.startswith("pieces:"):
        return f"Pieces: {text.removeprefix('pieces:').strip()}."
    if text.startswith("ok:") or text.startswith("success:"):
        return text.partition(":")[2].strip().capitalize()
    return text or "I ran the chess tool, but it did not return a readable result."


def _fallback_reply(tool_calls: list[str], tool_results: list[str]) -> str:
    """When the model gives an empty/leaked final reply, narrate the last FACT
    result. Skip load_skill bodies (a skill's markdown is not a user-facing fact)."""
    for call, res in zip(reversed(tool_calls), reversed(tool_results)):
        if "load_skill" in call:
            continue
        return narrate_tool_result(res)
    return "What would you like to look at on the board?"


def _result_signal(result: str) -> str | None:
    """A short distinctive token the reply should contain IF it narrated this fact
    (the eval number, the top move's SAN, the review label). Returns None when we
    can't extract one — then we don't risk a spurious append."""
    import re as _r
    t = result.strip()
    if t.startswith("score:"):
        m = _r.search(r"[-+]?\d+\.\d+", t)          # the eval number, e.g. 0.00 / +0.44
        return m.group(0) if m else None
    if t.startswith(("best:", "best_line:", "best_moves:")):
        m = _r.search(r"[KQRBN]?[a-h]?x?[a-h][1-8]", t)  # the first SAN move
        return m.group(0) if m else None
    if t.startswith("review:"):
        m = _r.search(r"label=(\w+)", t)
        return m.group(1) if m else None
    return None


def _ensure_required_narrated(reply: str, required: dict, tool_calls: list[str],
                              tool_results: list[str]) -> str:
    """Answer-coverage: tool-coverage guarantees the required tools RAN; this
    guarantees their results are REFLECTED in the reply. For each required tool
    whose fact the reply doesn't mention, append a grounded one-liner from the real
    tool output (never fabricated). Fixes the model gathering eval then dropping it
    from the answer on a compound request."""
    if not required:
        return reply
    low = reply.lower()
    additions: list[str] = []
    for name in required:
        res = next((r for c, r in zip(reversed(tool_calls), reversed(tool_results))
                    if (parse_call(c)[0] or "") == name), None)
        if not res or res.startswith("error"):
            continue
        sig = _result_signal(res)
        if sig and sig.lower() not in low:          # required fact missing from the reply
            additions.append(narrate_tool_result(res))
    if not additions:
        return reply
    facts = " ".join(additions)
    import re as _r
    r = reply.rstrip()
    if r.endswith("?"):  # slot the fact BEFORE the trailing guiding question, not after it
        parts = _r.split(r"(?<=[.!?])\s+", r)
        head = " ".join(parts[:-1])
        return (head + " " if head else "") + facts + " " + parts[-1]
    return r + " " + facts


_NUM = __import__("re").compile(r"[-+]?\d+\.\d+")  # signed decimal (eval-like; ignores depth=18 etc.)


def _eval_numbers(tool_results: list[str]) -> tuple[list[float], list[str]]:
    """(every numeric value that legitimately appears in ANY tool result, as floats;
    the rendered token of each `score:` eval, e.g. '+0.37'). The first set is what the
    reply is ALLOWED to contain; the second identifies the single eval to correct toward."""
    true_nums: list[float] = []
    eval_tokens: list[str] = []
    for res in tool_results:
        for m in _NUM.finditer(res):
            true_nums.append(float(m.group(0)))
        if res.strip().startswith("score:"):
            m = _NUM.search(res)
            if m:
                eval_tokens.append(m.group(0))
    return true_nums, eval_tokens


def _correct_eval_number(reply: str, tool_results: list[str]) -> str:
    """Number-consistency guard: if the reply states an eval-like number that matches NO
    real tool number (i.e. the model fabricated it — coarse quant like Q4_0 can do this)
    and there is exactly ONE eval result, replace that fabricated number with the real
    eval value. Conservative: acts only on a SINGLE unmatched number against a SINGLE eval
    source — 0 unmatched means nothing's wrong; >1 is ambiguous, so we never guess. Legit
    numbers the model quoted from any tool (e.g. best_move scores) are in true_nums and so
    are never touched."""
    true_nums, eval_tokens = _eval_numbers(tool_results)
    if len(eval_tokens) != 1:
        return reply  # no single eval to correct toward -> leave it
    bad = [(m.start(), m.end()) for m in _NUM.finditer(reply)
           if not any(abs(float(m.group(0)) - t) <= 0.01 for t in true_nums)]
    if len(bad) != 1:
        return reply  # nothing fabricated, or too ambiguous to correct safely
    s, e = bad[0]
    return reply[:s] + eval_tokens[0] + reply[e:]


def normalize_tool_call(text: str) -> str:
    # The trained shape is an optional lead-in sentence then ONE <tool>; the loop
    # stops at "</tool>", so close the tag if the stop trimmed it.
    call = text.strip()
    if "<tool>" in call and "</tool>" not in call:
        call += "</tool>"
    return call


import re as _re

# Known tool names, longest-first so e.g. best_move matches before a prefix.
_TOOL_NAMES = sorted({t["name"] for t in official_tools()} | {"load_skill"}, key=len, reverse=True)
_NAME_ALT = "|".join(_re.escape(n) for n in _TOOL_NAMES)
# </tool> always closes a call; the bare > / end-of-string forms (for a stop-
# trimmed call like "<move san=b3") only count when NOT followed by a word char,
# so prose like "<eval>ated badly" can't be mistaken for an eval call.
_MALFORMED = _re.compile(r"<(" + _NAME_ALT + r")\b([^<>]*?)(?:</tool>|(?:>|$)(?!\w))")
# Echo of a hint example with the OPENING <tool> dropped, e.g. the model copies
# "move san=Nf3</tool>" from the routing hint. Anchor the name at a token boundary
# so "remove</tool>" / "improve" can't false-match.
_ECHO = _re.compile(r"(?:^|[\s:>\"'`])(" + _NAME_ALT + r")\b([^<>]*)</tool>")
# Bare call with NO tags at all — the whole reply is a tool name + at least one
# k=v arg, e.g. "review_move depth=1". Requiring args keeps prose ("undo that move")
# and one-word replies from false-matching; a tool name followed only by k=v pairs
# is unambiguously a leaked call, so recover + execute it instead of showing it.
_BARE = _re.compile(r"^(" + _NAME_ALT + r")((?:\s+\w+=\S+)+)$")


def extract_call(decision: str) -> str | None:
    """Return a canonical '<tool>NAME args</tool>' (lead-in preserved) if `decision`
    is — even malformedly — a tool call, else None for a plain final reply.

    A small model sometimes drops the <tool> wrapper and uses the tool name as the
    tag, e.g. 'I will play b3. <move san=b3</tool>'. Recover that to the canonical
    form so the move actually executes instead of leaking into the chat."""
    # Gemma natively wraps calls in <tool_code>…</tool_code>; the harness speaks
    # <tool>. Map it so those calls execute instead of leaking into the reply.
    s = decision.strip().replace("<tool_code>", "<tool>").replace("</tool_code>", "</tool>")
    # Some variants emit a channel-token form, e.g. "<|tool_call>call:board_state fields=all"
    # (seen live). Strip the <|...|> channel tokens and a leading "call:" so the bare-name /
    # malformed recovery below can canonicalize it instead of it leaking as the reply.
    if "<|" in s or "call:" in s:
        s = _re.sub(r"<\|[^>]*\|?>", "", s)
        s = _re.sub(r"\bcall:\s*", "", s).strip()
    if "<tool>" in s:
        return normalize_tool_call(s)
    m = _MALFORMED.search(s)
    if m:
        name, rest = m.group(1), m.group(2).strip()
        canon = f"<tool>{name}{(' ' + rest) if rest else ''}</tool>"
        return (s[:m.start()] + canon + s[m.end():]).strip()
    e = _ECHO.search(s)  # closing-tag-only echo of a hint example, opening <tool> dropped
    if e:
        name, rest = e.group(1), e.group(2).strip()
        canon = f"<tool>{name}{(' ' + rest) if rest else ''}</tool>"
        return (s[:e.start(1)] + canon).strip()
    b = _BARE.match(s)  # whole reply IS a tagless tool call, e.g. "review_move depth=1"
    if b:
        return f"<tool>{b.group(1)}{b.group(2).rstrip()}</tool>"
    return None


def serving_skills_index() -> list[dict]:
    """All installed SKILL.md as catalog entries (name + description only)."""
    return [
        {"name": skill.name, "description": skill.description,
         "plugin": "chess-official", "source": "official_plugin", "enabled": True}
        for skill in load_skills()
    ]


def build_system_prompt(agent_overlay: str = "", plugin_context: dict | None = None, game=None) -> str:
    base = build_system(serving_skills_index(), official_tools(),
                        plugin_context or PLUGIN_CONTEXT, agent_overlay)
    from . import plugins  # prompt-start hooks: pre-load always-on plugin context
    hook = plugins.prompt_start({"game": game})
    return base + (("\n\n" + hook) if hook else "")


class CoachLoop:
    def __init__(self, model: ModelBackend, executor: ToolExecutor, agent_overlay: str = "",
                 plugin_context: dict | None = None) -> None:
        self.model = model
        self.executor = executor
        self.agent_overlay = agent_overlay
        self.plugin_context = plugin_context
        self.window = _build_window(model)

    def respond(self, history: list[dict], user_message: str, coverage: bool = True,
                on_event=None) -> dict:
        """history: prior user/assistant turns (no system). Returns the reply,
        display fields (tool_call, tool_result), and the context-window stats.

        The session memory is bounded here: `window.fit` evicts the oldest turns
        so the prompt stays inside the model's token budget — recent context is
        kept, old context is dropped, the window never overflows.

        `coverage` (default on) is the reliability layer: the user's words are mapped
        to a deterministic set of REQUIRED tools, and the loop will not accept a final
        reply while one is ungathered — it first injects a "Wait, you still need X"
        steer (s1-style budget forcing; the model usually complies and looks smart),
        then force-routes the tool as a backstop. This is what makes the model do
        multi-tool reliably in one session. Set coverage=False for the ablation."""
        # Deterministic routing layer: if the user's words clearly map to a tool,
        # remind the model of it explicitly (fixes small-model routing slips like
        # narrating "I'll play b3" without calling move, or stopping before eval).
        game_over = self.executor.game.over_status()
        system = build_system_prompt(self.agent_overlay, self.plugin_context,
                                     self.executor.game) + routing_hints(user_message, game_over)
        if not game_over:  # on a finished game, state the result — don't spin up a skill
            system += skill_hints(user_message, serving_skills_index())
        # Coverage set: tool -> canonical call for each detected intent. Empty on a
        # finished game or when coverage is off.
        required = {} if (game_over or not coverage) else matched_calls(user_message)
        kept_history, ctx_stats = self.window.fit(system, history, user_message)
        convo = [{"role": "system", "content": system}, *kept_history,
                 {"role": "user", "content": user_message}]
        new_turns = [{"role": "user", "content": user_message}]
        tool_calls: list[str] = []
        tool_results: list[str] = []
        seen_calls: set[str] = set()   # full <tool> spans, for dedup (same call never re-runs)
        seen_names: set[str] = set()   # tool NAMES gathered, for coverage (best_move != best_move top=3)

        for _ in range(MAX_TOOL_CALLS):
            raw = self.model.generate(convo, max_new_tokens=96, stop=["</tool>", "</tool_code>"]).strip()
            decision = extract_call(raw)  # canonical call (recovers a dropped <tool> wrapper) or None
            if decision is None:  # the model wants to give the final reply
                outstanding = [t for t in required if t not in seen_names]
                if not outstanding:
                    # All required intents covered. An empty final reply after tools ran
                    # would show a blank bubble — narrate the last fact instead.
                    reply = raw if raw else _fallback_reply(tool_calls, tool_results)
                    # Number guard FIRST (fix a fabricated eval number -> the real one),
                    # then answer-coverage (append any required fact still missing). Order
                    # matters: a corrected number then reads as present, so it isn't doubled.
                    reply = _correct_eval_number(reply, tool_results)
                    reply = _ensure_required_narrated(reply, required, tool_calls, tool_results)
                    new_turns.append({"role": "assistant", "content": reply})
                    return {
                        "reply": reply,
                        "tool_call": tool_calls[-1] if tool_calls else None,
                        "tool_result": tool_results[-1] if tool_results else None,
                        "tool_calls": tool_calls,
                        "tool_results": tool_results,
                        "turns": new_turns,
                        "context": ctx_stats.as_payload(),
                    }
                # A required intent is still ungathered: force-route it directly (deterministic
                # coverage). We don't spend a generation "nudging" the model to call it — on a
                # small model that retry is usually wasted; the model still does multi-tool on
                # its own when proactive, and this guarantees the rest without the latency.
                decision = required[outstanding[0]]
            # Dedup on the call itself, not the full text — a differing lead-in
            # ("Let me check" vs "I'll look") must not let the same call re-run and
            # re-hit the engine. Key = the <tool>…</tool> span.
            i0 = decision.find("<tool>")
            key = decision[i0:] if i0 >= 0 else decision
            tool_result = "error: duplicate_tool_call" if key in seen_calls else self.executor.execute(decision)
            seen_calls.add(key)
            name = parse_call(decision)[0] or ""
            if name:
                seen_names.add(name)
            tool_calls.append(decision)
            tool_results.append(tool_result)
            if on_event:  # live progress: surface each tool step as it completes (streaming UI)
                on_event({"type": "tool", "name": name, "call": decision, "result": tool_result})
            convo += [{"role": "assistant", "content": decision},
                      {"role": "tool", "content": tool_result}]
            new_turns += [{"role": "assistant", "content": decision},
                          {"role": "tool", "content": tool_result}]
            if tool_result == "error: duplicate_tool_call":
                break

        # Budget forcing (s1): out of tool steps, the user is waiting — answer now.
        convo.append({"role": "user", "content":
                      "You're out of tool steps and the user is waiting — give your best answer now using the results you have."})
        reply = self.model.generate(convo, max_new_tokens=160, stop=[]).strip()
        if contains_tool_call(reply) or not reply:
            # leaked a tool tag, or produced nothing — narrate the last fact so the
            # user never sees a raw tag or an empty bubble.
            reply = _fallback_reply(tool_calls, tool_results)
        reply = _correct_eval_number(reply, tool_results)   # fix a fabricated eval number first
        reply = _ensure_required_narrated(reply, required, tool_calls, tool_results)
        new_turns.append({"role": "assistant", "content": reply})
        return {
            "reply": reply,
            "tool_call": tool_calls[-1] if tool_calls else None,
            "tool_result": tool_results[-1] if tool_results else None,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "turns": new_turns,
            "context": ctx_stats.as_payload(),
        }
