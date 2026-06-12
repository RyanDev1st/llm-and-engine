"""The spec section-4 turn loop: decide -> (execute tool) -> narrate.

The model uses the role of the latest message to pick Mode 1 vs Mode 2; we keep
the same system prompt across phases. A `ModelBackend` just needs generate()."""
from __future__ import annotations

from typing import Protocol

from llm_dataset.v1.catalog import official_tools
from llm_training.system_prompt import build_system

from .context_window import ContextWindow, WindowConfig, estimate_tokens
from .skills import load_skills
from .tool_hints import routing_hints
from .tools import ToolExecutor

MAX_TOOL_CALLS = 6
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


def extract_call(decision: str) -> str | None:
    """Return a canonical '<tool>NAME args</tool>' (lead-in preserved) if `decision`
    is — even malformedly — a tool call, else None for a plain final reply.

    A small model sometimes drops the <tool> wrapper and uses the tool name as the
    tag, e.g. 'I will play b3. <move san=b3</tool>'. Recover that to the canonical
    form so the move actually executes instead of leaking into the chat."""
    # Gemma natively wraps calls in <tool_code>…</tool_code>; the harness speaks
    # <tool>. Map it so those calls execute instead of leaking into the reply.
    s = decision.strip().replace("<tool_code>", "<tool>").replace("</tool_code>", "</tool>")
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
    return None


def serving_skills_index() -> list[dict]:
    """All installed SKILL.md as catalog entries (name + description only)."""
    return [
        {"name": skill.name, "description": skill.description,
         "plugin": "chess-official", "source": "official_plugin", "enabled": True}
        for skill in load_skills()
    ]


def build_system_prompt(agent_overlay: str = "", plugin_context: dict | None = None) -> str:
    return build_system(serving_skills_index(), official_tools(),
                        plugin_context or PLUGIN_CONTEXT, agent_overlay)


class CoachLoop:
    def __init__(self, model: ModelBackend, executor: ToolExecutor, agent_overlay: str = "",
                 plugin_context: dict | None = None) -> None:
        self.model = model
        self.executor = executor
        self.agent_overlay = agent_overlay
        self.plugin_context = plugin_context
        self.window = _build_window(model)

    def respond(self, history: list[dict], user_message: str) -> dict:
        """history: prior user/assistant turns (no system). Returns the reply,
        display fields (tool_call, tool_result), and the context-window stats.

        The session memory is bounded here: `window.fit` evicts the oldest turns
        so the prompt stays inside the model's token budget — recent context is
        kept, old context is dropped, the window never overflows."""
        # Deterministic routing layer: if the user's words clearly map to a tool,
        # remind the model of it explicitly (fixes small-model routing slips like
        # narrating "I'll play b3" without calling move, or stopping before eval).
        game_over = self.executor.game.over_status()
        system = build_system_prompt(self.agent_overlay, self.plugin_context) + routing_hints(user_message, game_over)
        kept_history, ctx_stats = self.window.fit(system, history, user_message)
        convo = [{"role": "system", "content": system}, *kept_history,
                 {"role": "user", "content": user_message}]
        new_turns = [{"role": "user", "content": user_message}]
        tool_calls: list[str] = []
        tool_results: list[str] = []
        seen_calls: set[str] = set()

        for _ in range(MAX_TOOL_CALLS):
            raw = self.model.generate(convo, max_new_tokens=96, stop=["</tool>", "</tool_code>"]).strip()
            decision = extract_call(raw)  # canonical call (recovers a dropped <tool> wrapper) or None
            if decision is None:  # no tool call -> this is the final reply
                # An empty final reply after tools ran (e.g. board_state+load_skill then
                # nothing) would show a blank bubble — narrate the last fact instead.
                reply = raw if raw else _fallback_reply(tool_calls, tool_results)
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
            # Dedup on the call itself, not the full text — a differing lead-in
            # ("Let me check" vs "I'll look") must not let the same call re-run and
            # re-hit the engine. Key = the <tool>…</tool> span.
            i0 = decision.find("<tool>")
            key = decision[i0:] if i0 >= 0 else decision
            tool_result = "error: duplicate_tool_call" if key in seen_calls else self.executor.execute(decision)
            seen_calls.add(key)
            tool_calls.append(decision)
            tool_results.append(tool_result)
            convo += [{"role": "assistant", "content": decision},
                      {"role": "tool", "content": tool_result}]
            new_turns += [{"role": "assistant", "content": decision},
                          {"role": "tool", "content": tool_result}]
            if tool_result == "error: duplicate_tool_call":
                break

        reply = self.model.generate(convo, max_new_tokens=160, stop=[]).strip()
        if contains_tool_call(reply) or not reply:
            # leaked a tool tag, or produced nothing — narrate the last fact so the
            # user never sees a raw tag or an empty bubble.
            reply = _fallback_reply(tool_calls, tool_results)
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
