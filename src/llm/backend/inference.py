"""The spec section-4 turn loop: decide -> (execute tool) -> narrate.

The model uses the role of the latest message to pick Mode 1 vs Mode 2; we keep
the same system prompt across phases. A `ModelBackend` just needs generate()."""
from __future__ import annotations

import re as _re
from typing import Protocol

from llm_dataset.v1.catalog import compute_tools, official_tools
from llm_training.system_prompt import build_system

from .context_window import ContextWindow, WindowConfig, estimate_tokens
from .skills import load_skills
from .tool_hints import routing_hints, skill_hints, matched_calls
from .toolfmt import parse_call
from .tools import ToolExecutor

MAX_TOOL_CALLS = 8  # headroom for the coverage "Wait" nudges (each can cost a step)
# Per-generation token budget. The same call detects a tool decision OR produces the
# final reply: tool calls stop early at </tool> (the stop seq), so this cap only ever
# bounds a genuine final reply. 96 was truncating longer answers mid-sentence; 320
# (~240 words) lets a coaching reply finish while still capping runaway generation.
REPLY_TOKENS = 320
DEFAULT_N_CTX = 4096  # used only if a backend can't report its own context limit
# One action per generation step (the contract). A tool call closes at </tool>
# (Gemma-native </tool_code>) and a skill load at </skill>; stopping at the FIRST
# close tag is what holds a step to exactly one action. </skill> was missing from
# the action stop list, so a skill-load generation ran on past the close tag and
# emitted a second <skill>/garbage tail — the auto-mode "same skill twice" + partial
# <skill> + missing </skill> artifacts. Every action-deciding gen uses this; only the
# terminal budget-forced reply (which must run to a full answer) keeps an empty stop.
ACTION_STOP = ["</tool>", "</tool_code>", "</skill>"]
# Serve == train: the catalog of installed skills + the official tool manifest are
# rendered by the SAME build_system() the loader uses. Skills appear by name +
# description only (progressive disclosure); load_skill pulls the body on demand.
# Installed plugin bundles; `enabled` are active (contribute tools+skills to the served
# surface). chess-official + openings + analysis on by default so cross-bundle routing is
# testable out of the box. The Plugins panel toggles `enabled` live.
PLUGIN_CONTEXT = {"installed": ["chess-official", "openings", "analysis", "puzzles"],
                  "enabled": ["chess-official", "openings", "analysis", "puzzles"],
                  "marketplace": ["endgame-trainer", "study-notes"]}


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
    if text.startswith("error: unknown_tool"):
        return "That tool isn't available here. Let me use one I have."
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


def _only_context(tool_calls: list[str]) -> bool:
    """The turn ran ONLY context-loading (load_skill) or nothing — no fact/board tool.
    Such a turn still owes the user a real answer; it's the gate for the answer-retry and
    the self-verification step (both fire only here, so normal/fact turns never pay)."""
    return (not tool_calls) or all("load_skill" in c for c in tool_calls)


def _fallback_reply(tool_calls: list[str], tool_results: list[str]) -> str:
    """When the model gives an empty/leaked final reply, narrate the last FACT
    result. Skip load_skill bodies (a skill's markdown is not a user-facing fact)."""
    for call, res in zip(reversed(tool_calls), reversed(tool_results)):
        if "load_skill" in call:
            continue
        return narrate_tool_result(res)
    return "What would you like to look at on the board?"


# A terminal reply that is JUST a tool-intent lead-in (the model narrated "I'll load X"
# / "let me check" then stopped without calling the tool) is a non-answer. If tools ran
# this turn, we narrate the real result instead of showing the dangling lead-in.
_LEADIN_ONLY = _re.compile(
    r"^(loading\b|let me (load|check|look|grab|pull|see)|i'?ll (load|check|use|look)"
    r"|first,?\s|now (i|let)|let me (get|find))\b[^.?!]*[.?!]?\s*$", _re.I)


def _is_leadin_only(reply: str) -> bool:
    r = (reply or "").strip()
    return bool(r) and len(r) <= 70 and bool(_LEADIN_ONLY.match(r))


# A DEFLECTION: after loading a skill the model answers with a generic capability blurb /
# ask-back ("I'm here to help with tactics, positions… what's on your mind?") instead of
# DOING what the user asked — it treats loading the skill as the task. This evades the whiff
# guard (it's fluent, not empty/lead-in) and the self-verify (the model judges its own blurb
# as DONE, and a teaching/identity ask has no tool to name). Detected DETERMINISTICALLY (no
# extra round-trip) by the generic redirect phrases — distinct from a trained guiding question,
# which addresses the SPECIFIC position ("want the attacking plan, or shore up the defense?")
# and carries none of these. Used only to gate the answer-now retry on a skill-load-only turn.
_DEFLECTION = _re.compile(
    r"what(?:'?s| is) on your mind"
    r"|what would you like"
    r"|what do you want to (?:do|work on|look at|focus)"
    r"|how can i (?:help|assist)"
    r"|what can i (?:help|do)\b"
    r"|i'?m here to (?:help|assist)"
    r"|here to help you"
    r"|let me know (?:what|how|if)"
    r"|feel free to (?:ask|let me)"
    r"|happy to help\b", _re.I)


def _is_deflection(reply: str) -> bool:
    """True when the reply is a generic capability blurb / ask-back rather than an answer
    (the post-skill-load deflection). Keyed on generic redirect phrases, NOT on ending with a
    question, so a position-specific guiding question is never mis-flagged."""
    return bool(_DEFLECTION.search(reply or ""))


# A leading sentence that ANNOUNCES a skill/tool action ("Loading the chess-coach
# skill. <real answer>"). Training never puts skill/tool narration in the final reply
# (finals.py: "skill loads / tool calls never appear here"), so strip a single such
# leading sentence when real content follows it. Requires an explicit skill/tool word
# so coaching prose ("Use your rook to...") is never touched. Whole-reply lead-ins are
# handled by _is_leadin_only instead.
_ANNOUNCE_LEAD = _re.compile(
    r"^\s*((?:i'?(?:ve|ll|m)?\s+)?(?:now\s+)?(?:just\s+|then\s+)?"
    r"(?:load(?:ing|ed)?|using|used?|call(?:ing|ed)?|invok(?:ing|ed)?|"
    r"let me (?:load|use|call|invoke|grab|pull))\b"
    r"[^.?!]*?\b(?:skill|skills|tool|tools|load_skill)\b[^.?!]*[.?!])\s+(?=\S)", _re.I)


def _strip_announce_leadin(reply: str) -> str:
    m = _ANNOUNCE_LEAD.match((reply or "").strip())
    if not m:
        return reply
    rest = (reply or "").strip()[m.end():].strip()
    return rest if rest else reply   # only strip when a real answer remains after it


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


# SAN move token (incl. castling). Used to compare the moves the reply names against the
# real moves a best_move tool returned, to catch fabricated move lists.
_SAN_TOK = __import__("re").compile(
    r"\b(?:O-O-O|O-O|[KQRBN][a-h1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|[a-h]x[a-h][1-8](?:=[QRBN])?|[a-h][1-8])\b")


def _best_move_sans(tool_results: list[str]) -> list[str] | None:
    """The REAL moves from the most recent best_move/best_line/best_moves result, in order.
    None if no such result this turn (then we don't touch move names)."""
    for res in reversed(tool_results):
        t = res.strip()
        if t.startswith(("best_line:", "best:", "best_moves:")):
            head = t.split(":", 1)[1].split(", score")[0].split("; best_line")[0]
            sans = _SAN_TOK.findall(head)
            return list(dict.fromkeys(sans)) if sans else None
    return None


def _correct_move_names(reply: str, tool_results: list[str]) -> str:
    """Move-name guard: a best_move tool ran, but the reply names moves that the engine
    did NOT return (the model fabricated a move list — e.g. tool gave 'e4 c6 d4', reply
    said 'Nf3, e4, Nc3'). Append a short grounded correction with the REAL moves so the
    truth is present and labeled. Conservative: fires only when a best_move result exists,
    the reply names ≥1 SAN, and ≥1 named SAN is NOT in the real set. Append-only — never
    mangles the sentence; the deterministic backstop the model's narration can't be trusted
    to do itself."""
    real = _best_move_sans(tool_results)
    if not real:
        return reply
    named = _SAN_TOK.findall(reply)
    if not named:
        return reply
    real_set = {m.rstrip("+#") for m in real}
    if all(m.rstrip("+#") in real_set for m in named):
        return reply  # every move the reply names is real -> nothing fabricated
    correction = " (Engine's actual moves: " + ", ".join(real) + ".)"
    return reply.rstrip() + correction


def normalize_tool_call(text: str) -> str:
    # The trained shape is an optional lead-in sentence then ONE <tool>; the loop
    # stops at "</tool>", so close the tag if the stop trimmed it.
    call = text.strip()
    if "<tool>" in call and "</tool>" not in call:
        call += "</tool>"
    return call


# Known tool names, longest-first so e.g. best_move matches before a prefix.
_TOOL_NAMES = sorted(
    {t["name"] for t in official_tools()} | {t["name"] for t in compute_tools()} | {"load_skill"},
    key=len, reverse=True)
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
    # <skill>NAME</skill> is the trained skill-load VERB. The executor still loads a
    # skill via the canonical 'load_skill name=NAME' tool, so translate the verb to
    # that internal form so it EXECUTES. Take the name from a legacy inner
    # "load_skill name=X" if present, else the first token.
    def _skill_sub(m: "_re.Match") -> str:
        inner = m.group(1).strip()
        mm = _re.search(r"name=([A-Za-z0-9_][A-Za-z0-9_-]*)", inner)
        name = mm.group(1) if mm else (inner.split()[0] if inner else "")
        return f"<tool>load_skill name={name}</tool>" if name else ""
    s = _re.sub(r"<skill>(.*?)</skill>", _skill_sub, s, flags=_re.S)
    s = _re.sub(r"<skill>\s*([A-Za-z0-9_][A-Za-z0-9_-]*)",   # close-tag dropped
                r"<tool>load_skill name=\1</tool>", s)
    s = s.replace("<skill>", "<tool>").replace("</skill>", "</tool>")
    # Strip model artifacts that pollute the call: channel tokens — ANY <...> containing a
    # pipe, e.g. "<|tool_call>", "<tool_call|>" (both seen live; <tool>/</tool> have no pipe
    # so they're left intact) — a leading "call:", and JSON-ish brace junk ({}, {...}). Seen
    # live: "load_skill name=chess-coach{}<tool_call|>" parsed the skill name as
    # "chess-coach{}<tool_call|>" -> unknown_skill. Cleaning these lets the real call run.
    if "|" in s or "call:" in s or "{" in s:
        s = _re.sub(r"<[^<>]*\|[^<>]*>", "", s)   # channel token with a pipe (either side)
        s = _re.sub(r"\bcall:\s*", "", s)
        s = _re.sub(r"\{[^{}]*\}", "", s)          # {} / {..} JSON artifact
        s = s.strip()
    # Hallucinated gerund/spacing variants of a tool name, e.g. "loading_skill" /
    # "load skill" instead of "load_skill" (seen leaking as the whole reply). Map the
    # known ones so the bare-call recovery below canonicalizes + executes them.
    s = _re.sub(r"\bloading[_ ]skill\b", "load_skill", s, flags=_re.I)
    s = _re.sub(r"\bload skill\b", "load_skill", s, flags=_re.I)
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


# Map the internal load_skill execution form BACK to the trained <skill>NAME</skill> verb,
# for conversation history (train==serve) and UI display. Preserves any lead-in text.
_LOAD_SKILL_CALL = _re.compile(r"<tool>\s*load_skill\s+name=([A-Za-z0-9_][A-Za-z0-9_-]*)\s*</tool>")


def _to_skill_verb(call: str) -> str:
    return _LOAD_SKILL_CALL.sub(lambda m: f"<skill>{m.group(1)}</skill>", call)


# --- PLAN-mode serve handling (Stage 1) --------------------------------------
# A <goal>/<plan> panel turn commits the objective + checklist but carries NO
# executable action, so the loop must surface it and KEEP working the boxes — not
# treat it as the final reply. A box is satisfied when its bound skill/tool runs.
_PLAN_TAG = _re.compile(r"<plan>(.*?)</plan>", _re.DOTALL)
_GOAL_TAG = _re.compile(r"<goal>(.*?)</goal>", _re.DOTALL)
_THINK_TAG = _re.compile(r"<think>(.*?)</think>", _re.DOTALL)
_PLAN_BOX = _re.compile(r"-\s*\[[ xX]\]\s*.+?\(([^)]+)\)")
_SKILL_ARG = _re.compile(r"name=([A-Za-z0-9_][A-Za-z0-9_-]*)")


def _split_reasoning(reply: str) -> tuple[str, list[str], str | None]:
    """Separate the user-visible answer from its reasoning trace. Returns
    (visible, think_blocks, goal). `<think>` (the decision trace) and `<goal>` (the
    objective) belong in the thinking / goal panels — NEVER the chat bubble. The trained
    VISIBLE reply is rendered exactly this way (multiturn turn-1 carries no <think>), so
    stripping them here keeps serve == the trained visible shape. Conservative: removes
    only the tag blocks, leaves all other prose intact."""
    text = reply or ""
    thinks = [m.group(1).strip() for m in _THINK_TAG.finditer(text) if m.group(1).strip()]
    gm = _GOAL_TAG.search(text)
    goal = gm.group(1).strip() if gm and gm.group(1).strip() else None
    visible = _GOAL_TAG.sub("", _THINK_TAG.sub("", text))
    visible = _re.sub(r"\n{3,}", "\n\n", visible).strip()
    return visible, thinks, goal


def is_plan_panel(raw: str) -> bool:
    """True when `raw` is a plan-mode panel turn: it carries a `<plan>` checklist and no
    executable action (so it must NOT be accepted as the final reply). A bare leading
    `<goal>` with a prose answer (a think/auto DIRECT answer) is NOT a panel — its `<goal>`
    is lifted to the goal panel and the prose is the reply (else the answer is discarded)."""
    return bool(_PLAN_TAG.search(raw)) and extract_call(raw) is None


def plan_bindings(raw: str) -> list[str]:
    """Box bindings (skill/tool names) from an emitted <plan>, minus the
    'none'/'synthesis' markers — the actions the model committed to taking."""
    m = _PLAN_TAG.search(raw)
    out: list[str] = []
    if m:
        for b in _PLAN_BOX.findall(m.group(1)):
            b = b.strip()
            if b not in ("none", "synthesize", "synthesis") and b not in out:
                out.append(b)
    return out


def fired_binding(call: str) -> str:
    """The plan-box binding a just-executed call satisfies: the loaded skill's name
    for load_skill, else the tool name."""
    name = parse_call(call)[0] or ""
    if name == "load_skill":
        m = _SKILL_ARG.search(call)
        return m.group(1) if m else name
    return name


def serving_skills_index(plugin_context: dict | None = None) -> list[dict]:
    """Catalog entries (name + description) for every skill the model may load: the
    SKILL.md files from the skills dir PLUS the skills bundled by enabled plugins."""
    from . import plugins
    base = [
        {"name": skill.name, "description": skill.description,
         "plugin": "chess-official", "source": "official_plugin", "enabled": True}
        for skill in load_skills()
    ]
    return base + plugins.plugin_skills(plugin_context)


def serving_tool_manifest(plugin_context: dict | None = None) -> list[dict]:
    """The full callable tool manifest: official catalog tools + the domain-neutral
    compute tool (calc) + enabled plugins' tools."""
    from . import plugins
    return official_tools() + compute_tools() + plugins.plugin_tools(plugin_context)


def build_system_prompt(agent_overlay: str = "", plugin_context: dict | None = None, game=None,
                        reasoning_mode: str = "") -> str:
    # reasoning_mode ("think"|"fast"|"auto") must match what the corpus trained on
    # (same build_system signal). Default "" keeps current serve behavior until the
    # toggle is wired to it post-training (reconcile with the StagedLoop then).
    pc = plugin_context or PLUGIN_CONTEXT
    base = build_system(serving_skills_index(pc), serving_tool_manifest(pc), pc, agent_overlay,
                        reasoning_mode=reasoning_mode)
    from . import plugins  # prompt-start hooks: pre-load always-on plugin context
    hook = plugins.prompt_start({"game": game}, pc)
    return base + (("\n\n" + hook) if hook else "")


class CoachLoop:
    def __init__(self, model: ModelBackend, executor: ToolExecutor, agent_overlay: str = "",
                 plugin_context: dict | None = None) -> None:
        self.model = model
        self.executor = executor
        self.agent_overlay = agent_overlay
        self.plugin_context = plugin_context
        # the executor dispatches plugin tools + loads plugin skills against this context
        executor.plugin_context = plugin_context
        self.window = _build_window(model)

    @staticmethod
    def _force_answer(convo: list[dict], gen, tool_calls: list[str]) -> str:
        """The model loaded a skill (context) but then gave no real answer. Retry ONCE
        with an explicit 'answer now' instruction so the loaded skill is USED to answer,
        not left as a dead turn. Only fires when the turn produced context-only (skill
        loads or nothing) — never when a fact tool ran (that has a grounded fallback).
        Returns the answer, or '' if the retry still whiffs / leaks a tool call."""
        if not _only_context(tool_calls):
            return ""
        convo.append({"role": "user", "content":
                      "Now answer my question directly in plain text using what you loaded. "
                      "Do not call a tool, do not greet — just give the answer."})
        try:
            forced = gen(REPLY_TOKENS, ACTION_STOP)
        finally:
            convo.pop()  # the nudge is scaffolding; never persist it
        # Reject a re-whiff: empty, a leaked tool tag, a bare lead-in, OR another deflection
        # (the model blurbing capabilities again instead of answering) — '' lets the caller
        # fall back rather than surface a second non-answer as the reply.
        if not forced or contains_tool_call(forced) or _is_leadin_only(forced) or _is_deflection(forced):
            return ""
        return forced

    @staticmethod
    def _verify_fulfilled(convo: list[dict], gen_quiet, user_message: str, draft: str) -> str | None:
        """Design B — the robust self-verification step. After a context-only (skill-load)
        turn produces a final reply, ask the model whether that draft actually DID what the
        user asked, or merely loaded context / asked back ("skill loaded, what would you
        like?"). If it deflected, the model returns the ONE next tool to call to fulfil the
        request (we execute it and the loop continues); if it genuinely answered, returns
        None. Model-driven (not keyword matching) so it generalizes to any phrasing/skill.
        Runs QUIET (no token streaming) — the verdict is scaffolding, not user-facing. The
        caller gates this to skill-load-without-fact turns and to a single pass."""
        convo.append({"role": "user", "content":
            f'Self-check before you reply. The user asked: "{user_message}". Your draft '
            f'reply is: "{draft}". Did you actually DO what they asked, or did you only load '
            "a skill / ask them what they want? If you fully answered, reply with the single "
            "word DONE. If not, output the ONE next tool call that fulfils the request now "
            "(e.g. <tool>best_move depth=18</tool>)."})
        try:
            verdict = gen_quiet(96, ACTION_STOP)
        finally:
            convo.pop()  # scaffolding; never persisted
        return extract_call(verdict)  # next tool to continue with, or None (DONE/fulfilled)

    @staticmethod
    def _next_plan_action(convo: list[dict], gen_quiet, pending_box: str) -> str | None:
        """PLAN-box backstop (Stage 1 anti-early-stop). When the model tries to finalize
        but a plan box is still unfilled, ask it for the ONE next action that fills it.
        Returns the next call to execute, or None when it says blocked / nothing more (so
        the loop finalizes as honest-partial). Scaffolding — never persisted. Generalizes
        to any domain via the model's OWN plan, no keyword map needed."""
        convo.append({"role": "user", "content":
            f'Your plan is not finished — "{pending_box}" still needs doing. Output the ONE '
            "next action (a <skill> or <tool> call) that does it now. If it genuinely cannot "
            "be done, say what is blocked instead — do not fake it."})
        try:
            out = gen_quiet(96, ACTION_STOP)
        finally:
            convo.pop()  # scaffolding; never persisted
        return extract_call(out)

    def _finalize(self, reply: str, required: dict, tool_calls: list[str],
                  tool_results: list[str], new_turns: list[dict], ctx_stats, on_event=None) -> dict:
        """Apply the output guards (strip skill/tool announce, fix a fabricated eval number
        then a fabricated move list, append any required fact still missing) and build the
        return payload. Shared by every final-reply path so the guards never diverge.

        First lifts the reasoning trace out of the visible reply: `<think>` -> the thinking
        panel, `<goal>` -> the goal panel (via on_event), then strips both from the bubble
        text — reasoning is never shown in the user-facing answer (the trained visible shape)."""
        reply, thinks, goal = _split_reasoning(reply)
        if on_event:
            if goal:
                on_event({"type": "goal", "content": goal})
            for t in thinks:
                on_event({"type": "think", "content": t})
        reply = _strip_announce_leadin(reply)
        reply = _correct_eval_number(reply, tool_results)
        reply = _correct_move_names(reply, tool_results)
        reply = _ensure_required_narrated(reply, required, tool_calls, tool_results)
        new_turns.append({"role": "assistant", "content": reply})
        # Display the trained <skill> verb in the payload, not the internal load_skill form.
        disp_calls = [_to_skill_verb(c) for c in tool_calls]
        return {
            "reply": reply,
            "tool_call": disp_calls[-1] if disp_calls else None,
            "tool_result": tool_results[-1] if tool_results else None,
            "tool_calls": disp_calls,
            "tool_results": tool_results,
            "turns": new_turns,
            "context": ctx_stats.as_payload(),
        }

    def respond(self, history: list[dict], user_message: str, coverage: bool = True,
                on_event=None, reasoning_mode: str = "") -> dict:
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
                                     self.executor.game, reasoning_mode=reasoning_mode) \
            + routing_hints(user_message, game_over)
        if not game_over:  # on a finished game, state the result — don't spin up a skill
            system += skill_hints(user_message, serving_skills_index(self.plugin_context))
        # Coverage set: tool -> canonical call for each detected intent. Empty on a
        # finished game or when coverage is off.
        required = {} if (game_over or not coverage) else matched_calls(user_message)
        kept_history, ctx_stats = self.window.fit(system, history, user_message, compress=True)
        # Clever compaction: when old turns are evicted, their distilled note (goal/plan
        # anchor + what's already done) rides the system prompt so the model keeps the
        # thread instead of forgetting. Empty string when nothing was evicted.
        sys_content = system + (("\n\n" + ctx_stats.digest) if ctx_stats.digest else "")
        convo = [{"role": "system", "content": sys_content}, *kept_history,
                 {"role": "user", "content": user_message}]
        new_turns = [{"role": "user", "content": user_message}]
        tool_calls: list[str] = []
        tool_results: list[str] = []
        seen_calls: set[str] = set()   # full <tool> spans, for dedup (same call never re-runs)
        seen_names: set[str] = set()   # tool NAMES gathered, for coverage (best_move != best_move top=3)
        plan_boxes: list[str] = []     # outstanding <plan> box bindings (Stage 1 plan mode)
        plan_fired: set[str] = set()   # bindings satisfied by an executed skill/tool
        plan_nudges = 0                # plan-box steers used this turn (cap = number of boxes)
        goal_shown = [False]           # emit the <goal> to its panel once, when first seen

        # True token streaming: when the caller wants events AND the backend supports it,
        # stream each generation's tokens out as `token` events so the UI fills live. The
        # frontend clears a provisional bubble when the generation turns out to be a tool
        # decision (tool event), and keeps it when it's the final reply.
        import inspect as _inspect
        can_stream = on_event is not None and "on_token" in _inspect.signature(self.model.generate).parameters

        def gen(mx: int, stop: list[str]) -> str:
            if can_stream:
                return self.model.generate(convo, mx, stop,
                                           on_token=lambda t: on_event({"type": "token", "text": t})).strip()
            return self.model.generate(convo, mx, stop).strip()

        def gen_quiet(mx: int, stop: list[str]) -> str:
            # never streams: internal scaffolding (answer-retry, self-verify) whose tokens
            # must NOT leak into the user's chat bubble.
            return self.model.generate(convo, mx, stop).strip()

        verified = False   # the self-verify step (Design B) runs at most once per turn
        for _ in range(MAX_TOOL_CALLS):
            raw = gen(REPLY_TOKENS, ACTION_STOP)
            if on_event and not goal_shown[0]:  # surface the objective to the goal panel early
                gm = _GOAL_TAG.search(raw)
                if gm and gm.group(1).strip():
                    on_event({"type": "goal", "content": gm.group(1).strip()})
                    goal_shown[0] = True
            decision = extract_call(raw)  # canonical call (recovers a dropped <tool> wrapper) or None
            if decision is None and is_plan_panel(raw):
                # PLAN-mode panel turn (<goal>/<plan>): surface it, register the boxes, and
                # KEEP GOING — it is NOT the final answer. Without this the loop would show the
                # raw panel as the reply and never work the boxes (the Stage-1 serve trap).
                plan_boxes = plan_bindings(raw) or plan_boxes
                convo.append({"role": "assistant", "content": raw})
                new_turns.append({"role": "assistant", "content": raw})
                if on_event:
                    on_event({"type": "plan", "content": raw})
                continue
            if decision is None:  # the model wants to give the final reply
                outstanding = [t for t in required if t not in seen_names]
                if not outstanding:
                    # PLAN-box backstop (Stage 1 anti-early-stop): if a <plan> was emitted and
                    # a box is still unfilled, don't accept the final yet — ask for the ONE next
                    # action that fills it (or let it declare honest-partial). Keyed on plan
                    # PRESENCE, so it generalizes to any domain; a no-op when no plan was emitted.
                    pending = [b for b in plan_boxes if b not in plan_fired]
                    if pending and plan_nudges < len(plan_boxes):
                        plan_nudges += 1
                        nxt = self._next_plan_action(convo, gen_quiet, pending[0])
                        if nxt is not None:
                            decision = nxt           # fill the box; fall through to execute
                    if decision is None:
                        # All required intents covered. An empty reply — or a bare tool-intent
                        # lead-in the model stopped on after tools ran — is a non-answer.
                        whiffed = (not raw) or bool(tool_calls and _is_leadin_only(raw))
                        if not whiffed:
                            reply = raw
                        else:
                            # Whiffed: if it ONLY loaded context (skills), retry once with an
                            # explicit "answer now" nudge before the generic fallback.
                            reply = self._force_answer(convo, gen_quiet, tool_calls) \
                                or _fallback_reply(tool_calls, tool_results)
                        reply = _strip_announce_leadin(reply)
                        # Design B — self-verification after a context-only (skill-load) turn:
                        # the reply may be a fluent DEFLECTION that ignores the request. Ask the
                        # model to self-check; if not fulfilled it returns the next tool to run.
                        loaded_skill_only = bool(tool_calls) and _only_context(tool_calls)
                        if coverage and not verified and loaded_skill_only:
                            verified = True
                            nxt = self._verify_fulfilled(convo, gen_quiet, user_message, reply)
                            if nxt is not None:
                                decision = nxt           # deflection that NEEDS a tool -> continue
                            elif _is_deflection(reply):
                                # Verify accepted the model's own "DONE", but the reply is a generic
                                # capability blurb / ask-back answerable from the loaded skill (or the
                                # injected board), not a tool — the self-verify can't fix a no-tool
                                # deflection. Force a direct answer (the strong fixer). Deterministic
                                # detection (no gen); the +1 force gen lands ONLY on this specific
                                # failure, so good answers and tool-routed deflections are unaffected.
                                forced = self._force_answer(convo, gen_quiet, tool_calls)
                                return self._finalize(_strip_announce_leadin(forced) if forced else reply,
                                                      required, tool_calls, tool_results, new_turns,
                                                      ctx_stats, on_event)
                            else:
                                return self._finalize(reply, required, tool_calls,
                                                      tool_results, new_turns, ctx_stats, on_event)
                        else:
                            return self._finalize(reply, required, tool_calls,
                                                  tool_results, new_turns, ctx_stats, on_event)
                        # reached only when verify returned a tool (decision=nxt) -> fall through
                else:
                    # A required intent is still ungathered: force-route it directly
                    # (deterministic coverage) — no nudge round-trip on a small model.
                    decision = required[outstanding[0]]
            # Dedup on the call itself, not the full text — a differing lead-in
            # ("Let me check" vs "I'll look") must not let the same call re-run and
            # re-hit the engine. Key = the <tool>…</tool> span.
            # Dedup + execution use the canonical load_skill form; HISTORY + DISPLAY use
            # the <skill> verb the model was TRAINED on. Storing the load_skill form in
            # convo fed off-distribution history back to the model (it never saw load_skill
            # in training) — a cause of skill re-loading. hist keeps train==serve.
            i0 = decision.find("<tool>")
            key = decision[i0:] if i0 >= 0 else decision
            tool_result = "error: duplicate_tool_call" if key in seen_calls else self.executor.execute(decision)
            seen_calls.add(key)
            name = parse_call(decision)[0] or ""
            if name:
                seen_names.add(name)
            plan_fired.add(fired_binding(decision))   # mark the plan box this call satisfies
            hist = _to_skill_verb(decision)        # <tool>load_skill name=X</tool> -> <skill>X</skill>
            tool_calls.append(decision)
            tool_results.append(tool_result)
            if on_event:  # live progress: surface each tool step as it completes (streaming UI)
                ev_name = "skill" if name == "load_skill" else name
                on_event({"type": "tool", "name": ev_name, "call": hist, "result": tool_result})
            convo += [{"role": "assistant", "content": hist},
                      {"role": "tool", "content": tool_result}]
            new_turns += [{"role": "assistant", "content": hist},
                          {"role": "tool", "content": tool_result}]
            if tool_result == "error: duplicate_tool_call":
                break

        # Budget forcing (s1): out of tool steps, the user is waiting — answer now.
        convo.append({"role": "user", "content":
                      "You're out of tool steps and the user is waiting — give your best answer now using the results you have."})
        reply = gen(REPLY_TOKENS, [])
        if contains_tool_call(reply) or not reply or (tool_calls and _is_leadin_only(reply)):
            # leaked a tool tag, produced nothing, or stopped on a bare tool-intent lead-in
            # after tools ran — narrate the last fact so the user gets a real answer.
            reply = _fallback_reply(tool_calls, tool_results)
        return self._finalize(reply, required, tool_calls, tool_results, new_turns, ctx_stats, on_event)
