"""The tool executor. Maps a `<tool>NAME args</tool>` call against the live
python-chess board + Stockfish to the exact return strings of spec section 2,
plus load_skill (progressive disclosure — returns a skill's SKILL.md body).

review_move and threats are computed here because they compose board + engine."""
from __future__ import annotations

import re

import chess
import chess.engine

from . import ask_kb
from .engine import Engine
from .game import Game
from .skills import load_skills
from .toolfmt import clamp_depth, fmt_white_score, parse_call

LABELS = (  # (max cp loss, label); >last -> blunder
    (10, "excellent"), (50, "good"), (100, "inaccuracy"), (200, "mistake"),
)
DEFAULT_EVAL_DEPTH = 18


def parse_range(value: str | None, default: int, high: int) -> int:
    try:
        n = int(value) if value is not None else default
    except ValueError:
        n = default
    return max(1, min(high, n))


def format_score(kind: str, val) -> str:
    if kind == "mate":
        side, n = val
        return f"M{n} for {side}"
    return f"{int(val) / 100:+.2f}"


# Executor-accurate call validation. DELIBERATELY NARROWER than catalog.OFFICIAL_TOOLS'
# "args" display schema: that schema is a TRAINING-DISPLAY hint, and the executor
# tolerates loose args (depth defaults via clamp_depth, board_state.fields accepts a
# superset turn/last_move/..., best_move.top/series clamp to 1-5). Validating against the
# display schema would REJECT calls the executor handles fine. So this lists ONLY what the
# executor truly constrains: a required arg whose absence yields a cryptic downstream error,
# and enum args the executor SILENTLY DEFAULTS on a bad value (so a typo would otherwise
# return wrong-but-plausible data instead of an error the model can self-correct from).
_REQUIRED_ARGS = {"move": "san"}                       # move("") -> cryptic; name the gap
_ENUM_ARGS = {                                         # silent-default enums (wrong -> wrong data)
    "list_pieces": ("color", ("white", "black", "mine")),
    "random_position": ("kind", ("puzzle", "scramble", "open")),
}


def validate_call(name: str, args: dict[str, str]) -> str | None:
    """Corrective error (for the model to self-correct from) when a known call is missing a
    truly-required arg or passes an out-of-range value to a silent-default enum; else None.
    Unknown tools and harmless extra/defaulted args are left to dispatch — never over-reject."""
    req = _REQUIRED_ARGS.get(name)
    if req and not args.get(req):
        return f"error: tool '{name}' needs '{req}=...' — e.g. <tool>{name} {req}=...</tool>"
    enum = _ENUM_ARGS.get(name)
    if enum:
        arg, allowed = enum
        if args.get(arg) is not None and args[arg] not in allowed:
            return (f"error: arg '{arg}' for '{name}' must be one of "
                    f"{'/'.join(allowed)} (got '{args[arg]}')")
    return None


_SKILL_BODY_CAP = 2800   # chars (~700 tok) — a SAFETY bound for a pathological dropped-in
                         # SKILL.md only; a well-formed skill (incl. chess-coach) is never cut,
                         # so this never severs a rule mid-body. Terseness is the SKILL.md's job.


def _condense_skill_body(body: str) -> str:
    """Return the instructional body the model should follow, shaped like what it TRAINED
    on (a terse `# name / steps`), not a verbose markdown dump:
    - strip the YAML frontmatter (name+description are ALREADY in the system-prompt catalog —
      re-sending them in the body is pure redundancy the model never saw at train time);
    - collapse blank-line runs;
    - cap length (protects against a pathological dropped-in skill re-encoding every loop step).
    Why: training set the load_skill result to a 17-93 token lesson; serving the full 616-token
    SKILL.md was both off-distribution (a cause of post-load deflection/reloading) and the
    skill-load LATENCY (the body re-prefills on every subsequent step)."""
    text = (body or "").strip()
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:                       # drop the frontmatter block + its closing fence
            text = text[end + 4:].lstrip("-\n ").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > _SKILL_BODY_CAP:
        cut = text.rfind("\n", 0, _SKILL_BODY_CAP)   # cut on a line boundary, not mid-sentence
        text = text[: cut if cut > _SKILL_BODY_CAP // 2 else _SKILL_BODY_CAP].rstrip() + "\n…"
    return text


class ToolExecutor:
    def __init__(self, game: Game, engine: Engine, plugin_context: dict | None = None) -> None:
        self.game = game
        self.engine = engine
        # Which plugin bundles are enabled — gates which plugin tools dispatch + which
        # plugin skill bodies load_skill can fetch. Set by the loop/web_app.
        self.plugin_context = plugin_context

    def execute(self, tool_call: str) -> str:
        name, args = parse_call(tool_call)
        if not name:
            return "error: invalid_syntax"
        bad = validate_call(name, args)            # missing required / bad enum -> corrective error
        if bad:
            return bad
        try:
            return self._dispatch(name, args)
        except chess.engine.EngineError:
            return "error: engine_unavailable"
        except Exception:
            return "error: engine_unavailable"

    def _dispatch(self, name: str, args: dict[str, str]) -> str:
        if name == "load_skill":
            return self._load_skill(args.get("name", ""))
        if name == "board_state":
            return self._board_state(args.get("fields", "basic"))
        if name == "move":
            return self.game.move(args.get("san", ""))
        if name == "load_fen":
            fen = args.get("fen", "")
            if self.game.load_fen(fen):
                return self._board_state("all")
            return "error: invalid_fen"
        if name == "random_position":
            from .positions import random_position
            return random_position(self.game, args.get("kind", "puzzle"), engine=self.engine)
        if name == "fetch_puzzle":
            from .online_positions import fetch_puzzle
            return fetch_puzzle(self.game)
        if name == "undo":
            return self.game.undo()
        if name == "legal_moves":
            return self.game.legal_moves(args.get("square"))
        if name == "list_pieces":
            return self.game.list_pieces(args.get("color", "mine"))
        if name == "ask_chessbot":
            return ask_kb.answer(args.get("query", ""))
        if name == "python":
            # Domain-neutral verification primitive: ground a computed/checkable
            # claim by RUNNING a short script and reading stdout, not by asserting.
            # Sandboxed subprocess — see sandbox.run_python.
            from .sandbox import run_python
            return run_python(args.get("code", ""))
        if name == "eval":
            depth = clamp_depth(args, DEFAULT_EVAL_DEPTH)
            if self.game.board.fen() == chess.STARTING_FEN:
                return f"score: 0.00 pawns from white POV, depth={depth} (starting position is equal)"
            kind, val = self.engine.eval_white_cp(self.game.board, depth)
            return fmt_white_score(kind, val, depth)
        if name == "best_move":
            return self._best_move(args)
        if name == "review_move":
            return self._review(clamp_depth(args, 15))
        if name == "threats":
            return self._threats(clamp_depth(args, 12))
        # Enabled plugins' tools (openings/analysis/...) — the registry routes to the
        # plugin that owns the name. This is the surface that tests cross-bundle routing.
        from . import plugins
        plugin_res = plugins.dispatch(name, args, self, self.plugin_context)
        if plugin_res is not None:
            return plugin_res
        # A skill is NOT a tool. The harness contract loads a skill with the
        # <skill>NAME</skill> verb. If the model calls a known skill BY NAME as a
        # tool (<tool>chess-coach</tool>), do NOT silently accept it — that masks
        # the protocol violation. Return a corrective error naming the right verb
        # so the loop retries with <skill> (self-correction, enforced).
        plugin_skill_names = {s["name"] for s in plugins.plugin_skills(self.plugin_context)}
        if name in {s.name for s in load_skills()} | plugin_skill_names:
            return f"error: '{name}' is a skill, not a tool — load it with <skill>{name}</skill>"
        # Parsed a name that is neither a known tool, a plugin tool, nor a skill: a
        # hallucinated tool. Say so by NAME (not generic invalid_syntax, which also means
        # "unparseable") so the model can re-route to a real tool from its manifest.
        return f"error: unknown_tool '{name}'"

    def _known_tool_names(self) -> set[str]:
        """Every callable tool name in the current surface (official + compute + enabled plugins)."""
        from llm_dataset.v1.catalog import compute_tools, official_tools
        from . import plugins
        return ({t["name"] for t in official_tools()} | {t["name"] for t in compute_tools()}
                | {t["name"] for t in plugins.plugin_tools(self.plugin_context)})

    def _load_skill(self, name: str) -> str:
        """Progressive disclosure: return the named skill's body so the model can follow
        it — CONDENSED (see _condense_skill_body) so it matches the terse body shape the
        model trained on and doesn't re-encode a 600+ token markdown dump every loop step."""
        bodies = {skill.name: skill.content for skill in load_skills()}
        if name in bodies:
            return _condense_skill_body(bodies[name])
        from . import plugins
        body = plugins.skill_body(name, self.plugin_context)
        if body is not None:
            return _condense_skill_body(body)
        # A TOOL loaded as a skill (<skill>metronome_bpm</skill> — seen on out-of-domain
        # routing). Symmetric to the skill-as-tool error in _dispatch: name the right verb so
        # the loop self-corrects to <tool>, instead of a dead 'unknown_skill' (which made the
        # model flail back to its training-dominant skill).
        if name in self._known_tool_names():
            return f"error: '{name}' is a tool, not a skill — call it with <tool>{name} ...</tool>"
        return "error: unknown_skill"

    def _board_state(self, fields: str) -> str:
        board = self.game.board
        requested = {part.strip() for part in fields.split(",") if part.strip()}
        if not requested or "basic" in requested:
            requested = {"turn", "last_move", "check", "legal_count"}
        if "all" in requested:
            requested = {"turn", "fen", "last_move", "check", "legal_count", "history"}
        values = {
            "turn": "white" if board.turn == chess.WHITE else "black",
            "fen": board.fen(),
            "last_move": self.game.san_stack[-1] if self.game.san_stack else "none",
            "check": "yes" if board.is_check() else "no",
            "legal_count": str(board.legal_moves.count()),
            "history": " ".join(self.game.san_stack) or "none",
        }
        parts = [f"{key}={values[key]}" for key in values if key in requested]
        return "board_state: " + ", ".join(parts)

    def _best_move(self, args: dict[str, str]) -> str:
        depth = clamp_depth(args, DEFAULT_EVAL_DEPTH)
        top = parse_range(args.get("top"), 1, 5)
        series = parse_range(args.get("series"), 1, 5)
        if "top" in args:
            moves = self.engine.best_moves(self.game.board, depth, top)
            if not moves:
                return "best: none (no legal moves)"
            if series > 1:
                sans, (kind, val) = self.engine.best_line(self.game.board, depth, series)
                score = fmt_white_score(kind, val, depth).removeprefix("score: ").split(", depth")[0]
                head = "; ".join(f"{i}. {san} ({format_score(kind, val)})" for i, (san, (kind, val)) in enumerate(moves, 1))
                return f"best_moves: {head}; best_line: {' '.join(sans)}, score: {score}"
            return "best_moves: " + "; ".join(
                f"{i}. {san} ({format_score(kind, val)})" for i, (san, (kind, val)) in enumerate(moves, 1))
        sans, (kind, val) = self.engine.best_line(self.game.board, depth, series)
        if not sans:
            return "best: none (no legal moves)"
        score = fmt_white_score(kind, val, depth).removeprefix("score: ").split(", depth")[0]
        if series == 1:
            # Always attach the score, in the same ", score: ... pawns from white POV"
            # idiom as best_line/best_moves. The bare "best: <move>" form carried no
            # number, and the model (trained only on scored best_* results) then
            # invented one from opening priors. Grounding it kills that fabrication.
            return f"best: {sans[0]}, score: {score}"
        return f"best_line: {' '.join(sans)}, score: {score}"

    def _review(self, depth: int) -> str:
        board = self.game.board
        if not board.move_stack:
            return "error: no moves to review"
        played = self.game.san_stack[-1] if self.game.san_stack else board.peek().uci()
        mv = board.pop()
        mover = board.turn
        info_before = self.engine.analyse(board, depth)
        best = info_before.get("pv", [None])[0]
        best_san = board.san(best) if best else "?"
        before = info_before["score"].pov(mover).score(mate_score=100000)
        board.push(mv)
        after = self.engine.analyse(board, depth)["score"].pov(mover).score(mate_score=100000)
        loss = before - after
        label = next((lab for cap, lab in LABELS if loss <= cap), "blunder")
        return f"review: {played}, label={label}, delta={(after - before) / 100:+.2f} pawns, best_was={best_san}"

    def _threats(self, depth: int) -> str:
        board = self.game.board
        opp = not board.turn
        base = self.engine.analyse(board, depth)["score"].pov(opp).score(mate_score=100000)
        nb = board.copy()
        nb.push(chess.Move.null())
        best_san, (kind, val) = self.engine.best_for_side_to_move(nb, depth)
        if kind == "mate":
            return f"threats: opponent's best is {best_san}, score for them: mate in {abs(val)}"
        swing = val - base
        if swing < 50:
            return f"threats: none significant (best opponent move only changes eval by {swing / 100:.2f})"
        return f"threats: opponent's best is {best_san}, score for them: {val / 100:+.2f} pawns"
