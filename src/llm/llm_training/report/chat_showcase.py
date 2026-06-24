"""Authentic chat showcase for the report — REAL model, REAL harness, captured timing + tok/s.

Runs the hand-written scenarios in `chat_suites` through the live CoachLoop on the actual model.
Nothing is fabricated: each coach reply is what the model produced, and every turn records wall
seconds + generated tokens -> tok/s (the latency the user feels). Two sections:
  PLAIN — the bare harness (board hook OFF), casual coaching/concept asks.
  WEB   — the chess-web sandbox (a real board per scenario, board hook ON), multi-turn session.
Writes a markdown transcript (docs/findings/<date>-chat-showcase.md) + one PNG card per section.
  python -m llm_training.report.chat_showcase --adapter <best>     # or --gguf <path> / --server URL
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_training.eval_confusion import _load_model            # noqa: E402
from llm_training.report import chat_suites                    # noqa: E402
from llm_training.report import ppt_charts                     # noqa: E402

REPO = Path(__file__).resolve().parents[4]
ASSETS = REPO / "docs" / "findings" / "report_assets"


class _Timed:
    """Wrap a model so every generate() call accrues (generated tokens, wall seconds) for the
    CURRENT turn. No on_token param -> the loop runs the non-streaming path (clean timing)."""
    def __init__(self, model) -> None:
        self.model = model
        self.reset()

    def reset(self) -> None:
        self.tokens = 0
        self.seconds = 0.0

    def generate(self, messages: list[dict], max_new_tokens: int, stop: list[str]) -> str:
        t0 = time.time()
        out = self.model.generate(messages, max_new_tokens, stop)
        self.seconds += time.time() - t0
        self.tokens += self.model.count_tokens(out)
        return out

    def count_tokens(self, text: str) -> int:
        return self.model.count_tokens(text)

    def context_limit(self) -> int:
        return self.model.context_limit()


def _steps(events: list[dict]) -> list[str]:
    """Compact one-line render of the executed skill/tool steps (for the markdown transcript)."""
    out = []
    for ev in events:
        if ev.get("type") == "tool":
            verb = "skill" if ev.get("name") == "skill" else "tool"
            out.append(f"{verb} `{ev.get('call', '')}` -> `{(ev.get('result') or '')[:120]}`")
    return out


def run_section(model: _Timed, scenarios: list[dict], *, board_hook: bool, engine, label: str) -> list[dict]:
    """Run each scenario as a SESSION (shared loop/board/history) and capture per-turn timing. Sets
    the board hook to mirror the surface being shown (bare harness vs web sandbox)."""
    from backend import inference
    from backend.game import Game
    from backend.inference import CoachLoop
    from backend.tools import ToolExecutor
    prev = inference._BOARD_HOOK
    inference._BOARD_HOOK = board_hook
    turns: list[dict] = []
    try:
        for sc in scenarios:
            game = Game()
            if sc.get("fen"):
                game.load_fen(sc["fen"])
            loop = CoachLoop(model, ToolExecutor(game, engine, None), plugin_context=None)
            history: list[dict] = []
            for text, mode in sc["turns"]:
                model.reset()
                events: list[dict] = []
                res = loop.respond(history, text, coverage=True, on_event=events.append,
                                   reasoning_mode=mode)
                secs, tok = model.seconds, model.tokens
                turns.append({"section": label, "scenario": sc["title"], "fen": sc.get("fen", ""),
                              "prompt": text, "mode": mode, "reply": res.get("reply", "").strip(),
                              "steps": _steps(events), "secs": secs, "gen_tokens": tok,
                              "tok_s": tok / max(secs, 1e-6)})
                history = res.get("turns", history)
                print(f"  [{label}] {text[:46]!r} -> {secs:.1f}s {tok}tok "
                      f"{tok / max(secs, 1e-6):.0f}tok/s", flush=True)
    finally:
        inference._BOARD_HOOK = prev
    return turns


def _md_section(label: str, intro: str, turns: list[dict]) -> list[str]:
    L = [f"## {label}", "", intro, ""]
    cur = None
    for t in turns:
        if t["scenario"] != cur:
            cur = t["scenario"]
            fen = f"  ·  board `{t['fen']}`" if t["fen"] else ""
            L += [f"### {cur}{fen}", ""]
        L.append(f"**[{t['mode']}] User:** {t['prompt']}")
        for s in t["steps"]:
            L.append(f"- {s}")
        L += [f"**Coach:** {t['reply']}",
              f"_⏱ {t['secs']:.1f}s · {t['gen_tokens']} tok · {t['tok_s']:.0f} tok/s_", "", "---", ""]
    L += ["", f"**{label} timing**", "", "| turn | time (s) | tokens | tok/s |", "|---|---|---|---|"]
    for t in turns:
        L.append(f"| {t['prompt'][:40]} | {t['secs']:.1f} | {t['gen_tokens']} | {t['tok_s']:.0f} |")
    return L + [""]


def render(plain: list[dict], web: list[dict], model_label: str) -> str:
    head = ["Parent: docs/reference/harness-architecture.md", "",
            f"# Authentic chat showcase — Gemma 4 chess-coach ({model_label})", "",
            "Real end-to-end runs of the serve loop on hand-written, realistic prompts (slang, vague,",
            "tricky). Replies + timing captured verbatim. Every turn lists wall seconds + generated",
            "tokens + tok/s — the latency a user actually feels.", ""]
    body = _md_section("Section 1 — bare harness (no board)",
                       "The model + the trained harness, nothing the web adds. Coaching/concept asks.",
                       plain)
    body += _md_section("Section 2 — chess-web sandbox (live board)",
                        "The full website surface: a real board (LIVE BOARD line) + multi-turn session.",
                        web)
    return "\n".join(head + body)


def write_showcase(text: str, plain: list[dict], web: list[dict], date_str: str) -> Path:
    out = REPO / "docs" / "findings" / f"{date_str}-chat-showcase.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    ASSETS.mkdir(parents=True, exist_ok=True)
    ppt_charts.chat_card("Section 1 — bare harness", plain, ASSETS / "chat-section1-plain.png",
                         "real model + real harness · casual / vague asks")
    ppt_charts.chat_card("Section 2 — chess-web sandbox", web, ASSETS / "chat-section2-web.png",
                         "live board + multi-turn session")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--server", default="http://127.0.0.1:7861")
    ap.add_argument("--tag", default=None, help="model KEY (e.g. e4b-nf4/e4b-q5): writes the measured "
                    "mean tok/s to report_assets/measured-<tag>.json for the cross-model chart")
    args = ap.parse_args()
    from datetime import date
    try:
        from backend.engine import Engine
        engine = Engine()
    except Exception as exc:                                   # chess tools degrade, PLAIN still runs
        print(f"(no Stockfish engine: {exc}; WEB best_move/eval will error gracefully)", flush=True)
        engine = None
    model = _Timed(_load_model(args))
    label = "GGUF " + Path(args.gguf).name if args.gguf else ("adapter" if args.adapter else "server")
    plain = run_section(model, chat_suites.PLAIN_CHATS, board_hook=False, engine=engine, label="plain")
    web = run_section(model, chat_suites.WEB_CHATS, board_hook=True, engine=engine, label="web")
    out = write_showcase(render(plain, web, label), plain, web, f"{date.today():%Y-%m-%d}")
    if args.tag:                                        # feed the cross-model line chart (tok/s)
        from llm_training.report.measured import update
        allt = plain + web
        mean_tok_s = sum(t["tok_s"] for t in allt) / len(allt) if allt else None
        update(ASSETS, args.tag, tok_s=mean_tok_s)
    print(f"\nwrote {out}\nwrote PNG cards under {ASSETS}", flush=True)


if __name__ == "__main__":
    main()
