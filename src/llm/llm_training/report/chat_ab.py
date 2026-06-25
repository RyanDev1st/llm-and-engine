"""Reply-CORRECTNESS A/B — the same hand-written prompts run under different SERVE configs, replies
shown side by side so you can judge which config gives APT content. This isolates whether off-point
answers come from the off-distribution LIVE BOARD prompt (board_hook) or the rescue layer (thin), vs
the weights. Model loads ONCE; no benchmark, no eval — just the chats.

Variants (web suite; plain is board-independent so it always runs board-hook-off):
  board_on  — board hook ON  (current LIVE config: the LIVE BOARD line injected each turn)
  board_off — board hook OFF (the CLEAN trained prompt: board hidden, model calls board_state)
  thin      — board hook ON + CHESS_THIN_HARNESS (rescue/coverage layer dropped)

  python -m llm_training.report.chat_ab --adapter <best> [--variants board_on,board_off,thin]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_training.eval_confusion import _load_model            # noqa: E402
from llm_training.report import chat_suites                    # noqa: E402
from llm_training.report.chat_showcase import _Timed, _board_panel, run_section  # noqa: E402

REPO = Path(__file__).resolve().parents[4]

# label -> (web board_hook, thin_harness). plain always runs board-hook-off (no board in the prompt).
_VARIANTS = {
    "board_on": (True, False),
    "board_off": (False, False),
    "thin": (True, True),
}
_DESC = {"board_on": "board hook ON (current LIVE)", "board_off": "board hook OFF (clean trained prompt)",
         "thin": "thin harness (rescue layer off)"}


def run_ab(model, engine, variants: list[str]) -> dict:
    """Run plain + web suites under each variant (one model load). Returns
    {suite: [ {prompt, scenario, fen, by_variant: {label: row}} per turn ]}."""
    from backend import inference
    prev_thin = inference._THIN_HARNESS
    per: dict[str, dict[str, list]] = {}
    try:
        for label in variants:
            web_hook, thin = _VARIANTS[label]
            inference._THIN_HARNESS = thin
            plain = run_section(model, chat_suites.PLAIN_CHATS, board_hook=False, engine=engine, label="plain")
            web = run_section(model, chat_suites.WEB_CHATS, board_hook=web_hook, engine=engine, label="web")
            per[label] = {"plain": plain, "web": web}
            print(f"  [variant {label}] done", flush=True)
    finally:
        inference._THIN_HARNESS = prev_thin
    out: dict[str, list] = {}
    for suite in ("plain", "web"):
        ref = per[variants[0]][suite]
        groups = []
        for i, r in enumerate(ref):
            groups.append({"prompt": r["prompt"], "scenario": r["scenario"], "fen": r["fen"],
                           "by": {lab: per[lab][suite][i] for lab in variants}})
        out[suite] = groups
    return out


def render(ab: dict, variants: list[str], model_label: str) -> str:
    L = ["Parent: docs/findings/2026-06-24-harness-live-vs-benchmark-gap.md", "",
         f"# Chat A/B — reply CORRECTNESS across serve configs ({model_label})", "",
         "Same prompts, different serve config — judge which gives APT content. Variants:", ""]
    for v in variants:
        L.append(f"- **{v}** — {_DESC.get(v, v)}")
    for suite, title in (("plain", "Section 1 — bare harness (no board)"),
                         ("web", "Section 2 — chess-web sandbox (live board)")):
        L += ["", f"## {title}", ""]
        cur = None
        for g in ab[suite]:
            if g["scenario"] != cur:
                cur = g["scenario"]
                fen = f"  ·  board `{g['fen']}`" if g["fen"] else ""
                L += [f"### {cur}{fen}", ""]
            L.append(f"**User:** {g['prompt']}")
            for v in variants:
                r = g["by"][v]
                L.append(f"- **{v}** ({r['secs']:.1f}s · {r['tok_s']:.0f} tok/s): {r['reply'].strip()}")
                for s in r.get("steps", []):                  # the tool/skill steps this variant ran
                    L.append(f"    - {s}")
                if g["fen"]:                                  # board before -> after, per variant
                    L += ["  "] + _board_panel(r)
            L += ["", "---", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--server", default="http://127.0.0.1:7861")
    ap.add_argument("--variants", default="board_on,board_off,thin",
                    help="comma list from board_on,board_off,thin (drop any to save time)")
    args = ap.parse_args()
    from datetime import date
    variants = [v.strip() for v in args.variants.split(",") if v.strip() in _VARIANTS]
    assert variants, "no valid variants"
    try:
        from backend.engine import Engine
        engine = Engine()
    except Exception as exc:
        print(f"(no Stockfish: {exc}; web analysis tools will error gracefully)", flush=True)
        engine = None
    model = _Timed(_load_model(args))
    label = "GGUF " + Path(args.gguf).name if args.gguf else ("adapter" if args.adapter else "server")
    ab = run_ab(model, engine, variants)
    text = render(ab, variants, label)
    out = REPO / "docs" / "findings" / f"{date.today():%Y-%m-%d}-chat-ab.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print("\n" + text, flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
    from llm_training.clean_exit import flush_and_exit
    flush_and_exit()   # benign torch/CUDA exit-time SIGABRT must not fail the notebook run
