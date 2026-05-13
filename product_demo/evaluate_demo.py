from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone

from chess_tool_demo import Board, Move, best_move, evaluate, run_tool_turn
from train_router import load_router_examples, predict

HUMAN_PROMPT_CASES = [
    {"prompt": "Can you evaluate this position for me?", "expected_tool": "eval", "arguments": {}},
    {"prompt": "What move should I consider here?", "expected_tool": "best_move", "arguments": {}},
    {"prompt": "Was my move e2e4 good?", "expected_tool": "review_move", "arguments": {"uci": "e2e4"}},
    {"prompt": "Is g1f3 a good move in this position?", "expected_tool": "review_move", "arguments": {"uci": "g1f3"}},
]


@dataclass(frozen=True)
class MatchResult:
    game: int
    demo_color: str
    plies: int
    outcome: str
    final_score_cp_white: int
    final_bucket: str


def load_model(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def prompt_cases_from_eval(eval_path: str) -> list[dict]:
    cases = []
    for example in load_router_examples(eval_path):
        cases.append({"prompt": example.text, "expected_tool": example.tool_name, "arguments": arguments_for_prompt(example.text, example.tool_name)})
    return cases


def arguments_for_prompt(prompt: str, tool_name: str) -> dict:
    if tool_name != "review_move":
        return {}
    for token in prompt.lower().replace("?", "").split():
        if len(token) in (4, 5) and token[0] in "abcdefgh" and token[2] in "abcdefgh":
            return {"uci": token}
    return {"uci": "e2e4"}


def simulate_prompt_cases(model: dict, eval_path: str | None) -> dict:
    cases = HUMAN_PROMPT_CASES + (prompt_cases_from_eval(eval_path) if eval_path else [])
    rows = []
    correct = 0
    tool_ok = 0
    for case in cases:
        predicted, scores = predict(model, case["prompt"])
        router_ok = predicted == case["expected_tool"]
        board = Board.from_fen()
        turn = run_tool_turn(board, predicted, case["arguments"])
        status_ok = turn["tool_result"].get("status") == "ok"
        correct += int(router_ok)
        tool_ok += int(status_ok)
        rows.append(
            {
                "prompt": case["prompt"],
                "expected_tool": case["expected_tool"],
                "predicted_tool": predicted,
                "router_ok": router_ok,
                "tool_status": turn["tool_result"].get("status"),
                "narration": turn["narration"],
                "scores": scores,
            }
        )
    return {
        "cases": len(cases),
        "router_correct": correct,
        "router_accuracy": correct / len(cases),
        "tool_ok": tool_ok,
        "tool_success_rate": tool_ok / len(cases),
        "rows": rows,
    }


def weak_bot_move(board: Board, rng: random.Random) -> Move | None:
    legal = board.legal_moves()
    if not legal:
        return None
    captures = [move for move in legal if board.piece_at(move.target)]
    if captures and rng.random() < 0.7:
        return rng.choice(captures)
    return rng.choice(legal)


def demo_engine_move(board: Board) -> Move | None:
    choice = best_move(board.clone())
    if choice["status"] != "ok":
        return None
    return Move.parse(choice["best_move"])


def winner_from_score(score: int, demo_color: str) -> str:
    if abs(score) < 80:
        return "drawish"
    white_winning = score > 0
    demo_is_white = demo_color == "w"
    return "demo_win" if white_winning == demo_is_white else "demo_loss"


def play_game(game: int, demo_color: str, seed: int, max_plies: int) -> MatchResult:
    board = Board.from_fen()
    rng = random.Random(seed)
    plies = 0
    for _ in range(max_plies):
        move = demo_engine_move(board) if board.turn == demo_color else weak_bot_move(board, rng)
        if move is None:
            break
        result = board.move(move)
        if result["status"] != "ok":
            break
        plies += 1
    final_eval = evaluate(board)
    return MatchResult(game, demo_color, plies, winner_from_score(final_eval["score_cp_white"], demo_color), final_eval["score_cp_white"], final_eval["bucket"])


def run_match_suite(games: int, max_plies: int) -> dict:
    results = [play_game(index + 1, "w" if index % 2 == 0 else "b", 1000 + index, max_plies) for index in range(games)]
    wins = sum(1 for result in results if result.outcome == "demo_win")
    losses = sum(1 for result in results if result.outcome == "demo_loss")
    drawish = sum(1 for result in results if result.outcome == "drawish")
    return {
        "opponent": "weak community-style random/capture bot baseline, seeded locally because GitHub search returned no usable 400-800 ELO bot candidate",
        "games": games,
        "max_plies": max_plies,
        "demo_wins": wins,
        "demo_losses": losses,
        "drawish": drawish,
        "demo_score_rate": (wins + 0.5 * drawish) / games,
        "rows": [result.__dict__ for result in results],
    }


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_summary(path: str, prompt_result: dict, match_result: dict) -> None:
    lines = [
        "# Demo SFT Router and Chess Engine Evaluation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## SFT Router Prompt Simulation",
        "",
        f"- Cases: {prompt_result['cases']}",
        f"- Router accuracy: {prompt_result['router_accuracy']:.3f} ({prompt_result['router_correct']}/{prompt_result['cases']})",
        f"- Tool success rate: {prompt_result['tool_success_rate']:.3f} ({prompt_result['tool_ok']}/{prompt_result['cases']})",
        "",
        "## Engine Match Evaluation",
        "",
        f"- Opponent: {match_result['opponent']}",
        f"- Games: {match_result['games']}",
        f"- Demo wins: {match_result['demo_wins']}",
        f"- Demo losses: {match_result['demo_losses']}",
        f"- Drawish: {match_result['drawish']}",
        f"- Demo score rate: {match_result['demo_score_rate']:.3f}",
        "",
        "## Proficiency Conclusion",
        "",
        "The demo engine shows basic low-level tactical proficiency against a weak random/capture baseline, but it is not a real rated chess engine. Current oracle is material/mobility only, with no castling, en passant, draw rules, Stockfish search, or calibrated ELO rating.",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local SFT router and demo chess engine")
    parser.add_argument("--model", default="product_demo/trained_router.json")
    parser.add_argument("--eval", default="product_demo/training_data/eval_sft.jsonl")
    parser.add_argument("--out-dir", default="results/demo_eval")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-plies", type=int, default=80)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = load_model(args.model)
    prompt_result = simulate_prompt_cases(model, args.eval if os.path.exists(args.eval) else None)
    match_result = run_match_suite(args.games, args.max_plies)
    search_result = {"github_search": "No usable community 400-800 ELO bot candidate found via gh search in this environment.", "fallback": match_result["opponent"]}

    write_json(os.path.join(args.out_dir, "sft_prompt_simulation.json"), prompt_result)
    write_json(os.path.join(args.out_dir, "engine_match_results.json"), match_result)
    write_json(os.path.join(args.out_dir, "bot_search_result.json"), search_result)
    write_summary(os.path.join(args.out_dir, "summary.md"), prompt_result, match_result)
    print(json.dumps({"out_dir": args.out_dir, "router_accuracy": prompt_result["router_accuracy"], "tool_success_rate": prompt_result["tool_success_rate"], "demo_score_rate": match_result["demo_score_rate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
