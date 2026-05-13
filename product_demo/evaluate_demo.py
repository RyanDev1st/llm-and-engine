from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone

import torch
import chess

from chess_tool_demo import Board, run_tool_turn
from train_chess_engine import predict_move, static_eval
from train_sft_poc import RouterLmPredictor, extract_uci, load_router_examples, metric_summary_from_counts, predict_router, predict_router_lm, stockfish_status

HUMAN_PROMPT_CASES = [
    {"prompt": "Can you evaluate this position for me?", "expected_tool": "eval", "arguments": {}},
    {"prompt": "How is the current board looking?", "expected_tool": "eval", "arguments": {}},
    {"prompt": "What move should I consider here?", "expected_tool": "best_move", "arguments": {}},
    {"prompt": "Find one practical candidate move.", "expected_tool": "best_move", "arguments": {}},
    {"prompt": "Was my move e2e4 good?", "expected_tool": "review_move", "arguments": {"uci": "e2e4"}},
    {"prompt": "Is g1f3 a good move in this position?", "expected_tool": "review_move", "arguments": {"uci": "g1f3"}},
    {"prompt": "Please review b1c3 without guessing.", "expected_tool": "review_move", "arguments": {"uci": "b1c3"}},
]


@dataclass(frozen=True)
class MatchResult:
    game: int
    engine_color: str
    plies: int
    outcome: str
    final_score_cp_white: float
    final_bucket: str
    termination: str


def load_model(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def prompt_cases_from_eval(eval_path: str) -> list[dict]:
    cases = []
    for example in load_router_examples(eval_path):
        cases.append({"prompt": example.text, "expected_tool": example.tool_name, "arguments": example.arguments, "internal_fen": example.internal_fen})
    return cases


def metric_summary(rows: list[dict]) -> dict:
    cases = len(rows)
    router_correct = sum(1 for row in rows if row["router_ok"])
    end_to_end_correct = sum(1 for row in rows if row.get("parse_ok", True) and row["router_ok"] and row["argument_ok"] is not False)
    tool_ok = sum(1 for row in rows if row["tool_status"] == "ok")
    return {
        "cases": cases,
        "router_correct": router_correct,
        "router_accuracy": router_correct / cases if cases else 0.0,
        "end_to_end_correct": end_to_end_correct,
        "end_to_end_accuracy": end_to_end_correct / cases if cases else 0.0,
        "tool_ok": tool_ok,
        "tool_success_rate": tool_ok / cases if cases else 0.0,
    }


def simulate_prompt_cases(model: dict, eval_path: str | None, device: torch.device) -> dict:
    cases = HUMAN_PROMPT_CASES + (prompt_cases_from_eval(eval_path) if eval_path else [])
    rows = []
    failure_counts: dict[str, int] = {}
    labels = sorted({case["expected_tool"] for case in cases} | set(model.get("labels", [])))
    confusion = {label: {inner: 0 for inner in labels} for label in labels}
    is_lm = model.get("model_type") == "local-transformers-causal-lm-router-sft-v1"
    lm_predictor = RouterLmPredictor(model, device) if is_lm else None
    for case in cases:
        raw_generation = None
        parse_error = None
        if is_lm:
            predicted, predicted_arguments, raw_generation, parse_error = predict_router_lm(model, case["prompt"], predictor=lm_predictor)
            predicted = predicted or "eval"
            predicted_arguments = predicted_arguments or {}
        else:
            predicted = predict_router(model, case["prompt"])
            uci = extract_uci(case["prompt"])
            predicted_arguments = {"uci": uci} if predicted == "review_move" and uci else {}
        router_ok = predicted == case["expected_tool"]
        argument_ok = predicted_arguments == case["arguments"] if predicted == case["expected_tool"] else False
        if predicted not in confusion[case["expected_tool"]]:
            for row in confusion.values():
                row[predicted] = 0
            confusion[predicted] = {label: 0 for label in confusion}
        confusion[case["expected_tool"]][predicted] += 1
        board = Board.from_fen(case.get("internal_fen"))
        turn = run_tool_turn(board, predicted, predicted_arguments)
        status_ok = turn["tool_result"].get("status") == "ok"
        reason = parse_error or ("ok" if router_ok and status_ok else f"router={router_ok};tool_status={turn['tool_result'].get('status')}")
        failure_counts[reason] = failure_counts.get(reason, 0) + 1
        rows.append(
            {
                "prompt": case["prompt"],
                "expected_tool": case["expected_tool"],
                "predicted_tool": predicted,
                "router_ok": router_ok,
                "expected_arguments": case["arguments"],
                "predicted_arguments": predicted_arguments,
                "argument_ok": argument_ok,
                "tool_status": turn["tool_result"].get("status"),
                "narration": turn["narration"],
                "why": reason,
                "board_source": "eval_fen" if case.get("internal_fen") else "default_start",
                "raw_generation": raw_generation,
                "parse_ok": parse_error is None,
                "parse_error": parse_error,
            }
        )
    by_source = {
        source: metric_summary([row for row in rows if row["board_source"] == source])
        for source in sorted({row["board_source"] for row in rows})
    }
    totals = metric_summary(rows)
    minimum_support_required = 10
    labels = sorted(confusion)
    per_tool_metrics, macro_f1 = metric_summary_from_counts(confusion, labels)
    return {
        "cases": totals["cases"],
        "router_correct": totals["router_correct"],
        "router_accuracy": totals["router_accuracy"],
        "end_to_end_correct": totals["end_to_end_correct"],
        "end_to_end_accuracy": totals["end_to_end_accuracy"],
        "macro_f1": macro_f1,
        "tool_ok": totals["tool_ok"],
        "tool_success_rate": totals["tool_success_rate"],
        "by_board_source": by_source,
        "per_tool": {tool: {"examples": per_tool_metrics[tool]["support"], "accuracy": per_tool_metrics[tool]["recall"], "precision": per_tool_metrics[tool]["precision"], "recall": per_tool_metrics[tool]["recall"], "f1": per_tool_metrics[tool]["f1"], "minimum_support_required": minimum_support_required, "minimum_support_passed": per_tool_metrics[tool]["support"] >= minimum_support_required} for tool in sorted(per_tool_metrics)},
        "failure_slices": dict(sorted((reason, count) for reason, count in failure_counts.items() if reason != "ok")),
        "outcome_slices": dict(sorted(failure_counts.items())),
        "rows": rows,
    }


def baseline_move(board, rng: random.Random):
    legal = list(board.legal_moves)
    if not legal:
        return None
    captures = [move for move in legal if board.is_capture(move)]
    if captures and rng.random() < 0.7:
        return rng.choice(captures)
    return rng.choice(legal)


def engine_move(board, engine_model: dict):
    return predict_move(board, engine_model["weights"])


def score_bucket(score: float) -> str:
    if score > 120:
        return "white_better"
    if score < -120:
        return "black_better"
    return "balanced"


def winner_from_score(score: float, engine_color: str) -> str:
    if abs(score) < 80:
        return "drawish"
    white_winning = score > 0
    engine_is_white = engine_color == "w"
    return "engine_win" if white_winning == engine_is_white else "engine_loss"


def play_game(game: int, engine_color: str, seed: int, max_plies: int, engine_model: dict) -> MatchResult:
    board = chess.Board()
    rng = random.Random(seed)
    plies = 0
    termination = "max_plies"
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            termination = board.outcome(claim_draw=True).termination.name if board.outcome(claim_draw=True) else "game_over"
            break
        move = engine_move(board, engine_model) if ((board.turn and engine_color == "w") or (not board.turn and engine_color == "b")) else baseline_move(board, rng)
        if move is None:
            termination = "no_legal_move"
            break
        board.push(move)
        plies += 1
    final_score = static_eval(board)
    return MatchResult(game, engine_color, plies, winner_from_score(final_score, engine_color), final_score, score_bucket(final_score), termination)


def run_match_suite(engine_model: dict, games: int, max_plies: int, stockfish: dict) -> dict:
    results = [play_game(index + 1, "w" if index % 2 == 0 else "b", 1000 + index, max_plies, engine_model) for index in range(games)]
    zero_ply_games = sum(1 for result in results if result.plies == 0)
    if games > 0 and zero_ply_games == games:
        raise RuntimeError("Engine match sanity gate failed: every game ended with zero plies.")
    wins = sum(1 for result in results if result.outcome == "engine_win")
    losses = sum(1 for result in results if result.outcome == "engine_loss")
    drawish = sum(1 for result in results if result.outcome == "drawish")
    plies = [result.plies for result in results]
    return {
        "engine_model_type": engine_model.get("model_type"),
        "legality_backend": engine_model.get("legality_backend"),
        "opponent": "seeded capture/random baseline using python-chess legal move generation; separate from training-time same-heuristic baseline",
        "starting_position": "chess.STARTING_FEN",
        "stockfish": stockfish,
        "stockfish_available": bool(stockfish.get("available")),
        "games": games,
        "max_plies": max_plies,
        "min_plies": min(plies) if plies else 0,
        "mean_plies": sum(plies) / len(plies) if plies else 0.0,
        "max_observed_plies": max(plies) if plies else 0,
        "zero_ply_games": zero_ply_games,
        "engine_wins": wins,
        "engine_losses": losses,
        "drawish": drawish,
        "engine_score_rate": (wins + 0.5 * drawish) / games if games else 0.0,
        "rows": [result.__dict__ for result in results],
    }


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_summary(path: str, prompt_result: dict, match_result: dict) -> None:
    lines = [
        "# SFT Router and Chess Engine Evaluation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## SFT Router Prompt Simulation",
        "",
        f"- Cases: {prompt_result['cases']}",
        f"- Router tool-name accuracy: {prompt_result['router_accuracy']:.3f} ({prompt_result['router_correct']}/{prompt_result['cases']})",
        f"- Router end-to-end tool-call accuracy: {prompt_result['end_to_end_accuracy']:.3f} ({prompt_result['end_to_end_correct']}/{prompt_result['cases']})",
        f"- Tool success rate: {prompt_result['tool_success_rate']:.3f} ({prompt_result['tool_ok']}/{prompt_result['cases']})",
        "- Metrics by board source:",
    ]
    for source, metrics in prompt_result.get("by_board_source", {}).items():
        lines.append(f"  - {source}: tool_accuracy={metrics['router_accuracy']:.3f} ({metrics['router_correct']}/{metrics['cases']}), end_to_end_accuracy={metrics['end_to_end_accuracy']:.3f} ({metrics['end_to_end_correct']}/{metrics['cases']}), tool_success_rate={metrics['tool_success_rate']:.3f} ({metrics['tool_ok']}/{metrics['cases']})")
    lines.extend([
        "- Where it succeeds:",
    ])
    lines.extend(f"  - {tool}: {data['accuracy']:.3f} over {data['examples']} prompts, minimum_support_passed={data.get('minimum_support_passed')}" for tool, data in prompt_result["per_tool"].items())
    lines.extend([
        "- Why failures happen:",
    ])
    if prompt_result["failure_slices"]:
        lines.extend(f"  - {reason}: {count}" for reason, count in prompt_result["failure_slices"].items())
    else:
        lines.append("  - No prompt simulation failures recorded.")
    lines.extend([
        "",
        "## Engine Match Evaluation",
        "",
        f"- Engine model: {match_result.get('engine_model_type')}",
        f"- Legality backend: {match_result.get('legality_backend')}",
        f"- Starting position: {match_result['starting_position']}",
        f"- Stockfish available: {match_result['stockfish_available']}",
        f"- Opponent: {match_result['opponent']}",
        f"- Games: {match_result['games']}",
        f"- Plies min/mean/max: {match_result['min_plies']}/{match_result['mean_plies']:.1f}/{match_result['max_observed_plies']}",
        f"- Zero-ply games: {match_result['zero_ply_games']}",
        f"- Engine wins: {match_result['engine_wins']}",
        f"- Engine losses: {match_result['engine_losses']}",
        f"- Drawish: {match_result['drawish']}",
        f"- Engine score rate: {match_result['engine_score_rate']:.3f}",
        "",
        "## Proficiency Conclusion",
        "",
        "Current engine match uses trained basic linear evaluator artifact with python-chess legal move generation and terminal rules. Match opponent is seeded capture/random baseline for lightweight sanity only; training artifacts may report separate same-heuristic baseline metrics. Metrics here are measured locally and are not calibrated ELO; Stockfish availability is reported only as environment smoke check.",
    ])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local SFT router and chess engine")
    parser.add_argument("--model", default="product_demo/poc_models/router_model.json")
    parser.add_argument("--engine-model", default="product_demo/poc_models/chess_engine_model.json")
    parser.add_argument("--eval", default="product_demo/training_data/eval_sft.jsonl")
    parser.add_argument("--out-dir", default="results/demo_eval")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--stockfish-path")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    model = load_model(args.model)
    engine_model = load_model(args.engine_model)
    stockfish = stockfish_status(args.stockfish_path)
    prompt_result = simulate_prompt_cases(model, args.eval if os.path.exists(args.eval) else None, device)
    match_result = run_match_suite(engine_model, args.games, args.max_plies, stockfish)
    search_result = {"engine_backend": "python-chess", "engine_model_type": engine_model.get("model_type"), "opponent": match_result["opponent"], "stockfish": stockfish, "stockfish_available": bool(stockfish.get("available")), "zero_ply_games": match_result["zero_ply_games"]}

    write_json(os.path.join(args.out_dir, "sft_prompt_simulation.json"), prompt_result)
    write_json(os.path.join(args.out_dir, "engine_match_results.json"), match_result)
    write_json(os.path.join(args.out_dir, "engine_backend.json"), search_result)
    write_summary(os.path.join(args.out_dir, "summary.md"), prompt_result, match_result)
    print(json.dumps({"out_dir": args.out_dir, "router_tool_accuracy": prompt_result["router_accuracy"], "router_end_to_end_accuracy": prompt_result["end_to_end_accuracy"], "tool_success_rate": prompt_result["tool_success_rate"], "engine_score_rate": match_result["engine_score_rate"], "zero_ply_games": match_result["zero_ply_games"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
