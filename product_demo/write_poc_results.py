from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_json_optional(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    return load_json(path)


def audit_findings(sft: dict, engine: dict, manifest: dict) -> list[str]:
    training = sft.get("training", {})
    readiness = sft.get("readiness", {})
    return [
        f"Generated FEN-blind SFT JSONL from {manifest.get('accepted_positions', 0)} accepted positions with {manifest.get('fen_leakage_records_dropped', 0)} leaked records dropped.",
        f"Trained router path uses {training.get('trainer', 'unknown')} on {training.get('device', 'unknown')}; Qwen/Gemma status is {training.get('qwen_gemma_status', 'not_reported')}.",
        f"Router eval reports tool_accuracy={sft['router']['accuracy']:.3f}, end_to_end_accuracy={sft['router'].get('end_to_end_accuracy', 0.0):.3f}, macro_f1={sft['router'].get('macro_f1', 0.0):.3f}, and review_move argument accuracy={sft['router'].get('argument_accuracy')}.",
        f"Narrator eval reports grounded_rate={sft['narrator']['grounded_rate']:.3f} and factual_accuracy={sft['narrator'].get('factual_accuracy', 0.0):.3f}.",
        f"Chess evaluator uses {engine.get('legality_backend', 'unknown')} legal move generation and same-heuristic baseline metrics, not calibrated ELO.",
        f"Production-valid data is {manifest.get('is_production_valid', False)}; readiness.production_ready={readiness.get('production_ready', False)} with blockers={readiness.get('blockers', [])}.",
    ]


def write_summary(path: str, sft: dict, engine: dict, audit: dict, manifest: dict) -> None:
    router = sft["router"]
    narrator = sft["narrator"]
    training = sft.get("training", {})
    match = engine["match"]
    progress = engine["progress"]
    initial_accuracy = progress[0]["accuracy"] if progress else 0.0
    final_accuracy = progress[-1]["accuracy"] if progress else 0.0
    lines = [
        "# Proof of Concept Results",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        f"- Data mode: {manifest.get('data_mode', 'unknown')}",
        f"- Production-valid data: {manifest.get('is_production_valid', False)}",
        f"- Source rows: {manifest.get('input_rows_total', 0)}",
        f"- Accepted positions: {manifest.get('accepted_positions', 0)}",
        f"- Rejected positions: {manifest.get('rejected_positions', 0)}",
        f"- FEN leakage audit passed: {manifest.get('fen_leakage_audit_passed', False)}",
        f"- Router training path: {training.get('trainer', 'unknown')}",
        f"- Local-only SFT: {training.get('local_only')}",
        f"- Download attempted: {training.get('download_attempted')}",
        f"- Qwen/Gemma status: {training.get('qwen_gemma_status', 'not_reported')}",
        f"- Trainer blockers: {training.get('trainer_blockers', [])}",
        "- Chess engine is intentionally light: custom linear evaluator from FEN positions using python-chess legal move generation.",
        "",
        "## SFT Router Results",
        "",
        f"- Trainer: {training.get('trainer', 'unknown')}",
        f"- Device: {training.get('device', 'unknown')}",
        f"- CUDA available: {training.get('cuda_available', False)}",
        f"- CUDA device: {training.get('cuda_device_name')}",
        f"- Router eval examples: {router['examples']}",
        f"- Router tool-name accuracy: {router['accuracy']:.3f} ({router['correct']}/{router['examples']})",
        f"- Router end-to-end tool-call accuracy: {router.get('end_to_end_accuracy', 0.0):.3f} ({router.get('end_to_end_correct', 0)}/{router['examples']})",
        f"- Router macro F1: {router.get('macro_f1', 0.0):.3f}",
        f"- Review-move argument accuracy: {router.get('argument_accuracy')}",
        f"- Minimum eval support per tool: required={router.get('minimum_support', {}).get('required_per_tool')}, passed={router.get('minimum_support', {}).get('passed')}",
        "",
        "### Where router succeeds",
        "",
    ]
    minimum_support = router.get("minimum_support", {})
    required_support = minimum_support.get("required_per_tool")
    support_by_tool = minimum_support.get("per_tool", {})
    for tool, metrics in router.get("per_tool", {}).items():
        support = metrics.get("support", 0)
        qualifier = f", below minimum={required_support}" if required_support and support < required_support else ""
        lines.append(f"- {tool}: precision={metrics.get('precision', 0.0):.3f}, recall={metrics.get('recall', 0.0):.3f}, f1={metrics.get('f1', 0.0):.3f}, support={support_by_tool.get(tool, support)}{qualifier}")
    lines.extend([
        "",
        "### Why router fails",
        "",
    ])
    if router.get("failure_slices"):
        lines.extend(f"- {reason}: {count}" for reason, count in router["failure_slices"].items())
    else:
        lines.append("- No router failures recorded in this eval set.")
    lines.extend([
        "",
        "## Narrator Results",
        "",
        f"- Narrator eval examples: {narrator['examples']}",
        f"- Narrator exact accuracy: {narrator['exact_accuracy']:.3f}",
        f"- Narrator grounded rate: {narrator['grounded_rate']:.3f}",
        f"- Narrator factual accuracy: {narrator.get('factual_accuracy', 0.0):.3f}",
        "",
        "## Chess Engine Training Results",
        "",
        f"- Legality backend: {engine.get('legality_backend', 'unknown')}",
        f"- Training positions: {engine['training_positions']}",
        f"- Eval positions: {engine.get('eval_positions', 0)}",
        f"- Epochs: {len(progress)}",
        f"- Initial heuristic-agreement accuracy: {initial_accuracy:.3f}",
        f"- Final heuristic-agreement accuracy: {final_accuracy:.3f}",
        f"- Held-out heuristic-agreement accuracy: {engine.get('eval', {}).get('top1_accuracy', 0.0):.3f}",
        f"- Held-out legal prediction rate: {engine.get('eval', {}).get('legal_prediction_rate', 0.0):.3f}",
        f"- Learned weights: `{json.dumps(engine['weights'], sort_keys=True)}`",
        "",
        "## Engine Match Results",
        "",
        f"- Opponent: {match.get('opponent')}",
        f"- Games: {match['games']}",
        f"- Engine wins: {match['engine_wins']}",
        f"- Engine losses: {match['engine_losses']}",
        f"- Drawish: {match['drawish']}",
        f"- Score rate: {match['score_rate']:.3f}",
        "",
        "## Self-Audit",
        "",
    ])
    lines.extend(f"- {item}" for item in audit["findings"])
    lines.extend([
        "",
        "## Production Readiness",
        "",
        f"- Production ready: {sft.get('readiness', {}).get('production_ready', False)}",
        f"- Readiness blockers: {sft.get('readiness', {}).get('blockers', [])}",
        f"- Stockfish available: {training.get('stockfish', {}).get('available', False)}",
        f"- Stockfish UCI ready: {training.get('stockfish', {}).get('uci_ready', False)}",
        "",
        "## Conclusion",
        "",
        "Current artifacts report real measured router, narrator, data, and engine metrics. Production readiness is true only when local Qwen/Gemma SFT, eval support, real Kaggle data, and Stockfish calibration gates all pass.",
    ])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble POC result summary")
    parser.add_argument("--models-dir", default="product_demo/poc_models")
    parser.add_argument("--out-dir", default="results/poc")
    parser.add_argument("--manifest", default="product_demo/training_data/manifest.json")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sft = load_json(os.path.join(args.models_dir, "sft_eval.json"))
    engine = load_json(os.path.join(args.models_dir, "chess_engine_model.json"))
    manifest = load_json_optional(args.manifest)
    audit = {"findings": audit_findings(sft, engine, manifest)}
    write_summary(os.path.join(args.out_dir, "summary.md"), sft, engine, audit, manifest)
    with open(os.path.join(args.out_dir, "audit.json"), "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    print(json.dumps({"out_dir": args.out_dir, "router_tool_accuracy": sft["router"]["accuracy"], "router_end_to_end_accuracy": sft["router"].get("end_to_end_accuracy"), "router_macro_f1": sft["router"].get("macro_f1"), "engine_score_rate": engine["match"]["score_rate"], "production_valid_data": manifest.get("is_production_valid", False)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
