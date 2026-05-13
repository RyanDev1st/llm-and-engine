from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_summary(path: str, sft: dict, engine: dict, audit: dict) -> None:
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
        "- Basic local models only; no high-end model import.",
        "- SFT trained as CUDA/CPU linear router plus narrator template model from generated FEN-blind JSONL.",
        "- Chess engine trained as custom linear evaluator from FEN positions using python-chess legal move generation.",
        "- Progress-based artifacts saved as JSON under this folder.",
        "",
        "## SFT Training Results",
        "",
        f"- Device: {training.get('device', 'unknown')}",
        f"- CUDA available: {training.get('cuda_available', False)}",
        f"- CUDA device: {training.get('cuda_device_name')}",
        f"- Router eval examples: {router['examples']}",
        f"- Router accuracy: {router['accuracy']:.3f} ({router['correct']}/{router['examples']})",
        f"- Narrator eval examples: {narrator['examples']}",
        f"- Narrator exact accuracy: {narrator['exact_accuracy']:.3f}",
        f"- Narrator grounded rate: {narrator['grounded_rate']:.3f}",
        "",
        "## Chess Engine Training Results",
        "",
        f"- Legality backend: {engine.get('legality_backend', 'unknown')}",
        f"- Training positions: {engine['training_positions']}",
        f"- Eval positions: {engine.get('eval_positions', 0)}",
        f"- Epochs: {len(progress)}",
        f"- Initial training accuracy: {initial_accuracy:.3f}",
        f"- Final training accuracy: {final_accuracy:.3f}",
        f"- Held-out eval accuracy: {engine.get('eval', {}).get('top1_accuracy', 0.0):.3f}",
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
    ]
    lines.extend(f"- {item}" for item in audit["findings"])
    lines.extend([
        "",
        "## Conclusion",
        "",
        "Current product path uses python-chess legality and measured CUDA/router/chess metrics. It is still not calibrated ELO because Stockfish/UCI and full Kaggle data are not installed in this environment.",
    ])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble POC result summary")
    parser.add_argument("--models-dir", default="product_demo/poc_models")
    parser.add_argument("--out-dir", default="results/poc")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sft = load_json(os.path.join(args.models_dir, "sft_eval.json"))
    engine = load_json(os.path.join(args.models_dir, "chess_engine_model.json"))
    audit = {
        "findings": [
            "Implemented local training loop: generated SFT JSONL, trained CUDA router, trained narrator, trained chess evaluator, wrote measured metrics.",
            "No high-end model imported; models are torch linear router, template narration, and linear move evaluator.",
            "FEN remains internal; SFT visible prompts/narrations do not expose raw FEN.",
            "Chess legality now uses python-chess, including castling, en passant, terminal rules, and legal move generation.",
            "Kaggle CLI and Stockfish executable are not installed; current metrics use the available sample FEN CSV and static-eval baseline, not calibrated ELO.",
        ]
    }
    write_summary(os.path.join(args.out_dir, "summary.md"), sft, engine, audit)
    with open(os.path.join(args.out_dir, "audit.json"), "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    print(json.dumps({"out_dir": args.out_dir, "router_accuracy": sft["router"]["accuracy"], "engine_score_rate": engine["match"]["score_rate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
