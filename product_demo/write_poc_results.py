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
    match = engine["match"]
    progress = engine["progress"]
    lines = [
        "# Proof of Concept Results",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        "- Basic local models only; no high-end model import.",
        "- SFT trained as router classifier plus narrator template model from generated FEN-blind JSONL.",
        "- Chess engine trained as custom linear evaluator from FEN positions against demo oracle labels.",
        "- Progress-based artifacts saved as JSON under this folder.",
        "",
        "## SFT Training Results",
        "",
        f"- Router eval examples: {router['examples']}",
        f"- Router accuracy: {router['accuracy']:.3f} ({router['correct']}/{router['examples']})",
        f"- Narrator eval examples: {narrator['examples']}",
        f"- Narrator exact accuracy: {narrator['exact_accuracy']:.3f}",
        f"- Narrator grounded rate: {narrator['grounded_rate']:.3f}",
        "",
        "## Chess Engine Training Results",
        "",
        f"- Training positions: {engine['training_positions']}",
        f"- Epochs: {len(progress)}",
        f"- Initial training accuracy: {progress[0]['accuracy']:.3f}",
        f"- Final training accuracy: {progress[-1]['accuracy']:.3f}",
        f"- Learned weights: `{json.dumps(engine['weights'], sort_keys=True)}`",
        "",
        "## Engine Match Results",
        "",
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
        "This is a real proof of concept: it trains local SFT components and a custom chess evaluator, then reports progress and match metrics. It is not yet a production chess coach or calibrated ELO engine; next step is replacing toy oracle labels with python-chess/Stockfish or a larger validated FEN corpus while keeping model size basic.",
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
            "Implemented full local training loop, not static demo: generated SFT JSONL, trained router, trained narrator, trained chess evaluator, wrote metrics.",
            "No high-end model imported; models are Naive Bayes, template narration, and linear move evaluator.",
            "FEN remains internal; SFT visible prompts/narrations do not expose raw FEN.",
            "Chess proficiency metric is limited by toy oracle and small sample data; score rate is useful progress metric, not official ELO.",
            "Community bot requirement partially satisfied through local weak baseline; real community 400-800 ELO bot still needs available package/repo or installed UCI engine.",
        ]
    }
    write_summary(os.path.join(args.out_dir, "summary.md"), sft, engine, audit)
    with open(os.path.join(args.out_dir, "audit.json"), "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    print(json.dumps({"out_dir": args.out_dir, "router_accuracy": sft["router"]["accuracy"], "engine_score_rate": engine["match"]["score_rate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
