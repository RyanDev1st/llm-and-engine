from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class Example:
    text: str
    tool_name: str


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_router_examples(path: str) -> list[Example]:
    examples: list[Example] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            if not isinstance(messages, list):
                raise ValueError(f"Messages must be a list in {path}:{line_number}")
            user_text = next((str(item.get("content", "")) for item in messages if isinstance(item, dict) and item.get("role") == "user"), "")
            tool_call = next((item.get("tool_call") for item in messages if isinstance(item, dict) and item.get("role") == "assistant" and isinstance(item.get("tool_call"), dict)), None)
            if tool_call and user_text:
                tool_name = str(tool_call.get("tool_name", ""))
                if not tool_name:
                    raise ValueError(f"Missing tool name in {path}:{line_number}")
                examples.append(Example(user_text, tool_name))
    return examples


def train(examples: list[Example]) -> dict:
    if not examples:
        raise ValueError("No router training examples found.")
    class_counts: Counter[str] = Counter()
    token_counts: dict[str, Counter[str]] = defaultdict(Counter)
    vocabulary: set[str] = set()
    for example in examples:
        class_counts[example.tool_name] += 1
        tokens = tokenize(example.text)
        vocabulary.update(tokens)
        token_counts[example.tool_name].update(tokens)
    return {
        "model_type": "multinomial-naive-bayes-router-v1",
        "class_counts": dict(class_counts),
        "token_counts": {name: dict(counts) for name, counts in token_counts.items()},
        "vocabulary": sorted(vocabulary),
        "example_count": len(examples),
    }


def predict(model: dict, text: str) -> tuple[str, dict[str, float]]:
    class_counts = {name: int(count) for name, count in model["class_counts"].items()}
    token_counts = {name: Counter({token: int(count) for token, count in counts.items()}) for name, counts in model["token_counts"].items()}
    vocabulary = set(model["vocabulary"])
    total_examples = sum(class_counts.values())
    vocab_size = max(1, len(vocabulary))
    scores: dict[str, float] = {}
    tokens = tokenize(text)
    for name, class_count in class_counts.items():
        class_token_total = sum(token_counts[name].values())
        score = math.log(class_count / total_examples)
        for token in tokens:
            score += math.log((token_counts[name][token] + 1) / (class_token_total + vocab_size))
        scores[name] = score
    return max(scores, key=scores.get), scores


def evaluate(model: dict, examples: list[Example]) -> dict:
    correct = 0
    rows = []
    for example in examples:
        predicted, scores = predict(model, example.text)
        ok = predicted == example.tool_name
        correct += int(ok)
        rows.append({"text": example.text, "expected": example.tool_name, "predicted": predicted, "ok": ok, "scores": scores})
    accuracy = correct / len(examples) if examples else 0.0
    return {"examples": len(examples), "correct": correct, "accuracy": accuracy, "rows": rows}


def save_model(model: dict, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a local router on generated SFT tool-call records")
    parser.add_argument("--train", default="product_demo/training_data/train_sft.jsonl")
    parser.add_argument("--eval", default="product_demo/training_data/eval_sft.jsonl")
    parser.add_argument("--out", default="product_demo/trained_router.json")
    parser.add_argument("--predict")
    args = parser.parse_args()

    train_examples = load_router_examples(args.train)
    model = train(train_examples)
    save_model(model, args.out)

    payload = {"model_path": args.out, "train_examples": len(train_examples), "classes": sorted(model["class_counts"])}
    if os.path.exists(args.eval):
        eval_result = evaluate(model, load_router_examples(args.eval))
        payload["eval_examples"] = eval_result["examples"]
        payload["eval_correct"] = eval_result["correct"]
        payload["eval_accuracy"] = eval_result["accuracy"]
    if args.predict:
        predicted, scores = predict(model, args.predict)
        payload["prediction"] = {"text": args.predict, "tool_name": predicted, "scores": scores}
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
