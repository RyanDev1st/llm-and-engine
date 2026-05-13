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
class RouterExample:
    text: str
    tool_name: str


@dataclass(frozen=True)
class NarratorExample:
    tool_name: str
    tool_result: str
    narration: str


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("messages"), list):
                raise ValueError(f"Missing messages list in {path}:{line_number}")
            records.append(record)
    return records


def load_router_examples(path: str) -> list[RouterExample]:
    examples: list[RouterExample] = []
    for record in load_records(path):
        messages = record["messages"]
        user_text = next((str(item.get("content", "")) for item in messages if isinstance(item, dict) and item.get("role") == "user"), "")
        tool_call = next((item.get("tool_call") for item in messages if isinstance(item, dict) and item.get("role") == "assistant" and isinstance(item.get("tool_call"), dict)), None)
        if user_text and tool_call:
            examples.append(RouterExample(user_text, str(tool_call["tool_name"])))
    return examples


def load_narrator_examples(path: str) -> list[NarratorExample]:
    examples: list[NarratorExample] = []
    for record in load_records(path):
        messages = record["messages"]
        tool_msg = next((item for item in messages if isinstance(item, dict) and item.get("role") == "tool"), None)
        assistant_text = next((str(item.get("content", "")) for item in messages if isinstance(item, dict) and item.get("role") == "assistant" and "content" in item), "")
        if tool_msg and assistant_text:
            examples.append(NarratorExample(str(tool_msg.get("name", "")), json.dumps(tool_msg.get("content", {}), sort_keys=True), assistant_text))
    return examples


def train_router(examples: list[RouterExample]) -> dict:
    if not examples:
        raise ValueError("No router examples found.")
    class_counts: Counter[str] = Counter()
    token_counts: dict[str, Counter[str]] = defaultdict(Counter)
    vocabulary: set[str] = set()
    for example in examples:
        class_counts[example.tool_name] += 1
        tokens = tokenize(example.text)
        vocabulary.update(tokens)
        token_counts[example.tool_name].update(tokens)
    return {
        "model_type": "basic-naive-bayes-router-sft-v1",
        "class_counts": dict(class_counts),
        "token_counts": {name: dict(counts) for name, counts in token_counts.items()},
        "vocabulary": sorted(vocabulary),
        "example_count": len(examples),
    }


def predict_router(model: dict, text: str) -> str:
    class_counts = {name: int(count) for name, count in model["class_counts"].items()}
    token_counts = {name: Counter({token: int(count) for token, count in counts.items()}) for name, counts in model["token_counts"].items()}
    vocabulary = set(model["vocabulary"])
    total_examples = sum(class_counts.values())
    scores = {}
    for name, count in class_counts.items():
        total_tokens = sum(token_counts[name].values())
        score = math.log(count / total_examples)
        for token in tokenize(text):
            score += math.log((token_counts[name][token] + 1) / (total_tokens + max(1, len(vocabulary))))
        scores[name] = score
    return max(scores, key=scores.get)


def train_narrator(examples: list[NarratorExample]) -> dict:
    if not examples:
        raise ValueError("No narrator examples found.")
    templates: dict[str, Counter[str]] = defaultdict(Counter)
    for example in examples:
        templates[example.tool_name][example.narration] += 1
    return {
        "model_type": "basic-template-narrator-sft-v1",
        "templates": {name: counts.most_common(3) for name, counts in templates.items()},
        "example_count": len(examples),
    }


def predict_narration(model: dict, tool_name: str, tool_result: dict) -> str:
    templates = model["templates"].get(tool_name) or []
    if templates:
        return templates[0][0]
    return json.dumps(tool_result, sort_keys=True)


def evaluate_router(model: dict, examples: list[RouterExample]) -> dict:
    correct = 0
    rows = []
    for example in examples:
        predicted = predict_router(model, example.text)
        ok = predicted == example.tool_name
        correct += int(ok)
        rows.append({"text": example.text, "expected": example.tool_name, "predicted": predicted, "ok": ok})
    return {"examples": len(examples), "correct": correct, "accuracy": correct / len(examples) if examples else 0.0, "rows": rows}


def evaluate_narrator(model: dict, examples: list[NarratorExample]) -> dict:
    exact = 0
    grounded = 0
    rows = []
    for example in examples:
        tool_result = json.loads(example.tool_result)
        predicted = predict_narration(model, example.tool_name, tool_result)
        exact_ok = predicted == example.narration
        grounded_ok = "fen" not in predicted.lower() and (tool_result.get("status") != "ok" or len(predicted) > 0)
        exact += int(exact_ok)
        grounded += int(grounded_ok)
        rows.append({"tool_name": example.tool_name, "exact_ok": exact_ok, "grounded_ok": grounded_ok, "predicted": predicted, "expected": example.narration})
    return {
        "examples": len(examples),
        "exact_matches": exact,
        "exact_accuracy": exact / len(examples) if examples else 0.0,
        "grounded": grounded,
        "grounded_rate": grounded / len(examples) if examples else 0.0,
        "rows": rows,
    }


def save_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train basic local SFT router and narrator models")
    parser.add_argument("--train", default="product_demo/training_data/train_sft.jsonl")
    parser.add_argument("--eval", default="product_demo/training_data/eval_sft.jsonl")
    parser.add_argument("--out-dir", default="product_demo/poc_models")
    args = parser.parse_args()

    router = train_router(load_router_examples(args.train))
    narrator = train_narrator(load_narrator_examples(args.train))
    router_eval = evaluate_router(router, load_router_examples(args.eval))
    narrator_eval = evaluate_narrator(narrator, load_narrator_examples(args.eval))

    save_json(os.path.join(args.out_dir, "router_model.json"), router)
    save_json(os.path.join(args.out_dir, "narrator_model.json"), narrator)
    save_json(os.path.join(args.out_dir, "sft_eval.json"), {"router": router_eval, "narrator": narrator_eval})
    print(json.dumps({"out_dir": args.out_dir, "router_accuracy": router_eval["accuracy"], "narrator_grounded_rate": narrator_eval["grounded_rate"], "narrator_exact_accuracy": narrator_eval["exact_accuracy"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
