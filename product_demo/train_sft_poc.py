from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import torch

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


def build_vocab(examples: list[RouterExample]) -> list[str]:
    vocab = sorted({token for example in examples for token in tokenize(example.text)})
    if not vocab:
        raise ValueError("No router vocabulary found.")
    return vocab


def vectorize(text: str, token_to_index: dict[str, int], device: torch.device) -> torch.Tensor:
    values = torch.zeros(len(token_to_index), device=device)
    counts = Counter(tokenize(text))
    for token, count in counts.items():
        if token in token_to_index:
            values[token_to_index[token]] = float(count)
    return values


def train_router_torch(examples: list[RouterExample], epochs: int, learning_rate: float, device: torch.device) -> tuple[dict, list[dict]]:
    if not examples:
        raise ValueError("No router examples found.")
    vocab = build_vocab(examples)
    labels = sorted({example.tool_name for example in examples})
    token_to_index = {token: index for index, token in enumerate(vocab)}
    label_to_index = {label: index for index, label in enumerate(labels)}
    features = torch.stack([vectorize(example.text, token_to_index, device) for example in examples])
    targets = torch.tensor([label_to_index[example.tool_name] for example in examples], device=device, dtype=torch.long)
    model = torch.nn.Linear(len(vocab), len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    progress = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(features)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            predicted = logits.argmax(dim=1)
            accuracy = (predicted == targets).float().mean().item()
        progress.append({"epoch": epoch, "loss": float(loss.item()), "accuracy": float(accuracy)})
    weights = model.weight.detach().cpu().tolist()
    bias = model.bias.detach().cpu().tolist()
    return {
        "model_type": "basic-torch-linear-router-sft-v1",
        "device": str(device),
        "labels": labels,
        "vocabulary": vocab,
        "weights": weights,
        "bias": bias,
        "example_count": len(examples),
    }, progress


def predict_router(model: dict, text: str) -> str:
    vocab = model["vocabulary"]
    labels = model["labels"]
    token_to_index = {token: index for index, token in enumerate(vocab)}
    counts = Counter(tokenize(text))
    scores = []
    for label_index, label in enumerate(labels):
        score = float(model["bias"][label_index])
        for token, count in counts.items():
            if token in token_to_index:
                score += float(model["weights"][label_index][token_to_index[token]]) * count
        scores.append((score, label))
    return max(scores, key=lambda item: (item[0], item[1]))[1]


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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    train_router_examples = load_router_examples(args.train)
    router, progress = train_router_torch(train_router_examples, args.epochs, args.learning_rate, device)
    narrator = train_narrator(load_narrator_examples(args.train))
    router_eval = evaluate_router(router, load_router_examples(args.eval))
    narrator_eval = evaluate_narrator(narrator, load_narrator_examples(args.eval))
    training = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "router_progress": progress,
    }

    save_json(os.path.join(args.out_dir, "router_model.json"), router)
    save_json(os.path.join(args.out_dir, "narrator_model.json"), narrator)
    save_json(os.path.join(args.out_dir, "sft_eval.json"), {"router": router_eval, "narrator": narrator_eval, "training": training})
    print(json.dumps({"out_dir": args.out_dir, "device": str(device), "router_accuracy": router_eval["accuracy"], "narrator_grounded_rate": narrator_eval["grounded_rate"], "narrator_exact_accuracy": narrator_eval["exact_accuracy"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
