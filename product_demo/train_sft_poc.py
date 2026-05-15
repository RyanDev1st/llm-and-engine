from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass

import torch

TOKEN_RE = re.compile(r"[a-z0-9_]+")
UCI_RE = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b", re.IGNORECASE)
SUPPORTED_TOOLS = {"eval", "best_move", "review_move"}


@dataclass(frozen=True)
class RouterExample:
    text: str
    tool_name: str
    arguments: dict
    internal_fen: str | None = None


@dataclass(frozen=True)
class NarratorExample:
    tool_name: str
    tool_result: str
    narration: str


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


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


def validate_tool_call(tool_call: dict, source: str) -> tuple[str, dict]:
    tool_name = str(tool_call.get("tool_name", ""))
    arguments = tool_call.get("arguments", {})
    if tool_name not in SUPPORTED_TOOLS:
        raise ValueError(f"Unsupported tool {tool_name!r} in {source}")
    if not isinstance(arguments, dict):
        raise ValueError(f"Tool arguments must be object in {source}")
    if tool_name == "review_move" and not isinstance(arguments.get("uci"), str):
        raise ValueError(f"review_move requires uci argument in {source}")
    return tool_name, arguments


def load_router_examples(path: str) -> list[RouterExample]:
    examples: list[RouterExample] = []
    for record_index, record in enumerate(load_records(path), start=1):
        messages = record["messages"]
        user_text = next((str(item.get("content", "")) for item in messages if isinstance(item, dict) and item.get("role") == "user"), "")
        tool_call = next((item.get("tool_call") for item in messages if isinstance(item, dict) and item.get("role") == "assistant" and isinstance(item.get("tool_call"), dict)), None)
        if user_text and tool_call:
            tool_name, arguments = validate_tool_call(tool_call, f"{path}:{record_index}")
            examples.append(RouterExample(user_text, tool_name, arguments, record.get("internal_fen")))
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
    class_counts = Counter(example.tool_name for example in examples)
    class_weights = torch.tensor([len(examples) / (len(labels) * class_counts[label]) for label in labels], device=device)
    model = torch.nn.Linear(len(vocab), len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    progress = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(features)
        loss = torch.nn.functional.cross_entropy(logits, targets, weight=class_weights)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            predicted = logits.argmax(dim=1)
            accuracy = (predicted == targets).float().mean().item()
        progress.append({"epoch": epoch, "loss": float(loss.item()), "accuracy": float(accuracy)})
    weights = model.weight.detach().cpu().tolist()
    bias = model.bias.detach().cpu().tolist()
    return {
        "model_type": "basic-torch-linear-router-sft-v2",
        "trainer": "linear",
        "device": str(device),
        "labels": labels,
        "vocabulary": vocab,
        "weights": weights,
        "bias": bias,
        "example_count": len(examples),
        "class_counts": dict(sorted(class_counts.items())),
    }, progress


def predict_router(model: dict, text: str) -> str:
    if model.get("model_type") == "local-transformers-causal-lm-router-sft-v1":
        raise ValueError("Use predict_router_lm() or evaluate_router(..., device=...) for LM router inference.")
    if model.get("model_type") == "multinomial-naive-bayes-router-v1":
        class_counts = model["class_counts"]
        token_counts = model["token_counts"]
        vocabulary = set(model["vocabulary"])
        vocab_size = len(vocabulary)
        total_examples = sum(class_counts.values())
        counts = Counter(tokenize(text))
        scores = []
        for label in sorted(class_counts):
            label_token_counts = token_counts.get(label, {})
            total_label_tokens = sum(label_token_counts.values())
            score = math.log(class_counts[label] / total_examples)
            denominator = total_label_tokens + vocab_size
            for token, count in counts.items():
                if token in vocabulary:
                    score += count * math.log((label_token_counts.get(token, 0) + 1) / denominator)
            scores.append((score, label))
        return max(scores, key=lambda item: (item[0], item[1]))[1]
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


def extract_uci(text: str) -> str | None:
    match = UCI_RE.search(text)
    return match.group(0).lower() if match else None


def argument_ok(example: RouterExample, predicted_tool: str) -> bool | None:
    if example.tool_name != "review_move":
        return None
    if predicted_tool != "review_move":
        return False
    return extract_uci(example.text) == str(example.arguments.get("uci", "")).lower()


def empty_metrics(labels: list[str]) -> dict:
    return {
        "examples": 0,
        "correct": 0,
        "accuracy": 0.0,
        "end_to_end_correct": 0,
        "end_to_end_accuracy": 0.0,
        "macro_f1": 0.0,
        "per_tool": {label: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for label in labels},
        "confusion_matrix": {label: {inner: 0 for inner in labels} for label in labels},
        "argument_accuracy": None,
        "failure_slices": {},
        "where_it_succeeds": {},
        "minimum_support": {"required_per_tool": 10, "passed": False, "per_tool": {label: 0 for label in labels}},
        "rows": [],
    }


def failure_reason(example: RouterExample, predicted: str, arg_ok: bool | None) -> str:
    if predicted != example.tool_name:
        return f"tool_confusion:{example.tool_name}->{predicted}"
    if arg_ok is False:
        return "argument_mismatch:review_move_uci"
    return "ok"


def metric_summary_from_counts(confusion: dict[str, dict[str, int]], labels: list[str]) -> tuple[dict, float]:
    per_tool = {}
    f1_values = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in labels if actual != label)
        fn = sum(confusion[label][predicted] for predicted in labels if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(confusion[label].values())
        per_tool[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1_values.append(f1)
    return per_tool, sum(f1_values) / len(f1_values) if f1_values else 0.0


def evaluate_router(model: dict, examples: list[RouterExample], device: torch.device | None = None) -> dict:
    labels = sorted(set(model.get("labels", [])) | {example.tool_name for example in examples})
    if not examples:
        return empty_metrics(labels)
    correct = 0
    end_to_end_correct = 0
    rows = []
    confusion = {label: {inner: 0 for inner in labels} for label in labels}
    argument_total = 0
    argument_correct = 0
    failure_counts: Counter[str] = Counter()
    prompt_slice_counts: Counter[str] = Counter()
    prompt_slice_correct: Counter[str] = Counter()
    is_lm = model.get("model_type") == "local-transformers-causal-lm-router-sft-v1"
    lm_predictor = RouterLmPredictor(model, device or torch.device("cpu")) if is_lm else None
    for example in examples:
        raw_generation = None
        parsed_tool_call = None
        parse_error = None
        if is_lm:
            predicted, parsed_arguments, raw_generation, parse_error = predict_router_lm(model, example.text, predictor=lm_predictor)
            predicted = predicted or "eval"
            parsed_tool_call = {"tool_name": predicted, "arguments": parsed_arguments or {}}
        else:
            predicted = predict_router(model, example.text)
            parsed_arguments = {"uci": extract_uci(example.text)} if predicted == "review_move" and extract_uci(example.text) else {}
        ok = predicted == example.tool_name
        arg_ok = argument_ok(example, predicted)
        if is_lm and ok and predicted == "review_move":
            arg_ok = (parsed_arguments or {}).get("uci") == str(example.arguments.get("uci", "")).lower()
        if arg_ok is not None:
            argument_total += 1
            argument_correct += int(arg_ok)
        correct += int(ok)
        end_to_end_correct += int(ok and arg_ok is not False and parse_error is None)
        confusion[example.tool_name][predicted] += 1
        reason = parse_error or failure_reason(example, predicted, arg_ok)
        failure_counts[reason] += 1
        prompt_slice = prompt_family(example.text)
        prompt_slice_counts[prompt_slice] += 1
        prompt_slice_correct[prompt_slice] += int(ok and arg_ok is not False and parse_error is None)
        rows.append({"text": example.text, "expected": example.tool_name, "predicted": predicted, "arguments": example.arguments, "predicted_arguments": parsed_arguments if is_lm else parsed_arguments, "router_ok": ok, "argument_ok": arg_ok, "failure_reason": reason, "prompt_family": prompt_slice, "raw_generation": raw_generation, "parsed_tool_call": parsed_tool_call, "parse_ok": parse_error is None, "parse_error": parse_error})
    per_tool, macro_f1 = metric_summary_from_counts(confusion, labels)
    where = {
        name: {"examples": prompt_slice_counts[name], "accuracy": prompt_slice_correct[name] / prompt_slice_counts[name] if prompt_slice_counts[name] else 0.0}
        for name in sorted(prompt_slice_counts)
    }
    minimum_support_required = 10
    support_by_tool = {label: per_tool[label]["support"] for label in labels}
    return {
        "examples": len(examples),
        "correct": correct,
        "accuracy": correct / len(examples),
        "end_to_end_correct": end_to_end_correct,
        "end_to_end_accuracy": end_to_end_correct / len(examples),
        "macro_f1": macro_f1,
        "per_tool": per_tool,
        "confusion_matrix": confusion,
        "argument_accuracy": argument_correct / argument_total if argument_total else None,
        "argument_examples": argument_total,
        "failure_slices": dict(sorted((reason, count) for reason, count in failure_counts.items() if reason != "ok")),
        "outcome_slices": dict(sorted(failure_counts.items())),
        "where_it_succeeds": where,
        "minimum_support": {
            "required_per_tool": minimum_support_required,
            "passed": all(count >= minimum_support_required for count in support_by_tool.values()),
            "per_tool": support_by_tool,
        },
        "rows": rows,
    }


def prompt_family(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("was my move", "review", "good move", "legal and useful")):
        return "review_move"
    if any(token in lowered for token in ("consider", "candidate", "best", "move should")):
        return "best_move"
    if any(token in lowered for token in ("evaluate", "looking", "position")):
        return "eval"
    return "other"


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


def required_evidence_tokens(tool_name: str, tool_result: dict) -> list[str]:
    if tool_result.get("status") != "ok":
        return [str(tool_result.get("status", ""))]
    if tool_name == "review_move":
        return [str(tool_result.get("played", "")), str(tool_result.get("quality", ""))]
    if tool_name == "best_move":
        return [str(tool_result.get("best_move", ""))]
    if tool_name == "eval":
        return [str(tool_result.get("bucket", ""))]
    return []


def evaluate_narrator(model: dict, examples: list[NarratorExample]) -> dict:
    exact = 0
    grounded = 0
    factual = 0
    rows = []
    for example in examples:
        tool_result = json.loads(example.tool_result)
        predicted = predict_narration(model, example.tool_name, tool_result)
        exact_ok = predicted == example.narration
        grounded_ok = "fen" not in predicted.lower() and (tool_result.get("status") != "ok" or len(predicted) > 0)
        required_tokens = [token for token in required_evidence_tokens(example.tool_name, tool_result) if token]
        factual_ok = grounded_ok and all(token.lower() in predicted.lower() for token in required_tokens)
        exact += int(exact_ok)
        grounded += int(grounded_ok)
        factual += int(factual_ok)
        rows.append({"tool_name": example.tool_name, "exact_ok": exact_ok, "grounded_ok": grounded_ok, "factual_ok": factual_ok, "required_tokens": required_tokens, "predicted": predicted, "expected": example.narration})
    return {
        "examples": len(examples),
        "exact_matches": exact,
        "exact_accuracy": exact / len(examples) if examples else 0.0,
        "grounded": grounded,
        "grounded_rate": grounded / len(examples) if examples else 0.0,
        "factual_matches": factual,
        "factual_accuracy": factual / len(examples) if examples else 0.0,
        "rows": rows,
    }


@dataclass(frozen=True)
class LmFeatures:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def tool_call_json(example: RouterExample) -> str:
    return json.dumps({"tool_name": example.tool_name, "arguments": example.arguments}, sort_keys=True, separators=(",", ":"))


def router_prompt(text: str) -> str:
    return "Route chess-coach request to exactly one JSON tool call. Tools: eval({}), best_move({}), review_move({\"uci\":\"e2e4\"}). Request: " + text.strip() + "\nJSON:"


def first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("invalid_json:no_object")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise ValueError("invalid_json:unterminated_object")


def parse_tool_call(text: str, source: str) -> tuple[dict | None, str | None]:
    try:
        payload = json.loads(first_json_object(text))
    except (json.JSONDecodeError, ValueError):
        return None, "invalid_json"
    if not isinstance(payload, dict):
        return None, "invalid_json"
    if "tool_name" not in payload:
        return payload, "missing_tool_name"
    try:
        tool_name, arguments = validate_tool_call(payload, source)
    except ValueError as exc:
        message = str(exc)
        if "Unsupported tool" in message:
            return payload, "unsupported_tool"
        if "requires uci" in message:
            return payload, "review_move_missing_uci"
        return payload, "bad_arguments"
    return {"tool_name": tool_name, "arguments": arguments}, None


def lm_features(tokenizer, example: RouterExample, device: torch.device, max_length: int) -> LmFeatures:
    prompt = router_prompt(example.text)
    target = tool_call_json(example)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
    if len(target_ids) >= max_length:
        raise ValueError(f"Serialized tool call exceeds max_length={max_length} for supervised LM target.")
    prompt_ids = prompt_ids[-(max_length - len(target_ids)):]
    combined_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    padding = max_length - len(combined_ids)
    attention_mask = [1] * len(combined_ids) + [0] * padding
    input_ids = combined_ids + [pad_id] * padding
    labels = labels + [-100] * padding
    return LmFeatures(
        torch.tensor(input_ids, device=device, dtype=torch.long),
        torch.tensor(attention_mask, device=device, dtype=torch.long),
        torch.tensor(labels, device=device, dtype=torch.long),
    )


def lm_batch(items: list[LmFeatures]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([item.input_ids for item in items]),
        "attention_mask": torch.stack([item.attention_mask for item in items]),
        "labels": torch.stack([item.labels for item in items]),
    }


def chunked(sequence: list, size: int) -> list[list]:
    if size <= 0:
        raise ValueError("batch_size must be >= 1")
    return [sequence[index:index + size] for index in range(0, len(sequence), size)]


def target_length_summary(tokenizer, examples: list[RouterExample]) -> dict:
    target_lengths = [len(tokenizer(tool_call_json(example) + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]) for example in examples]
    return {
        "min": min(target_lengths) if target_lengths else 0,
        "mean": sum(target_lengths) / len(target_lengths) if target_lengths else 0.0,
        "max": max(target_lengths) if target_lengths else 0,
    }


def train_router_lm(examples: list[RouterExample], trainer: str, model_path: str, out_dir: str, epochs: int, learning_rate: float, device: torch.device, max_length: int, max_new_tokens: int, batch_size: int, grad_accum_steps: int) -> tuple[dict, list[dict]]:
    if not examples:
        raise ValueError("No router examples found.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(device)
    model.train()
    features = [lm_features(tokenizer, example, device, max_length) for example in examples]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    progress = []
    batches = chunked(features, batch_size)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        optimizer.zero_grad()
        for batch_index, feature_batch in enumerate(batches, start=1):
            batch = lm_batch(feature_batch)
            output = model(**batch)
            loss = output.loss / grad_accum_steps
            loss.backward()
            total_loss += float(output.loss.item())
            if batch_index % grad_accum_steps == 0 or batch_index == len(batches):
                optimizer.step()
                optimizer.zero_grad()
        progress.append({
            "epoch": epoch,
            "loss": total_loss / len(batches),
            "examples": len(features),
            "batches": len(batches),
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
        })
    artifact_dir = os.path.join(out_dir, "router_lm_model")
    model.save_pretrained(artifact_dir)
    tokenizer.save_pretrained(artifact_dir)
    return {
        "model_type": "local-transformers-causal-lm-router-sft-v1",
        "trainer": trainer,
        "base_model_path": model_path,
        "artifact_dir": artifact_dir,
        "local_files_only": True,
        "download_attempted": False,
        "trained": True,
        "labels": sorted({example.tool_name for example in examples}),
        "example_count": len(examples),
        "max_length": max_length,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "target_length_summary": target_length_summary(tokenizer, examples),
        "readiness_scope": "prototype_full_parameter_sft_not_production_validated",
        "readiness_limits": [
            "full_parameter_finetune_without_adapters",
            "single_node_local_only",
            "no_production_serving_validation",
            "no_calibrated_latency_or_cost_budget",
        ],
    }, progress


class RouterLmPredictor:
    def __init__(self, model_meta: dict, device: torch.device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_meta = model_meta
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_meta["artifact_dir"], local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_meta["artifact_dir"], local_files_only=True).to(device)
        self.model.eval()

    def predict(self, text: str) -> tuple[str | None, dict | None, str, str | None]:
        prompt = router_prompt(text)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.model.generate(**encoded, do_sample=False, max_new_tokens=int(self.model_meta.get("max_new_tokens", 96)), pad_token_id=self.tokenizer.pad_token_id)
        raw = self.tokenizer.decode(generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed, error = parse_tool_call(raw, "generation")
        if error or not parsed:
            return None, None, raw, error
        return parsed["tool_name"], parsed["arguments"], raw, None


def predict_router_lm(model_meta: dict, text: str, device: torch.device | None = None, predictor: RouterLmPredictor | None = None) -> tuple[str | None, dict | None, str, str | None]:
    if predictor is None:
        if device is None:
            raise ValueError("LM prediction requires explicit device or predictor.")
        predictor = RouterLmPredictor(model_meta, device)
    return predictor.predict(text)


def trainer_blockers(trainer: str, model_path: str | None) -> list[str]:
    if trainer == "linear":
        return []
    required = ["transformers", "accelerate"]
    blockers = [f"missing_python_package:{name}" for name in required if not dependency_available(name)]
    if not model_path:
        blockers.append("missing_local_model_path")
    elif not os.path.exists(model_path):
        blockers.append("local_model_path_not_found")
    return blockers


def stockfish_status(stockfish_path: str | None) -> dict:
    candidate = stockfish_path or shutil.which("stockfish")
    if not candidate:
        return {"available": False, "path": None, "uci_ready": False, "blocker": "stockfish_not_found"}
    try:
        process = subprocess.run([candidate], input="uci\nisready\nquit\n", text=True, capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "path": candidate, "uci_ready": False, "blocker": f"stockfish_smoke_failed:{type(exc).__name__}"}
    ready = "uciok" in process.stdout and "readyok" in process.stdout
    return {"available": ready, "path": candidate, "uci_ready": ready, "blocker": None if ready else "stockfish_uci_not_ready"}


def production_readiness(trainer: str, router_eval: dict, training: dict, manifest: dict | None, stockfish: dict) -> dict:
    blockers = []
    is_lm_trainer = trainer in {"qwen", "gemma4"}
    if not is_lm_trainer:
        blockers.append("llm_trainer_not_used")
    if is_lm_trainer and not training.get("local_transformers_sft_completed"):
        blockers.append("local_transformers_sft_not_completed")
    if is_lm_trainer and not training.get("batching_enabled"):
        blockers.append("lm_batching_not_enabled")
    if is_lm_trainer and training.get("qwen_gemma_status") != "trained_local_router_sft":
        blockers.append("llm_training_status_not_complete")
    if not router_eval.get("minimum_support", {}).get("passed"):
        blockers.append("router_eval_minimum_support_not_met")
    if router_eval.get("end_to_end_accuracy", 0.0) < 0.95:
        blockers.append("router_end_to_end_accuracy_below_0.95")
    if not manifest:
        blockers.append("missing_production_manifest")
    elif not manifest.get("is_production_valid"):
        blockers.append("real_kaggle_manifest_not_production_valid")
    if not stockfish.get("available"):
        blockers.append("stockfish_not_available_for_calibration")
    return {"production_ready": not blockers, "blockers": blockers}


def dependency_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_llm_trainer(trainer: str, model_path: str | None) -> None:
    blockers = trainer_blockers(trainer, model_path)
    if blockers:
        raise RuntimeError(f"{trainer} trainer blocked: {', '.join(blockers)}")


def save_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train local SFT router and narrator models")
    parser.add_argument("--train", default="product_demo/training_data/train_sft.jsonl")
    parser.add_argument("--eval", default="product_demo/training_data/eval_sft.jsonl")
    parser.add_argument("--out-dir", default="product_demo/poc_models")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--trainer", choices=["linear", "qwen", "gemma4"], default="linear")
    parser.add_argument("--model-path")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--stockfish-path")
    parser.add_argument("--manifest", default="product_demo/training_data/manifest.json")
    args = parser.parse_args()

    check_llm_trainer(args.trainer, args.model_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    train_router_examples = load_router_examples(args.train)
    if args.trainer == "linear":
        router, progress = train_router_torch(train_router_examples, args.epochs, args.learning_rate, device)
    else:
        router, progress = train_router_lm(train_router_examples, args.trainer, str(args.model_path), args.out_dir, args.epochs, args.learning_rate, device, args.max_length, args.max_new_tokens, args.batch_size, args.grad_accum_steps)
    narrator = train_narrator(load_narrator_examples(args.train))
    router_eval = evaluate_router(router, load_router_examples(args.eval), device)
    narrator_eval = evaluate_narrator(narrator, load_narrator_examples(args.eval))
    manifest = load_json(args.manifest) if args.manifest and os.path.exists(args.manifest) else {}
    stockfish = stockfish_status(args.stockfish_path)
    training = {
        "trainer": args.trainer,
        "model_path": args.model_path,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "router_progress": progress,
        "local_transformers_sft_completed": router.get("model_type") == "local-transformers-causal-lm-router-sft-v1",
        "local_only": bool(router.get("local_files_only", False)),
        "download_attempted": bool(router.get("download_attempted", False)),
        "batching_enabled": bool(router.get("batch_size", 0) > 1 or router.get("grad_accum_steps", 0) > 1),
        "target_length_summary": router.get("target_length_summary"),
        "readiness_scope": router.get("readiness_scope"),
        "readiness_limits": router.get("readiness_limits", []),
        "trainer_blockers": trainer_blockers(args.trainer, args.model_path),
        "qwen_gemma_status": "trained_local_router_sft" if args.trainer in {"qwen", "gemma4"} and router.get("trained") else ("not_run_linear_baseline" if args.trainer == "linear" else "blocked"),
        "stockfish": stockfish,
    }
    readiness = production_readiness(args.trainer, router_eval, training, manifest, stockfish)

    save_json(os.path.join(args.out_dir, "router_model.json"), router)
    save_json(os.path.join(args.out_dir, "narrator_model.json"), narrator)
    save_json(os.path.join(args.out_dir, "sft_eval.json"), {"router": router_eval, "narrator": narrator_eval, "training": training, "readiness": readiness})
    print(json.dumps({"out_dir": args.out_dir, "trainer": args.trainer, "device": str(device), "router_tool_accuracy": router_eval["accuracy"], "router_end_to_end_accuracy": router_eval["end_to_end_accuracy"], "router_macro_f1": router_eval["macro_f1"], "argument_accuracy": router_eval["argument_accuracy"], "narrator_grounded_rate": narrator_eval["grounded_rate"], "narrator_factual_accuracy": narrator_eval["factual_accuracy"], "production_ready": readiness["production_ready"], "production_blockers": readiness["blockers"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
