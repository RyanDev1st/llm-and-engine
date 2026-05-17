from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "src" / "models" / "gemma4_q4"
REQUIRED_FILES = ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors")


class GemmaService:
    def __init__(self) -> None:
        self.model_path = Path(os.environ.get("GEMMA_MODEL_PATH", DEFAULT_MODEL_PATH)).resolve()
        self.allow_cpu_offload = os.environ.get("GEMMA_ALLOW_CPU_OFFLOAD") == "1"
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.load_error: str | None = None
        self.lock = Lock()

    def validate_files(self) -> None:
        missing = [name for name in REQUIRED_FILES if not (self.model_path / name).exists()]
        if missing:
            raise FileNotFoundError(f"missing model files in {self.model_path}: {', '.join(missing)}")

    def load_config(self) -> Any:
        config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
        if self.allow_cpu_offload:
            config.quantization_config = self.offload_quantization_config()
        return config

    def offload_quantization_config(self) -> dict[str, Any]:
        data = json.loads((self.model_path / "config.json").read_text(encoding="utf-8"))
        quant = dict(data.get("quantization_config") or {})
        quant["llm_int8_enable_fp32_cpu_offload"] = True
        return quant

    def load_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"local_files_only": True, "device_map": "auto", "torch_dtype": "auto"}
        if self.allow_cpu_offload:
            kwargs["config"] = self.load_config()
            kwargs["quantization_config"] = self.offload_quantization_config()
        return kwargs

    def load(self) -> None:
        with self.lock:
            if self.model is not None or self.load_error is not None:
                return
            try:
                self.validate_files()
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **self.load_kwargs())
                self.model.eval()
            except Exception as exc:
                self.load_error = str(exc)
                raise

    def status(self) -> dict[str, Any]:
        cuda = torch.cuda.is_available()
        config = getattr(self.model, "config", None)
        return {
            "loaded": self.model is not None,
            "model_path": str(self.model_path),
            "model_type": getattr(config, "model_type", None),
            "architectures": getattr(config, "architectures", None),
            "cuda_available": cuda,
            "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
            "allow_cpu_offload": self.allow_cpu_offload,
            "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1048576, 1) if cuda else 0,
            "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1048576, 1) if cuda else 0,
            "error": self.load_error,
        }

    def prompt_text(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int, temperature: float) -> dict[str, Any]:
        self.load()
        assert self.model is not None and self.tokenizer is not None
        prompt = self.prompt_text(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        tokens = max(1, min(int(max_new_tokens), 512))
        temp = max(0.0, min(float(temperature), 2.0))
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=tokens,
                do_sample=temp > 0,
                temperature=temp if temp > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return {"reply": reply, "status": self.status()}


SERVICE = GemmaService()
