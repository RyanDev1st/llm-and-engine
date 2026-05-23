from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "src" / "models" / "gemma4"
REQUIRED_FILES = ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors")
MAX_GPU_MB = int(os.environ.get("GEMMA_MAX_GPU_MB", "5120"))


class GemmaService:
    def __init__(self) -> None:
        self.model_path = Path(os.environ.get("GEMMA_MODEL_PATH", DEFAULT_MODEL_PATH)).resolve()
        self.device = os.environ.get("GEMMA_DEVICE", "cpu")
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.load_error: str | None = None
        self.lock = Lock()

    def validate_files(self) -> None:
        missing = [name for name in REQUIRED_FILES if not (self.model_path / name).exists()]
        if missing:
            raise FileNotFoundError(f"missing model files in {self.model_path}: {', '.join(missing)}")

    def load(self) -> None:
        with self.lock:
            if self.model is not None or self.load_error is not None:
                return
            try:
                self.validate_files()
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                    device_map=self._device_map(),
                    torch_dtype=torch.bfloat16,
                )
                self.model.eval()
            except Exception as exc:
                self.load_error = str(exc)
                raise

    def _device_map(self) -> dict | str:
        if self.device == "cpu":
            return {"": "cpu"}
        return "auto"

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
            "device": self.device,
            "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1048576, 1) if cuda else 0,
            "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1048576, 1) if cuda else 0,
            "error": self.load_error,
        }

    def prompt_text(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int, temperature: float) -> dict[str, Any]:
        self.load()
        assert self.model is not None and self.tokenizer is not None
        prompt = self.prompt_text(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
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
