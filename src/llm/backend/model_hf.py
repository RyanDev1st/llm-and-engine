"""Transformers serving backend: gemma4_e2b in 4-bit + our LoRA adapter.

Used by the website. The Q4_0 GGUF (export step) is the shipped artifact; this
HF path is the reliable live runtime that loads the same 4-bit weights + adapter.
"""
from __future__ import annotations

from pathlib import Path

import torch

LLM_DIR = Path(__file__).resolve().parents[1]
BASE = LLM_DIR / "models" / "gemma4_e2b"


class HFModel:
    def __init__(self, base: str | Path = BASE, adapter: str | Path | None = None,
                 temperature: float = 0.6) -> None:
        from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
        self.temperature = temperature
        self.tok = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True)
        # Inference path: no grads/optimizer, so fit the whole 4-bit model on GPU
        # (cpu-offload was for training only; it strands pad/eos tensors on meta
        # at generate() time and crashes torch.isin).
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(base), local_files_only=True, quantization_config=quant,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0})
        if adapter:
            # Manual adapter attach: PeftModel.from_pretrained re-runs accelerate
            # dispatch and trips a Params4bit/_is_hf_initialized bug on this stack.
            # Build matching LoraConfig and load state_dict directly instead.
            import json as _json
            from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
            from safetensors.torch import load_file
            cfg_d = _json.loads((Path(adapter) / "adapter_config.json").read_text())
            lora = LoraConfig(
                task_type="CAUSAL_LM", r=cfg_d["r"], lora_alpha=cfg_d["lora_alpha"],
                lora_dropout=cfg_d["lora_dropout"], target_modules=cfg_d["target_modules"],
                bias=cfg_d.get("bias", "none"))
            self.model = get_peft_model(self.model, lora)
            sd = load_file(str(Path(adapter) / "adapter_model.safetensors"))
            set_peft_model_state_dict(self.model, sd)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, messages: list[dict], max_new_tokens: int, stop: list[str],
                 use_adapter: bool = True) -> str:
        # use_adapter=False runs the SAME base weights with the LoRA turned OFF
        # (PEFT disable_adapter) — lets one loaded model serve both the untrained
        # base and our SFT side by side for the comparison demo.
        enc = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]
        can_toggle = hasattr(self.model, "disable_adapter")
        if not use_adapter and can_toggle:
            with self.model.disable_adapter():
                out = self._gen(enc, max_new_tokens)
        else:
            out = self._gen(enc, max_new_tokens)
        text = self.tok.decode(out[0][prompt_len:], skip_special_tokens=True)
        return _truncate(text, stop)

    def _gen(self, enc: dict, max_new_tokens: int):
        return self.model.generate(
            **enc, max_new_tokens=max_new_tokens,
            do_sample=self.temperature > 0, temperature=max(self.temperature, 1e-3),
            top_p=0.9, pad_token_id=self.tok.pad_token_id)


def _truncate(text: str, stop: list[str]) -> str:
    text = text.strip()
    if "</tool>" in text:  # always close after the first tool call
        return text[: text.index("</tool>") + len("</tool>")]
    for s in stop:
        if s and s in text:
            text = text[: text.index(s)]
    return text.strip()
