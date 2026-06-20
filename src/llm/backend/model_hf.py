"""Transformers serving backend: gemma4_e2b in 4-bit + our LoRA adapter.

Used by the website. The Q4_0 GGUF (export step) is the shipped artifact; this
HF path is the reliable live runtime that loads the same 4-bit weights + adapter.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

LLM_DIR = Path(__file__).resolve().parents[1]
# Base model the adapter was trained on. Default e2b (back-compat); override with
# CHESS_HF_BASE for an E4B adapter (e.g. a downloaded unsloth/gemma-4-E4B-it dir).
# An E4B LoRA on an E2B base = shape mismatch, so this MUST match the training base.
BASE = Path(os.environ.get("CHESS_HF_BASE") or (LLM_DIR / "models" / "gemma4_e2b"))


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
            result = set_peft_model_state_dict(self.model, sd)
            # VERIFY the adapter actually took. A fresh get_peft_model has lora_B=0 (zero
            # effect = pure base model); the trained sd is nonzero. If key names don't line
            # up, set_peft_model_state_dict silently no-ops and we'd serve raw base Gemma
            # (coherent chat but invents <action>/<thought> on the tool loop). Fail loud.
            moved = sum(1 for n, p in self.model.named_parameters()
                        if "lora_B" in n and float(p.detach().abs().sum()) > 0)
            n_unexpected = len(getattr(result, "unexpected_keys", []) or [])
            print(f"[adapter] {len(sd)} tensors from {adapter}; unexpected_keys={n_unexpected}; "
                  f"nonzero lora_B modules={moved}", flush=True)
            if moved == 0:
                raise RuntimeError(
                    "[adapter] NOT applied — 0 lora_B weights took (key mismatch). The server "
                    "would be running raw base Gemma. Fix the adapter load before trusting output.")
        self.model.eval()

    @torch.inference_mode()
    def generate(self, messages: list[dict], max_new_tokens: int, stop: list[str],
                 use_adapter: bool = True) -> str:
        # use_adapter=False runs the SAME base weights with the LoRA turned OFF
        # (PEFT disable_adapter) — lets one loaded model serve both the untrained
        # base and our SFT side by side for the comparison demo.
        from llm_training.chat_format import remap_tool_messages
        # Same remap as training: Gemma drops role="tool", so render tool results
        # as user turns. MUST match data_pipeline or the model sees a new shape.
        messages = remap_tool_messages(messages)
        enc = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]
        can_toggle = hasattr(self.model, "disable_adapter")
        if not use_adapter and can_toggle:
            with self.model.disable_adapter():
                out = self._gen(enc, max_new_tokens, stop)
        else:
            out = self._gen(enc, max_new_tokens, stop)
        text = self.tok.decode(out[0][prompt_len:], skip_special_tokens=True)
        return _truncate(text, stop)

    def count_tokens(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def context_limit(self) -> int:
        # Cap at 8k: Gemma can report far more, but the KV cache must fit the 4060.
        return min(int(getattr(self.model.config, "max_position_embeddings", 8192)), 8192)

    def _gen(self, enc: dict, max_new_tokens: int, stop: list[str]):
        # repetition_penalty + no_repeat_ngram_size were added to kill the greedy loops of
        # the soup era. BUT they corrupt name COPYING: a correct skill name (chess-coach)
        # reuses words already in the prompt (the skills index lists it), so repetition_
        # penalty divides those logits down and pushes the model OFF the right name toward
        # novel garbage. Env-togglable so we can A/B; default 1.0/0 (clean) now that the
        # format is trained and the loops are gone — set CHESS_REP_PENALTY=1.2 to restore.
        rep = float(os.environ.get("CHESS_REP_PENALTY", "1.0"))
        nrn = int(os.environ.get("CHESS_NO_REPEAT_NGRAM", "0"))
        kw = dict(
            **enc, max_new_tokens=max_new_tokens,
            do_sample=self.temperature > 0, temperature=max(self.temperature, 1e-3),
            top_p=0.9, repetition_penalty=rep, no_repeat_ngram_size=nrn,
            pad_token_id=self.tok.pad_token_id)
        # Early-stop at the first action close tag (one action per step), so a ~15-token
        # decision doesn't run the full max_new_tokens (~20x less generation per tool/skill
        # step). _truncate is still applied as the clean-up. Falls back to a full generation
        # on a transformers too old for stop_strings.
        stops = [s for s in (list(stop) + ["</skill>", "</tool>", "</tool_code>"]) if s]
        try:
            return self.model.generate(**kw, stop_strings=stops, tokenizer=self.tok)
        except (TypeError, ValueError):
            return self.model.generate(**kw)


def _truncate(text: str, stop: list[str]) -> str:
    text = text.strip()
    # One action per generation: cut at the FIRST action close tag of ANY kind, INCLUSIVE
    # (keep the close tag). </skill> is an action close exactly like </tool> — leaving it out
    # of this list dropped the tag (the "missing </skill>" artifact) and let skill gens run on.
    ends = [text.index(c) + len(c) for c in ("</tool>", "</tool_code>", "</skill>") if c in text]
    if ends:
        return text[: min(ends)].strip()
    for s in stop:                       # otherwise honor any other caller stop (exclusive)
        if s and s in text:
            text = text[: text.index(s)]
    return text.strip()
