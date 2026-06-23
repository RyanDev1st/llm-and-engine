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

# CHESS_GEN_TRACE=1 prints per-generate decode cost (tokens + wall-time + tok/s + stop path) so a
# latency regression is measurable from the server log without a profiler. Off by default.
_GEN_TRACE = os.environ.get("CHESS_GEN_TRACE", "") not in ("", "0", "false", "False")
_warned_no_earlystop = False


def _warn_no_earlystop_once(exc: Exception) -> None:
    """Loud one-time warning when generate() can't accept stop_strings, so the run falls back to
    NO early-stop — i.e. every step decodes to max_new_tokens. That silent fallback is the prime
    suspect for the 'each step takes dozens of seconds' regression; make it impossible to miss."""
    global _warned_no_earlystop
    if not _warned_no_earlystop:
        _warned_no_earlystop = True
        print(f"[gen] WARNING: stop_strings unsupported here ({exc!r}) — early-stop OFF, every step "
              f"decodes to max_new_tokens. Latency will be high. Check the transformers version.",
              flush=True)


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
        from . import kv_cache
        self._kv = kv_cache.PrefixCache()   # prefix KV reuse across loop steps (self-guarding)

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
        # Prefix KV reuse only on the trusted greedy adapter-on path (the base/sampling paths
        # bypass it — different weights / non-deterministic, so the A/B self-check can't hold).
        reuse_ok = use_adapter and self.temperature <= 0
        if not use_adapter and can_toggle:
            with self.model.disable_adapter():
                out_ids = self._run_plain(enc, max_new_tokens, stop)
        else:
            out_ids = self._gen_cached(enc, max_new_tokens, stop) if reuse_ok \
                else self._run_plain(enc, max_new_tokens, stop)
        text = self.tok.decode(out_ids[prompt_len:], skip_special_tokens=True)
        return _truncate(text, stop)

    def count_tokens(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def context_limit(self) -> int:
        # Cap at 8k: Gemma can report far more, but the KV cache must fit the 4060.
        return min(int(getattr(self.model.config, "max_position_embeddings", 8192)), 8192)

    def _gen_kwargs(self, max_new_tokens: int):
        # repetition_penalty/no_repeat_ngram were the soup-era loop fix; they corrupt name
        # COPYING (a correct skill name reuses prompt words), so default 1.0/0 (clean) now that
        # the format is trained — set CHESS_REP_PENALTY=1.2 / CHESS_NO_REPEAT_NGRAM to restore.
        return dict(
            max_new_tokens=max_new_tokens,
            do_sample=self.temperature > 0, temperature=max(self.temperature, 1e-3),
            top_p=0.9, repetition_penalty=float(os.environ.get("CHESS_REP_PENALTY", "1.0")),
            no_repeat_ngram_size=int(os.environ.get("CHESS_NO_REPEAT_NGRAM", "0")),
            pad_token_id=self.tok.pad_token_id)

    def _run_generate(self, enc: dict, max_new_tokens: int, stop: list[str],
                      past=None, start: int = 0):
        """One model.generate. Returns (full 1-D token sequence, kv-cache-or-None). With
        `past` (a KV cache covering `start` exact prefix tokens) it prefills only the new tail
        — the reuse fast path. Early-stops at the first action close tag (one action/step)."""
        stops = [s for s in (list(stop) + ["</skill>", "</tool>", "</tool_code>"]) if s]
        kw = self._gen_kwargs(max_new_tokens)
        input_ids, attn = enc["input_ids"], enc.get("attention_mask")
        gkw: dict = {"return_dict_in_generate": True, "use_cache": True}
        prefix = None
        if past is not None and start > 0:
            # `past` already holds EXACTLY `start` positions (a pure prefix extension — see
            # kv_cache.reusable), so we extend it with the new tail; NO crop (Gemma's sliding-
            # window cache can't be cropped past its window — that was the live failure).
            prefix = input_ids[:, :start]
            gkw["input_ids"] = input_ids[:, start:]           # prefill only the new tail
            gkw["past_key_values"] = past
            # NB: do NOT pass cache_position — transformers 5.x rejects it ("model_kwargs not used:
            # ['cache_position']") and derives positions from the cache length itself. Passing it was
            # the live Colab failure that disabled reuse every session. The A/B self-check still
            # guards correctness if 5.x ever derives positions differently.
            if attn is not None:
                gkw["attention_mask"] = attn                  # full-length mask (past + new)
        else:
            gkw["input_ids"] = input_ids
            if attn is not None:
                gkw["attention_mask"] = attn
        import time as _time
        _t0 = _time.time()
        _in_len = gkw["input_ids"].shape[1]
        _path = "stop_strings"
        try:
            go = self.model.generate(**gkw, **kw, stop_strings=stops, tokenizer=self.tok)
        except (TypeError, ValueError) as _exc:
            # Retry WITHOUT stop_strings. If THIS succeeds, stop_strings was the culprit -> warn
            # (early-stop off = the latency knot, every step decodes to the cap). If it ALSO raises
            # (e.g. a reuse-cache kwarg), let it propagate to the caller's cache fallback — blaming
            # early-stop here would be a FALSE diagnosis (it was, on the cache_position error).
            go = self.model.generate(**gkw, **kw)
            _warn_no_earlystop_once(_exc)
            _path = "fallback(no-stop)"
        seq = go.sequences[0]
        if prefix is not None:                                # rebuild the full sequence
            seq = torch.cat([prefix[0], seq], dim=0)
        if _GEN_TRACE:                                        # CHESS_GEN_TRACE=1: per-call decode cost
            _gen = int(seq.shape[0]) - (_in_len + (start if prefix is not None else 0))
            _dt = _time.time() - _t0
            print(f"[gen] in={_in_len} cap={max_new_tokens} out={_gen} tok "
                  f"{_dt:.1f}s ({_gen / max(_dt, 1e-3):.1f} tok/s) stop={_path}", flush=True)
        return seq, getattr(go, "past_key_values", None)

    def _run_plain(self, enc: dict, max_new_tokens: int, stop: list[str]):
        return self._run_generate(enc, max_new_tokens, stop)[0]

    def _gen_cached(self, enc: dict, max_new_tokens: int, stop: list[str]):
        """Greedy adapter-on path with prefix KV reuse + the A/B self-check. Until reuse is
        verified, the first reuse OPPORTUNITY runs BOTH paths and returns the TRUSTED full-
        prefill output, enabling reuse only if they match (else reuse is disabled for good).
        Any error falls back to a full prefill. So output is never wrong — worst case = slow."""
        cache = self._kv
        ids = enc["input_ids"][0]
        if not cache.verified:
            # Compute the TRUSTED full prefill first. A reuse-attempt failure below must NOT discard
            # it (the old code did -> _run_plain decoded the whole turn a SECOND time, ~doubling the
            # first multi-step turn's latency). On the first reuse opportunity, A/B-check reuse
            # against this truth; any failure just disables reuse and returns the truth we already have.
            full_seq, full_kv = self._run_generate(enc, max_new_tokens, stop)
            try:
                ln = cache.reusable(ids)
            except Exception:
                ln = 0
            if ln > 0:                                        # first reuse opportunity -> A/B check
                try:
                    reuse_seq, _ = self._run_generate(enc, max_new_tokens, stop, past=cache.kv, start=ln)
                    if _gen_equal(full_seq, reuse_seq, enc["input_ids"].shape[1]):
                        cache.verified = True
                    else:
                        cache.disable("A/B self-check mismatch")
                except Exception as exc:                      # keep full_seq — never recompute it
                    cache.disable(f"reuse error: {type(exc).__name__}: {exc}")
            cache.store(full_seq, full_kv)
            return full_seq
        # verified -> trust reuse; any failure falls back to ONE full prefill
        try:
            ln = cache.reusable(ids)
            seq, kv = self._run_generate(enc, max_new_tokens, stop,
                                         past=cache.kv if ln > 0 else None, start=ln)
            cache.store(seq, kv)
            return seq
        except Exception as exc:
            cache.disable(f"reuse error: {type(exc).__name__}: {exc}")
            return self._run_plain(enc, max_new_tokens, stop)


def _gen_equal(a, b, prompt_len: int) -> bool:
    """The generated portions (past prompt_len) of two sequences are identical."""
    ga, gb = a[prompt_len:], b[prompt_len:]
    n = min(len(ga), len(gb))
    return n > 0 and bool((ga[:n] == gb[:n]).all())


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
