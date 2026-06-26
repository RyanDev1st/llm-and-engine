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

# Per-generate decode cost (tokens + wall-time + tok/s) printed to the server log so latency is
# measurable without a profiler. Default ON (one line per call, cheap) — set CHESS_GEN_TRACE=0 to mute.
_GEN_TRACE = os.environ.get("CHESS_GEN_TRACE", "1") not in ("0", "false", "False")
_ACTION_STOPS = ("</skill>", "</tool>", "</tool_code>")   # one-action-per-step close tags


def _build_stopper(tok, stops):
    """Version-robust early-stop: a StoppingCriteria that halts the moment any stop string appears in
    the freshly generated tail. Replaces transformers' `stop_strings=` kwarg, which silently no-ops on
    some transformers versions — THE Colab latency knot (every step ran to max_new_tokens, ~24s, even
    though eos already stopped the <pad> flood). Decodes only the last few tokens, so it's O(1)/step."""
    from transformers import StoppingCriteria, StoppingCriteriaList
    stops = [s for s in stops if s]
    if not stops:
        return None

    class _Stop(StoppingCriteria):
        def __call__(self, input_ids, scores=None, **kw) -> bool:
            tail = tok.decode(input_ids[0][-24:], skip_special_tokens=True)
            return any(s in tail for s in stops)

    return StoppingCriteriaList([_Stop()])


def _stop_token_ids(tok) -> list[int]:
    """Token ids that must END a generation: <eos>, the chat TURN-ENDER (<end_of_turn> / <turn|>),
    and <pad>. Gemma's generation_config sometimes omits the turn-ender from eos, so after a complete
    reply the model keeps decoding and floods <pad> until the token cap — the 'NUL' run in the log and
    the dozens-of-seconds latency on EVERY reply. Stopping on the turn-ender halts at the real end;
    <pad> is included so a degenerate pad run stops immediately instead of running to the cap."""
    ids: set[int] = set()
    for v in (getattr(tok, "eos_token_id", None), getattr(tok, "pad_token_id", None)):
        if v is not None:
            ids.add(int(v))
    unk = getattr(tok, "unk_token_id", None)
    for name in ("<end_of_turn>", "<turn|>"):
        i = tok.convert_tokens_to_ids(name)
        if isinstance(i, int) and i >= 0 and i != unk:
            ids.add(i)
    return sorted(ids)


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
        # Stop set = the model's configured eos UNION the turn-ender + pad, so a finished reply ends
        # instead of flooding <pad> to the cap (the NUL run = the real latency knot, all modes).
        gc_eos = getattr(self.model.generation_config, "eos_token_id", None)
        eos = set(gc_eos) if isinstance(gc_eos, (list, tuple)) else ({gc_eos} if gc_eos is not None else set())
        eos |= set(_stop_token_ids(self.tok))
        self._eos_ids = sorted(int(i) for i in eos if i is not None)
        print(f"[gen] stop ids = {self._eos_ids} (turn-ender + pad included)", flush=True)
        from . import kv_cache
        self._kv = kv_cache.PrefixCache()   # prefix KV reuse across loop steps (self-guarding)

    @torch.inference_mode()
    def generate(self, messages: list[dict], max_new_tokens: int, stop: list[str],
                 use_adapter: bool = True, on_token=None) -> str:
        # use_adapter=False runs the SAME base weights with the LoRA turned OFF
        # (PEFT disable_adapter) — lets one loaded model serve both the untrained
        # base and our SFT side by side for the comparison demo.
        import os
        from llm_training.chat_format import remap_tool_messages
        # Same remap as training: Gemma drops role="tool", so render tool results
        # as user turns. MUST match data_pipeline or the model sees a new shape.
        messages = remap_tool_messages(messages)
        # v4.1: native reasoning. CHESS_NATIVE_THINK=1 turns on the enable_thinking signal
        # (must match how the served model was trained) AND keeps special tokens in the
        # decode so the native <|channel>thought block survives for _split_reasoning to
        # lift to the panel (skip_special_tokens would delete the markers, leaking the
        # thought into the chat bubble). OFF by default -> v4 serve unchanged.
        native = os.environ.get("CHESS_NATIVE_THINK", "0") not in ("0", "false", "False")
        try:
            enc = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                enable_thinking=native)
        except TypeError:  # older template without the kwarg
            enc = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]
        can_toggle = hasattr(self.model, "disable_adapter")
        # Streaming path: emit each token via on_token AS it's generated (the UI fills live during the
        # long T4 decode — thinking + reply both stream). Used for the primary loop; base/coverage
        # runs pass no on_token. Runs the same disable_adapter toggle for the base side.
        if on_token is not None:
            if not use_adapter and can_toggle:
                with self.model.disable_adapter():
                    return self._gen_stream(enc, max_new_tokens, stop, on_token)
            return self._gen_stream(enc, max_new_tokens, stop, on_token)
        # Prefix KV reuse only on the trusted greedy adapter-on path (the base/sampling paths
        # bypass it — different weights / non-deterministic, so the A/B self-check can't hold).
        reuse_ok = use_adapter and self.temperature <= 0
        if not use_adapter and can_toggle:
            with self.model.disable_adapter():
                out_ids = self._run_plain(enc, max_new_tokens, stop)
        else:
            out_ids = self._gen_cached(enc, max_new_tokens, stop) if reuse_ok \
                else self._run_plain(enc, max_new_tokens, stop)
        text = self.tok.decode(out_ids[prompt_len:], skip_special_tokens=not native)
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
        kw = dict(
            max_new_tokens=max_new_tokens,
            repetition_penalty=float(os.environ.get("CHESS_REP_PENALTY", "1.0")),
            no_repeat_ngram_size=int(os.environ.get("CHESS_NO_REPEAT_NGRAM", "0")),
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self._eos_ids)   # stop at the turn-ender + pad, not just <eos>
        # Only pass sampling params when actually sampling — at temp 0 (greedy) transformers 5.x
        # warns "generation flags not valid: ['temperature','top_p']" and ignores them.
        if self.temperature > 0:
            kw.update(do_sample=True, temperature=max(self.temperature, 1e-3), top_p=0.9)
        else:
            kw["do_sample"] = False
        return kw

    def _run_generate(self, enc: dict, max_new_tokens: int, stop: list[str],
                      past=None, start: int = 0):
        """One model.generate. Returns (full 1-D token sequence, kv-cache-or-None). With
        `past` (a KV cache covering `start` exact prefix tokens) it prefills only the new tail
        — the reuse fast path. Early-stops at the first action close tag (one action/step)."""
        stops = [s for s in (list(stop) + list(_ACTION_STOPS)) if s]
        kw = self._gen_kwargs(max_new_tokens)
        crit = _build_stopper(self.tok, stops)        # version-robust early-stop (replaces stop_strings)
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
        if crit is not None:
            gkw["stopping_criteria"] = crit
        import time as _time
        _t0 = _time.time()
        _in_len = gkw["input_ids"].shape[1]
        go = self.model.generate(**gkw, **kw)
        seq = go.sequences[0]
        if prefix is not None:                                # rebuild the full sequence
            seq = torch.cat([prefix[0], seq], dim=0)
        if _GEN_TRACE:                                        # per-call decode cost (server log)
            _gen = int(seq.shape[0]) - (_in_len + (start if prefix is not None else 0))
            _dt = _time.time() - _t0
            print(f"[gen] in={_in_len} cap={max_new_tokens} out={_gen} tok "
                  f"{_dt:.1f}s ({_gen / max(_dt, 1e-3):.1f} tok/s)", flush=True)
        return seq, getattr(go, "past_key_values", None)

    def _gen_stream(self, enc: dict, max_new_tokens: int, stop: list[str], on_token) -> str:
        """Stream tokens live via on_token while generating (the UI fills during the long T4 decode).
        Uses the SAME version-robust StoppingCriteria as _run_generate (so streamed steps early-stop
        at </tool>/</skill> too, not run to the cap), with generate() on a worker thread feeding a
        TextIteratorStreamer. Returns the truncated full text, same contract as the non-stream path."""
        from threading import Thread
        from transformers import TextIteratorStreamer
        import time as _time
        stops = [s for s in (list(stop) + list(_ACTION_STOPS)) if s]
        crit = _build_stopper(self.tok, stops)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        gkw: dict = {"input_ids": enc["input_ids"], "use_cache": True, "streamer": streamer,
                     **self._gen_kwargs(max_new_tokens)}
        if enc.get("attention_mask") is not None:
            gkw["attention_mask"] = enc["attention_mask"]
        if crit is not None:
            gkw["stopping_criteria"] = crit
        box: dict = {}

        def _run() -> None:
            try:
                with torch.inference_mode():
                    self.model.generate(**gkw)
            except Exception as exc:                          # surface to the main thread below
                box["err"] = exc

        _t0 = _time.time()
        Thread(target=_run, daemon=True).start()
        text, n = "", 0
        for piece in streamer:                               # blocks until each new token is ready
            text += piece
            n += 1
            try:
                on_token(piece)
            except Exception:
                pass                                         # a client disconnect must not kill the gen
        if "err" in box:
            raise box["err"]
        if _GEN_TRACE:
            _dt = _time.time() - _t0
            print(f"[gen] stream cap={max_new_tokens} out~{n} tok {_dt:.1f}s "
                  f"({n / max(_dt, 1e-3):.1f} tok/s)", flush=True)
        return _truncate(text, stop)

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
