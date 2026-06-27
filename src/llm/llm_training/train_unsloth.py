"""Unsloth training engine (opt-in via --engine unsloth) — Gemma 4 native, ~2x
faster + ~70% less VRAM than the stock HF path, so E2B@seq1280 fits a T4 with
room (the DDP wall is gone). The proven HF engine in train_cuda.py is untouched
and remains the fallback.

We keep OUR validated logic: the 5x grounding-weighted loss and the train:false
multi-turn masking (both live in data_pipeline). Unsloth supplies only the fused
kernels + memory — its speedups are in the decoder layers, orthogonal to our
lm_head-swap loss, so the two compose. Unsloth's own fused cross-entropy is NOT
used (it can't apply per-token weights); we compute the weighted CE ourselves.

Single-GPU only (no accelerate/DDP here) — Unsloth's memory win removes the need.
UNVALIDATED until the 4h-account smoke: model load + lm_head locate on Unsloth's
module tree is the one integration point a CPU box can't check.
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
from typing import Any

import torch
import torch.nn.functional as F

from .train_gemma4_lora import TrainConfig

# attn-only -> attention adapters only (mirrors lora_target_modules("attn-only")).
_TARGET_FLAGS = {
    "attn-only": dict(finetune_attention_modules=True, finetune_mlp_modules=False),
    "qv": dict(finetune_attention_modules=True, finetune_mlp_modules=False),
    "all-linear": dict(finetune_attention_modules=True, finetune_mlp_modules=True),
}

# Weight names that are SUPPOSED to be created at load (the LoRA adapters + any
# fresh head/norm PEFT adds) — these are not the anomaly.
_OK_REINIT = re.compile(r"lora_|modules_to_save|adapter|\bscore\b", re.IGNORECASE)
# transformers warns like: "Some weights of X were not initialized ... and are
# newly initialized: ['...k_proj...', ...]. You should probably TRAIN this model".
_REINIT_LINE = re.compile(r"newly initialized[:\s]*\[([^\]]*)\]", re.IGNORECASE | re.DOTALL)


class BaseReinitError(RuntimeError):
    """Raised when the loader re-initialized BASE weights (the documented Unsloth
    + QAT-checkpoint anomaly). Training on a partially-random base produces a
    useless adapter — fail loud instead of wasting the GPU slot."""


def _reinit_base_weights(log_text: str) -> list[str]:
    """Parse captured load warnings; return BASE weight names that were newly
    initialized (excludes the expected LoRA/head adapters). Empty == clean load."""
    bad: list[str] = []
    for block in _REINIT_LINE.findall(log_text):
        for name in re.findall(r"['\"]([^'\"]+)['\"]", block):
            if not _OK_REINIT.search(name):
                bad.append(name)
    return bad


@contextlib.contextmanager
def _capture_load_warnings():
    """Collect transformers' modeling-load warnings (where the 'newly initialized'
    notice is emitted) so we can assert the base loaded fully."""
    records: list[str] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _H(level=logging.WARNING)
    loggers = [logging.getLogger("transformers"), logging.getLogger("transformers.modeling_utils")]
    prev = [(lg, lg.level, lg.propagate) for lg in loggers]
    for lg in loggers:
        lg.addHandler(handler)
        lg.setLevel(logging.WARNING)
        lg.propagate = True
    try:
        yield records
    finally:
        for lg, lvl, prop in prev:
            lg.removeHandler(handler)
            lg.setLevel(lvl)
            lg.propagate = prop


def _lm_head_holder(model: Any):
    """Find the module owning lm_head across HF/PEFT/Unsloth Gemma-4 trees."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    inner = getattr(base, "model", None)
    candidates = [base, inner, getattr(base, "language_model", None),
                  getattr(inner, "language_model", None)]
    for c in candidates:
        if c is not None and getattr(c, "lm_head", None) is not None:
            return c
    raise AttributeError("lm_head not found on Unsloth model (smoke must confirm tree)")


def _weighted_ce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, reduction="none")
    return (losses * weights).sum() / weights.sum().clamp(min=1.0)


def _masked_loss(model: Any, batch: dict, labels: torch.Tensor, weights: torch.Tensor | None,
                 unwrapped: Any = None) -> torch.Tensor:
    # Run the model's NORMAL forward and use its real logits — NO lm_head swap. The old
    # swap-to-Identity trick (to skip the 262k-vocab logit materialization) BROKE under
    # DDP: Unsloth's patched Gemma-4 forward reads lm_head.weight directly, so an Identity
    # head -> "'Identity' object has no attribute 'weight'". Paying the ~0.9GB transient
    # logits buys DDP-safety + simplicity AND keeps the per-token grounding weight (5x on
    # fact tokens) — the anti-fabrication mechanism. `unwrapped` is now unused (kept for
    # call-site compatibility); forward goes through `model` so DDP grads all-reduce.
    logits = model(**batch).logits
    labels = labels.to(logits.device)
    flat = logits[..., :-1, :].reshape(-1, logits.size(-1))
    flab = labels[..., 1:].reshape(-1)
    sel = flab != -100
    if not sel.any():
        return logits.sum() * 0.0
    sl = flat[sel]
    slab = flab[sel]
    sw = (weights.to(logits.device)[..., 1:].reshape(-1)[sel] if weights is not None else None)
    # Chunk the fp32 CE upcast so the supervised-token logits don't spike all at once.
    CH = 512
    tot = sl.new_zeros(())
    tw = sl.new_zeros(())
    for i in range(0, sl.size(0), CH):
        li = sl[i:i + CH].float()
        la = slab[i:i + CH]
        if sw is None:
            tot = tot + F.cross_entropy(li, la, reduction="sum")
            tw = tw + la.numel()
        else:
            wi = sw[i:i + CH]
            tot = tot + (F.cross_entropy(li, la, reduction="none") * wi).sum()
            tw = tw + wi.sum()
    return tot / tw.clamp(min=1.0)


@torch.no_grad()
def _evaluate(model: Any, batches: list[dict], device: torch.device) -> float:
    model.eval()
    total_loss = total_tok = 0.0
    for raw in batches:
        b = {k: v.to(device) for k, v in raw.items()}
        labels = b.pop("labels")
        b.pop("weights", None)  # val loss unweighted, for comparability
        n = int((labels != -100).sum().item())
        total_loss += _masked_loss(model, b, labels, None).item() * n
        total_tok += n
    model.train()
    return total_loss / total_tok if total_tok else float("nan")


def _print_gpu_mem(tag: str) -> None:
    """Per-GPU allocated/reserved — reveals whether device_map='balanced' actually
    SPLIT the model or piled it on one card (the OOM-on-GPU1 imbalance)."""
    if not torch.cuda.is_available():
        return
    parts = [f"GPU{i} {torch.cuda.memory_allocated(i)/1e9:.1f}/{torch.cuda.memory_reserved(i)/1e9:.1f}GB"
             for i in range(torch.cuda.device_count())]
    print(f"[mem:{tag}] " + "  ".join(parts), flush=True)


def _free_multimodal_towers(model) -> None:
    """We train TEXT only, but FastModel loads the full Gemma-4 multimodal stack — the
    vision + audio encoders sit in VRAM unused (~GBs). Detach them so that memory is
    reclaimed on BOTH shards. Safe: with no pixel/audio inputs the text forward never
    enters those branches. Defensive — any failure just leaves them loaded. Also prints
    the top-level module names so if the towers are named differently we can target them."""
    import gc
    targets = ("vision_tower", "audio_tower", "vision_model", "audio_model", "vision",
               "audio", "multi_modal_projector", "audio_projector", "vision_projector",
               "embed_vision", "embed_audio")
    roots = []
    seen = set()
    cur = model
    for _ in range(3):  # model -> .model -> .model.model
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        roots.append(cur)
        cur = getattr(cur, "model", None)
    freed = []
    for root in roots:
        try:
            names = [n for n, _ in root.named_children()]
        except Exception:
            continue
        for name in names:
            if name in targets:
                try:
                    setattr(root, name, None)
                    freed.append(f"{type(root).__name__}.{name}")
                except Exception:
                    pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[mem] freed multimodal towers: {freed or 'none found'}", flush=True)
    # one-time structure hint so we can target exact names if 'none found'
    try:
        top = [n for n, _ in model.named_children()]
        inner = getattr(model, "model", None)
        sub = [n for n, _ in inner.named_children()] if inner is not None else []
        print(f"[mem] module tree top={top} inner={sub}", flush=True)
    except Exception:
        pass


def _flush_mem() -> None:
    """Release cached/garbage GPU memory — called after a heavy save/eval so a long
    multi-session run doesn't accumulate fragmentation toward an OOM."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _hub_env() -> None:
    """Force the battle-tested LFS upload path. huggingface_hub now defaults NEW private
    repos to xet storage, and Kaggle images frequently lack `hf_xet` -> upload_folder
    errors while the plain-API create_repo succeeds. That exact split left the ckpt repo
    holding only .gitattributes for a whole session (the silent-failure bug). Disabling
    xet + hf_transfer makes uploads boring and reliable."""
    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# Files PEFT's save_pretrained writes that must NOT go to the Hub. The generated
# adapter model-card (README.md) carries base_model = the LOCAL load path, which the
# Hub's model-card validator REJECTS ("not a valid model id from hf.co/models"). It is
# a card only — resume reads adapter_model.safetensors + trainer_state.pt and serve
# reads the adapter weights; nothing reads README.md. Skip every .md so the folder
# upload has NO Hub-validated file left and cannot be rejected.
_HUB_IGNORE = ["README.md", "*.md"]


def _hub_push(local_dir, path_in_repo: str) -> bool:
    """Mirror a saved folder (checkpoint/ or best/) to a private HF Hub repo, OFF the
    kernel. A full E4B epoch is ~30h > Kaggle's 12h ceiling, so every session is
    SIGKILLed at the wall — and a SIGKILL is not a Python exception, so the in-process
    crash-save can't fire and a timed-out committed run's local Output is not guaranteed
    to persist. Pushing each checkpoint to the Hub makes resume bulletproof across
    sessions AND accounts (the Hub is the shared store). Opt-in via CHESS_CKPT_REPO.
    Returns True on success. Best-effort (never crashes training) BUT LOUD: full
    traceback on give-up + retries, because a silently-broken mirror = a lost session.
    Skips *.md (the only Hub-validated files) — see _HUB_IGNORE."""
    import os
    import time
    import traceback
    repo = os.environ.get("CHESS_CKPT_REPO")
    if not repo:
        return False
    _hub_env()
    last = None
    for attempt in range(1, 4):
        try:
            from huggingface_hub import create_repo, upload_folder
            create_repo(repo, repo_type="model", private=True, exist_ok=True)
            upload_folder(folder_path=str(local_dir), repo_id=repo, repo_type="model",
                          path_in_repo=path_in_repo, commit_message=f"{path_in_repo} sync",
                          ignore_patterns=_HUB_IGNORE)
            print(f"  [hub] pushed {path_in_repo} -> {repo}", flush=True)
            return True
        except Exception as exc:
            last = exc
            print(f"  [hub] push attempt {attempt}/3 failed: {exc!r}", flush=True)
            time.sleep(5 * attempt)
    print(f"  [hub] push GAVE UP for {path_in_repo}: {last!r} — full traceback:", flush=True)
    traceback.print_exc()
    return False


# --- ASYNC off-kernel mirror ----------------------------------------------------------
# The synchronous upload_folder blocks the train loop ~80-160s per save (222MB over the
# network) — at SAVE_EVERY=50 over 1000 steps that's ~30-50 min of pure waiting. So the
# slow network push runs on a BACKGROUND worker while training continues. The LOCAL save
# (save_pretrained + trainer_state) stays synchronous (fast, crash/resume-critical); only
# the Hub mirror is deferred. We SNAPSHOT the saved dir (fast local copy) before queuing so
# the NEXT save can overwrite the original freely without corrupting an in-flight upload.
# Single worker = uploads serialize (don't fight for bandwidth). _hub_drain() flushes the
# queue at exit so the final best/ + checkpoint actually land. Main process only.
_UPLOAD_Q = None
_UPLOAD_WORKER = None


def _uploader_worker() -> None:
    import shutil
    while True:
        job = _UPLOAD_Q.get()
        try:
            if job is None:
                return                       # sentinel: queue drained, stop
            snap_dir, tag = job
            try:
                _hub_push(snap_dir, tag)     # the real (retrying) upload
            finally:
                shutil.rmtree(snap_dir, ignore_errors=True)
        finally:
            _UPLOAD_Q.task_done()


def _hub_push_async(local_dir, tag: str) -> None:
    """Snapshot local_dir and upload it on a background thread so training never blocks on
    the network. Falls back to a synchronous push if the snapshot fails or the queue is
    backed up (slow net) — a checkpoint is never silently dropped."""
    import os
    import queue as _queue
    import shutil
    import tempfile
    import threading
    from pathlib import Path
    global _UPLOAD_Q, _UPLOAD_WORKER
    if not os.environ.get("CHESS_CKPT_REPO"):
        return
    # snapshot on the SAME filesystem (fast) so the next save can overwrite local_dir
    snap = tempfile.mkdtemp(prefix=f"hubpush_{tag}_", dir=str(Path(local_dir).parent))
    try:
        shutil.copytree(local_dir, snap, dirs_exist_ok=True)
    except Exception as exc:
        print(f"  [hub] snapshot failed ({exc}); pushing {tag} synchronously", flush=True)
        shutil.rmtree(snap, ignore_errors=True)
        _hub_push(local_dir, tag)
        return
    if _UPLOAD_Q is None:
        _UPLOAD_Q = _queue.Queue(maxsize=4)
        _UPLOAD_WORKER = threading.Thread(target=_uploader_worker, daemon=True)
        _UPLOAD_WORKER.start()
    try:
        _UPLOAD_Q.put((snap, tag), timeout=0.5)
        print(f"  [hub] queued {tag} async — training continues", flush=True)
    except _queue.Full:
        print(f"  [hub] upload queue full (slow net?); pushing {tag} synchronously", flush=True)
        try:
            _hub_push(snap, tag)
        finally:
            shutil.rmtree(snap, ignore_errors=True)


def _hub_drain(timeout: float = 1200.0) -> None:
    """Block until queued uploads finish — call at end of training / before exit so the
    final best/ + checkpoint actually reach the Hub. No-op if nothing was queued."""
    global _UPLOAD_Q, _UPLOAD_WORKER
    if _UPLOAD_Q is None or _UPLOAD_WORKER is None:
        return
    print("  [hub] draining async uploads before exit...", flush=True)
    _UPLOAD_Q.put(None)                       # sentinel after all pending jobs (FIFO)
    _UPLOAD_WORKER.join(timeout)
    if _UPLOAD_WORKER.is_alive():
        print("  [hub] drain TIMED OUT — some uploads may be incomplete (local saves intact)", flush=True)
    else:
        print("  [hub] async uploads drained.", flush=True)
    _UPLOAD_Q = None
    _UPLOAD_WORKER = None


def _hub_selftest() -> None:
    """Prove the mirror works in the FIRST MINUTE by exercising the EXACT real push path
    — not a toy file. A plain-file test passed last time while the real folder push died
    at step 50 on README model-card validation. So here we build a folder that mimics a
    saved PEFT adapter (a weights-like file + a README.md carrying the same invalid
    `base_model: /local/path` metadata PEFT generates), push it through _hub_push, and
    VERIFY the weights file (not the README) landed in the repo. If the real path is
    broken for ANY reason — xet, README validation, auth, transport — this RAISES now,
    before the run invests hours. Opt-in via CHESS_CKPT_REPO."""
    import os
    import tempfile
    from pathlib import Path
    repo = os.environ.get("CHESS_CKPT_REPO")
    if not repo:
        print("[hub] CHESS_CKPT_REPO unset — off-kernel mirror DISABLED (local Output only)", flush=True)
        return
    _hub_env()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        # mimic exactly what _save_ckpt uploads: a weights blob, a config, a state file,
        # AND the poison README that broke the real push.
        (p / "adapter_model.safetensors").write_bytes(b"\x00" * 64)
        (p / "adapter_config.json").write_text('{"base_model_name_or_path": "/kaggle/working/local/path"}')
        (p / "trainer_state.pt").write_bytes(b"\x00" * 16)
        (p / "README.md").write_text(
            "---\nbase_model: /kaggle/working/local/path\nlibrary_name: peft\n---\nselftest")
        ok = _hub_push(p, "_selftest")
    if not ok:
        raise RuntimeError(
            "[hub] off-kernel mirror SELF-TEST FAILED — the real checkpoint push path is broken "
            "(see the traceback above). Refusing to train a multi-session run with no failsafe. "
            "Fix the mirror (token write scope? repo?) before launching.")
    # confirm the weights actually landed (README is intentionally skipped).
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id=repo, repo_type="model")
    if "_selftest/adapter_model.safetensors" not in files:
        raise RuntimeError(
            f"[hub] self-test push reported OK but the weights file is NOT in {repo} "
            f"(files seen: {sorted(files)[:10]}). Mirror is unreliable — fix before training.")
    print(f"[hub] self-test OK -> {repo} (REAL folder round-trip verified; mirror is LIVE)", flush=True)


def _save_ckpt(model, optimizer, scheduler, update, epoch, best_val, config, tag="checkpoint") -> None:
    """Persist adapter + trainer state so a crash / 12h timeout / next session (even a
    different Kaggle account) can resume. Adapter save is the critical part; optimizer
    state is best-effort (resume degrades to a warm-start if it can't reload)."""
    out = config.output_dir / tag
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))                       # adapter weights (critical)
    state = {"update": update, "epoch": epoch, "best_val": best_val, "seed": config.seed}
    try:
        state["optimizer"] = optimizer.state_dict()
        state["scheduler"] = scheduler.state_dict()
    except Exception as exc:                              # paged 8-bit state can be odd
        print(f"  [ckpt] optimizer/scheduler state not saved ({exc}); resume will warm-start", flush=True)
    torch.save(state, out / "trainer_state.pt")
    print(f"  [ckpt] saved {tag} @ update {update}", flush=True)
    _hub_push_async(out, tag)                             # mirror off-kernel, NON-BLOCKING (drained at exit)
    _flush_mem()


def _load_ckpt(model, optimizer, scheduler, config) -> tuple[int, int, float]:
    """Resume from output_dir/checkpoint if --resume. Returns (start_update, start_epoch,
    best_val). Adapter reload FAILURE raises (a multi-day run must NOT silently restart
    from zero); optimizer reload failure only warm-starts (loud warning)."""
    if not getattr(config, "resume", False):
        return 0, 0, float("inf")
    ckpt = config.output_dir / "checkpoint"
    state_path = ckpt / "trainer_state.pt"
    if not state_path.exists():
        print(f"  [resume] no checkpoint at {ckpt} — starting fresh", flush=True)
        return 0, 0, float("inf")
    # adapter weights (critical) — fail loud if the format doesn't load
    from peft import set_peft_model_state_dict
    adapter_sd = None
    for fname in ("adapter_model.safetensors", "adapter_model.bin"):
        p = ckpt / fname
        if p.exists():
            if fname.endswith(".safetensors"):
                from safetensors.torch import load_file
                adapter_sd = load_file(str(p))
            else:
                adapter_sd = torch.load(str(p), map_location="cpu")
            break
    if adapter_sd is None:
        raise FileNotFoundError(f"[resume] {ckpt} has trainer_state but no adapter weights — refusing to restart from zero")
    # Load + VERIFY it actually took. A name/prefix mismatch on the Unsloth-wrapped PEFT
    # model could match ZERO keys and silently resume with a RANDOM adapter (disaster on a
    # multi-day run). So snapshot a trained LoRA weight, load, and assert it changed.
    lora_params = {n: p for n, p in model.named_parameters() if "lora_" in n.lower()}
    if not lora_params:
        raise RuntimeError("[resume] model has no lora_ params — wrong model/targets, refusing to resume")
    probe_name = next(iter(lora_params))
    before = lora_params[probe_name].detach().float().clone()
    result = set_peft_model_state_dict(model, adapter_sd)
    after = dict(model.named_parameters())[probe_name].detach().float()
    moved = not torch.equal(before, after)
    n_unexpected = len(getattr(result, "unexpected_keys", []) or [])
    n_loaded = len(adapter_sd) - n_unexpected
    print(f"  [resume] adapter keys: {n_loaded}/{len(adapter_sd)} loaded, probe_changed={moved}", flush=True)
    if n_loaded == 0 or not moved:
        raise RuntimeError(
            f"[resume] adapter load matched ~0 keys (unexpected={n_unexpected}) or weights didn't change "
            "— key/prefix mismatch. Refusing to resume from a random adapter. Fix the load path before a real run.")
    state = torch.load(state_path, map_location="cpu")
    try:
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
    except Exception as exc:
        print(f"  [resume] optimizer/scheduler not restored ({exc}); warm-starting from saved adapter", flush=True)
    su, se, bv = state.get("update", 0), state.get("epoch", 0), state.get("best_val", float("inf"))
    print(f"  [resume] continuing from update {su} (epoch {se}, best_val {bv:.4f})", flush=True)
    _flush_mem()
    return su, se, bv


def _loop(model, train_examples, val_batches, optimizer, scheduler, device, pad_id, config,
          total_updates, start_update=0, start_epoch=0, best_val=float("inf"),
          accelerator=None, unwrapped=None, save_model=None) -> tuple:
    import time
    from .data_pipeline import make_batches
    losses: list[float] = []
    val_history: list[dict] = []
    update, epoch = start_update, start_epoch
    save_every = config.save_every or config.eval_every  # crash-safety cadence
    # DDP: only rank 0 prints/saves; backward goes through accelerator so grads all-reduce.
    # save_model is the UNWRAPPED model (save_pretrained on the DDP wrapper writes nothing).
    main = (accelerator is None) or accelerator.is_main_process
    saver = save_model if save_model is not None else model
    eval_target = unwrapped if unwrapped is not None else model

    def _backward(loss):
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()

    _t_prev = time.time()
    try:
        while update < total_updates:
            # DDP: vary the shuffle seed by RANK too, so each replica sees different rows.
            seed = config.seed + epoch + (1000 * accelerator.process_index if accelerator is not None else 0)
            batches = make_batches(train_examples, config.batch_size, pad_id, config.shuffle, seed)
            for micro_idx in range(0, len(batches), config.grad_accum_steps):
                optimizer.zero_grad()
                accum_loss = 0.0
                chunk = batches[micro_idx: micro_idx + config.grad_accum_steps]
                for raw in chunk:
                    b = {k: v.to(device) for k, v in raw.items()}
                    labels = b.pop("labels")
                    weights = b.pop("weights", None)
                    loss = _masked_loss(model, b, labels, weights, unwrapped=unwrapped) / len(chunk)
                    _backward(loss)
                    accum_loss += loss.item()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.grad_clip)
                optimizer.step()
                scheduler.step()
                update += 1
                losses.append(accum_loss)
                _now = time.time()
                _dt = _now - _t_prev
                _t_prev = _now
                _eta_h = _dt * (total_updates - update) / 3600.0
                if main:
                    print(f"upd {update}/{total_updates} ep {epoch+1} loss={accum_loss:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e} {_dt:.1f}s/step eta~{_eta_h:.1f}h", flush=True)
                if val_batches and update % config.eval_every == 0:
                    vloss = _evaluate(eval_target, val_batches, device)
                    val_history.append({"update": update, "val_loss": vloss})
                    if main:
                        print(f"  val_loss={vloss:.4f}", flush=True)
                        if vloss < best_val:
                            best_val = vloss
                            saver.save_pretrained(str(config.output_dir / "best"))
                            print(f"  saved best (val_loss={vloss:.4f})", flush=True)
                            _hub_push_async(config.output_dir / "best", "best")   # mirror off-kernel, NON-BLOCKING
                            _flush_mem()           # release memory once a best is banked
                if main and update % save_every == 0:   # crash/timeout-safe resumable checkpoint
                    _save_ckpt(saver, optimizer, scheduler, update, epoch, best_val, config)
                torch.cuda.empty_cache()
                if update >= total_updates:
                    break
            epoch += 1
    except (KeyboardInterrupt, Exception) as exc:
        # 12h cutoff / OOM / disconnect: persist progress so the next session resumes.
        if main:
            print(f"\n[interrupted: {type(exc).__name__}: {exc}] saving checkpoint before exit...", flush=True)
            try:
                _save_ckpt(saver, optimizer, scheduler, update, epoch, best_val, config)
            except Exception as save_exc:
                print(f"[checkpoint-on-exit FAILED: {save_exc}]", flush=True)
            _hub_drain()        # flush the queued upload (incl. this exit save) before re-raising
        raise
    if main:
        _hub_drain()            # normal completion: ensure the final best/ + checkpoint land
    return losses, val_history


def run_unsloth_training(config: TrainConfig) -> dict:
    from unsloth import FastModel  # MUST precede transformers import for patches
    from .train_cuda import _materialize
    from .optim_sched import build_optimizer, build_scheduler

    # Multi-GPU (e.g. Kaggle 2xT4): SHARD the model across GPUs (model-parallel) so
    # E4B @ seq 1664 fits where one 16 GB T4 cannot (forward alone OOMs single-card).
    # This is device_map="balanced" — NOT DDP, which REPLICATES the model per GPU and
    # OOMs (memory ddp-not-viable-e2b-t4). On 1 GPU, omit device_map so Unsloth uses
    # its normal single-device placement.
    n_gpus = torch.cuda.device_count()
    load_kwargs = dict(model_name=str(config.model_path), max_seq_length=config.max_seq_len,
                       load_in_4bit=config.load_in_4bit, dtype=None)
    # DO NOT pass device_map by default. Unsloth loads on a single GPU and its gradient
    # checkpointing frees per-layer activations. Passing device_map="balanced" adds
    # accelerate DISPATCH HOOKS that BREAK that checkpointing -> every layer's activation
    # is retained -> ~8GB activation at seq 1664 even for tiny E2B (6.5GB weights) -> OOM.
    # That single mistake caused the whole OOM saga. Multi-GPU sharding is opt-in ONLY via
    # CHESS_GPU_CAP_GIB (experimental; still hook-bound, expect the same issue).
    cap = os.environ.get("CHESS_GPU_CAP_GIB")
    if n_gpus > 1 and cap:
        load_kwargs["device_map"] = "balanced"
        load_kwargs["max_memory"] = {i: f"{cap}GiB" for i in range(n_gpus)}
        print(f"[unsloth] EXPERIMENTAL forced shard: balanced + max_memory {cap}GiB/GPU", flush=True)
    else:
        print(f"[unsloth] single-GPU load (no device_map) — gradient checkpointing active", flush=True)
    with _capture_load_warnings() as load_log:
        model, processor = FastModel.from_pretrained(**load_kwargs)
    _print_gpu_mem("after load")
    # Guard the documented anomaly: Unsloth re-initializing base k/v on a QAT
    # checkpoint it doesn't fully recognize. Use an Unsloth-published base
    # (unsloth/gemma-3n-E4B-it...) to avoid it; this asserts it didn't happen.
    reinit = _reinit_base_weights("\n".join(load_log))
    if reinit:
        raise BaseReinitError(
            f"loader re-initialized {len(reinit)} BASE weights (e.g. {reinit[:4]}). "
            "Training would start from a partially-random base. Point --model at an "
            "Unsloth-published base (unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit), not a "
            "raw QAT checkpoint.")
    # Gemma4 is multimodal -> FastModel returns a PROCESSOR whose patched __call__
    # treats the first positional arg as `images` (text= is keyword-only). Our
    # text-only pipeline calls tokenizer(delta_text, return_offsets_mapping=True),
    # so unwrap to the raw text tokenizer which has the normal (text, ...) call.
    tokenizer = getattr(processor, "tokenizer", None) or processor
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Reclaim the unused vision/audio encoders BEFORE LoRA-wrapping (we train text only).
    try:
        _free_multimodal_towers(model)
    except Exception as exc:
        print(f"[mem] tower-free skipped ({exc})", flush=True)
    _print_gpu_mem("after tower-free")
    flags = _TARGET_FLAGS.get(config.lora_targets, _TARGET_FLAGS["attn-only"])
    model = FastModel.get_peft_model(
        model, r=config.lora_rank, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,  # was hardcoded 0.0 -> silently ignored config's 0.05;
                                           # light dropout regularizes the train(0.076)/val(3.1) gap
        bias="none", random_state=config.seed, finetune_vision_layers=False,
        finetune_language_layers=True, use_gradient_checkpointing="unsloth", **flags)
    # No KV cache during training (we never generate here) — frees activation memory
    # at long seq. Gradient checkpointing already implies this, but set it explicitly.
    for cfg_obj in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if cfg_obj is not None:
            cfg_obj.use_cache = False
    # CRITICAL for our CUSTOM loop: the base is frozen (only LoRA trains), so the embedding
    # output doesn't require grad -> torch.utils.checkpoint silently NO-OPS -> every layer's
    # activation is retained -> ~8GB for batch=1 seq=1664 even on tiny E2B (6.5GB weights)
    # -> OOM. This is THE line the proven HF path (train_cuda.py) has and this one lacked.
    # Registering an input-grad hook makes checkpointing actually fire. Also force-enable
    # HF checkpointing + train mode as belt-and-suspenders (Unsloth's flag alone wasn't
    # engaging under our manual forward).
    model.train()
    model.enable_input_require_grads()
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception as exc:
        print(f"[ckpt] gradient_checkpointing_enable skipped ({exc}) — relying on unsloth flag", flush=True)

    train_examples = _materialize(config.data_path, config, tokenizer, label="train")
    if not train_examples:
        raise ValueError("no training examples produced; check --data-path")
    val_examples = _materialize(config.val_path, config, tokenizer, label="val") if config.val_path else []
    from .data_pipeline import make_batches
    eval_subset = val_examples[: config.max_val_examples]
    val_batches = make_batches(eval_subset, config.batch_size, tokenizer.pad_token_id, False, config.seed) if eval_subset else []

    micro_per_epoch = math.ceil(len(train_examples) / config.batch_size)
    updates_per_epoch = math.ceil(micro_per_epoch / config.grad_accum_steps)
    total_updates = config.max_steps or updates_per_epoch * config.epochs
    warmup = max(1, int(config.warmup_ratio * total_updates))
    print(f"[unsloth] train_ex={len(train_examples)} total_updates={total_updates} warmup={warmup}", flush=True)

    optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    scheduler = build_scheduler(optimizer, warmup, total_updates)
    # Resume BEFORE any DDP wrap (set_peft_model_state_dict needs the raw PEFT model).
    # Hub PULL is done once in the notebook (Cell 6.6), before accelerate launch, so the
    # 2 DDP ranks just read the staged local checkpoint (no concurrent-download race).
    start_update, start_epoch, best_val = _load_ckpt(model, optimizer, scheduler, config)
    if start_update >= total_updates:
        print(f"[unsloth] already at/over target ({start_update}/{total_updates}); nothing to do. "
              "Raise --max-steps to train further.", flush=True)

    # DDP (Kaggle 2xT4): launched via `accelerate launch`/`torchrun` -> num_processes>1.
    # Each rank holds a FULL replica (data-parallel), trains a DISJOINT data shard, and
    # all-reduces the (tiny LoRA) grads each step -> ~2x throughput. Single-process: the
    # Accelerator is a no-op and the path is byte-identical to before.
    accelerator = unwrapped = save_model = None
    device = torch.device("cuda:0")
    try:
        from accelerate import Accelerator
        from accelerate.utils import DistributedDataParallelKwargs
        # v4.1: masking the thought from loss makes some LoRA params receive zero grad on certain
        # batches (data-dependent sparsity). 2-GPU DDP then errors ("parameters that were not used
        # in producing loss") unless told to expect it. find_unused_parameters=True handles it
        # (small per-step graph traversal; harmless for the single-process path, which skips DDP).
        acc = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
        if acc.num_processes > 1:
            train_examples = train_examples[acc.process_index::acc.num_processes]  # disjoint shard
            model, optimizer = acc.prepare(model, optimizer)
            accelerator = acc
            unwrapped = acc.unwrap_model(model)
            save_model = unwrapped
            device = acc.device
            if acc.is_main_process:
                print(f"[ddp] {acc.num_processes} processes; per-rank shard={len(train_examples)} "
                      f"examples; device={device}", flush=True)
    except Exception as exc:  # accelerate missing / single-process -> plain single-GPU
        print(f"[ddp] not active ({exc}); single-GPU path", flush=True)

    # Verify the off-kernel mirror BEFORE burning hours — fail fast on a broken failsafe.
    if (accelerator is None) or accelerator.is_main_process:
        _hub_selftest()

    losses, val_history = _loop(model, train_examples, val_batches, optimizer, scheduler,
                                device, tokenizer.pad_token_id, config, total_updates,
                                start_update=start_update, start_epoch=start_epoch, best_val=best_val,
                                accelerator=accelerator, unwrapped=unwrapped, save_model=save_model)

    is_main = (accelerator is None) or accelerator.is_main_process
    if accelerator is not None:
        accelerator.wait_for_everyone()
    if is_main:
        (save_model or model).save_pretrained(str(config.output_dir))
        processor.save_pretrained(str(config.output_dir))
        result = {"engine": "unsloth", "total_updates": total_updates, "training_completed": True,
                  "final_train_loss": losses[-1] if losses else None, "train_losses": losses,
                  "val_losses": val_history, "output_dir": str(config.output_dir),
                  "ddp_world": (accelerator.num_processes if accelerator is not None else 1)}
        (config.output_dir / "train_result.json").write_text(json.dumps(result, indent=2, default=str))
        return result
    return {"engine": "unsloth", "rank": accelerator.process_index, "training_completed": True}
