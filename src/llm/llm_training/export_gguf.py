"""Export the trained LoRA to a Q4_0 GGUF for llama.cpp deployment.

Steps: merge adapter into bf16 base (CPU, low-mem) -> convert_hf_to_gguf ->
llama-quantize Q4_0. The llama.cpp tooling is fetched as a prebuilt Windows
release (no C++ build). If conversion fails for the gemma4 architecture, the
web app still serves the same 4-bit weights + adapter via transformers.

Run from repo root (after training):
  python -m llm_training.export_gguf runs/gemma4_chess
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

LLM_DIR = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
BASE = LLM_DIR / "models" / "gemma4_e2b"
RUNTIME = LLM_DIR / "runtime" / "llamacpp"


def merge(adapter: Path, out: Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    print(f"merging {adapter} into base (cpu, bf16) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(BASE), local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": "cpu"})
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print(f"merged -> {out}", flush=True)
    return out


def find_tool(name: str) -> Path | None:
    hits = list(RUNTIME.rglob(name))
    return hits[0] if hits else None


def to_gguf(merged: Path, gguf_f16: Path) -> bool:
    conv = find_tool("convert_hf_to_gguf.py")
    if not conv:
        print("convert_hf_to_gguf.py not found under runtime/llamacpp - place a llama.cpp "
              "release there. Skipping GGUF; transformers path still works.", flush=True)
        return False
    cmd = [sys.executable, str(conv), str(merged), "--outfile", str(gguf_f16), "--outtype", "f16"]
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode == 0


def to_mmproj(base: Path, mmproj_out: Path) -> bool:
    """Export the VISION/AUDIO projector (mmproj) GGUF so the served GGUF keeps image
    reading. Sourced from the BASE model: our LoRA trains text attention only and never
    touches the vision tower, so the base projector IS the full, unsacrificed image
    capability. Verified supported by the bundled stack: convert registers
    Gemma4VisionAudioModel (GEMMA4V/GEMMA4A) and mtmd.dll decodes gemma4v at inference.
    Serve with: llama-mtmd-cli -m <text gguf> --mmproj <this> --image pic.jpg -p '...'.
    Best-effort: vision is a bonus over the text path, so a failure never blocks export."""
    conv = find_tool("convert_hf_to_gguf.py")
    if not conv:
        return False
    cmd = [sys.executable, str(conv), str(base), "--mmproj",
           "--outfile", str(mmproj_out), "--outtype", "f16"]
    print(" ".join(cmd), flush=True)
    try:
        return subprocess.run(cmd).returncode == 0
    except Exception as exc:
        print(f"mmproj export skipped ({exc}); text GGUF still serves (no image input).", flush=True)
        return False


def quantize(gguf_f16: Path, gguf_out: Path, quant_type: str = "Q4_0") -> bool:
    qbin = find_tool("llama-quantize.exe") or find_tool("quantize.exe")
    if not qbin:
        print("llama-quantize not found under runtime/llamacpp. Skipping quantize.", flush=True)
        return False
    cmd = [str(qbin), str(gguf_f16), str(gguf_out), quant_type]
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode == 0


def main() -> None:
    import os
    adapter = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "runs" / "gemma4_chess"
    # Quant target: Q4_0 default; CHESS_GGUF_QUANT=Q5_K_M for better number fidelity.
    quant = os.environ.get("CHESS_GGUF_QUANT", "Q4_0")
    merged = REPO / "runs" / "gemma4_chess_merged"
    gguf_f16 = REPO / "runs" / "gemma4-E2B-chesscoach-f16.gguf"
    gguf_out = REPO / "runs" / f"gemma4-E2B-chesscoach-{quant}.gguf"
    mmproj = REPO / "runs" / "mmproj-gemma4-vision-f16.gguf"
    # Reuse an existing f16 GGUF if present (skip the costly merge + convert).
    if not gguf_f16.exists():
        merge(adapter, merged)
        if not to_gguf(merged, gguf_f16):
            print("GGUF export incomplete; ship via transformers 4-bit + adapter (model_hf).", flush=True)
            return
    else:
        print(f"reusing existing f16 GGUF: {gguf_f16}", flush=True)
    if quantize(gguf_f16, gguf_out, quant):
        print(f"DONE: {gguf_out}", flush=True)
    else:
        print("quantize failed; ship via transformers 4-bit + adapter (model_hf).", flush=True)
    # Vision projector (keeps image reading on the GGUF path). Skip if already present.
    if mmproj.exists():
        print(f"reusing existing mmproj: {mmproj}", flush=True)
    elif to_mmproj(BASE, mmproj):
        print(f"DONE mmproj: {mmproj}", flush=True)
        print(f"  serve+image: llama-mtmd-cli -m {gguf_out.name} --mmproj {mmproj.name} "
              "--image <pic> -p 'describe this image'", flush=True)
    else:
        print("mmproj export skipped — text GGUF serves text-only; for image input serve "
              "via transformers (model_hf, base+adapter) instead.", flush=True)


if __name__ == "__main__":
    main()
