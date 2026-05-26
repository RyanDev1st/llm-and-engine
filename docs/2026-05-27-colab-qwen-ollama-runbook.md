Parent: implementation.md

# Status

Colab plan now targets Ollama model `qwen3.6:27b-q4_K_M` instead of Gemma GGUF/Hugging Face transfer. The notebook installs Ollama, pulls the public model, starts the chess-coach server against Ollama, verifies `/api/state`, then opens Cloudflare Tunnel.

# Scope

Use this for Qwen 27B Q4_K_M chess-coach web testing on Colab hardware. No Hugging Face token is required for the default public Ollama model. Local failed CUDA training is diagnostic only; it did not produce a confirmed fallback adapter.

# Evidence

- Notebook: `src/llm/llm_training/colab_qwen_ollama_host.ipynb`.
- Env template: `docs/2026-05-27-colab-qwen-env-template.txt`.
- Ollama backend: `src/llm/backend/model_ollama.py`.
- Server fallback path now loads Ollama when configured GGUF path does not exist.
- Local tests: `PYTHONPATH=src/llm python -m pytest src/llm/backend/test_colab_config.py -q` passed with 8 tests.

# Next

1. Open Colab site, not Cursor local Jupyter.
2. Upload/open `src/llm/llm_training/colab_qwen_ollama_host.ipynb`.
3. Select GPU runtime. Prefer H100, then A100.
4. Set non-token env values from `docs/2026-05-27-colab-qwen-env-template.txt`:
   - `CHESS_REPO_URL=https://github.com/RyanDev1st/llm-and-engine.git`
   - `CHESS_BRANCH=feat/chess-coach-sft`
   - `OLLAMA_MODEL=qwen3.6:27b-q4_K_M`
   - `CHESS_HOST=127.0.0.1`
   - `CHESS_PORT=7860`
5. Run notebook top to bottom.
6. Confirm it prints GPU memory, repo commit, pulled Ollama model, server ready line, and `/api/state` body.
7. Use Cloudflare Tunnel URL only while testing. Stop tunnel cell when done.
8. If Ollama pull fails because the model tag changes, run `ollama list`/`ollama pull` manually in Colab and update `OLLAMA_MODEL`.
