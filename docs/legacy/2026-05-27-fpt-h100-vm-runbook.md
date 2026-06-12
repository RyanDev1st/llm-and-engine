Parent: implementation.md

# Status

FPT H100 VM path is primary when Colab credits are low. Use a GPU Virtual Machine, not AI Notebook, so Ollama, Python server, Cloudflare Tunnel, tmux, and training can run without notebook lifecycle limits.

# Scope

This runbook targets Ubuntu GPU VM with public SSH access. It prepares repo code, Ollama Qwen hosting, tunnel, and bounded v1.2 training launch commands. It does not store secrets or expose inbound web ports beyond SSH; web access uses Cloudflare quick tunnel.

# Evidence

- Bootstrap script: `scripts/fpt_qwen_vm_bootstrap.sh`.
- Model: `qwen3.6:27b-q4_K_M` via Ollama.
- Branch: `feat/chess-coach-sft` from `https://github.com/RyanDev1st/llm-and-engine.git`.
- Prior local verification: `PYTHONPATH=src/llm python -m pytest src/llm/backend/test_colab_config.py src/llm/llm_training/test_training_defaults.py -q` passed with 13 tests.

# Next

1. Create FPT GPU VM with 1xH100, Ubuntu 22.04 or 24.04, SSH key auth, port 22 only.
2. SSH from Git Bash:
   ```bash
   ssh root@<PUBLIC_IP>
   ```
3. Bootstrap VM:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/RyanDev1st/llm-and-engine/feat/chess-coach-sft/scripts/fpt_qwen_vm_bootstrap.sh -o fpt_qwen_vm_bootstrap.sh
   bash fpt_qwen_vm_bootstrap.sh
   ```
4. Start web server:
   ```bash
   tmux new -s chess-server "$HOME/llm-and-engine/run_fpt_server.sh"
   ```
5. In a second SSH tab, start tunnel:
   ```bash
   tmux new -s chess-tunnel "$HOME/llm-and-engine/run_fpt_tunnel.sh"
   ```
6. Open Cloudflare tunnel URL printed by the tunnel session.
7. In a third SSH tab, start bounded training only after server path is confirmed:
   ```bash
   tmux new -s chess-train "$HOME/llm-and-engine/run_fpt_train.sh"
   ```
8. Watch sessions:
   ```bash
   tmux attach -t chess-server
   tmux attach -t chess-tunnel
   tmux attach -t chess-train
   ```
9. Stop/delete VM when done. FPT warns power-off may still bill until VM deletion.
