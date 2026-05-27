#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${CHESS_REPO_URL:-https://github.com/RyanDev1st/llm-and-engine.git}"
BRANCH="${CHESS_BRANCH:-feat/chess-coach-sft}"
WORKDIR="${CHESS_WORKDIR:-$HOME/llm-and-engine}"
MODEL="${OLLAMA_MODEL:-qwen3.6:27b-q4_K_M}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

log() { printf '\n==> %s\n' "$*"; }

log "GPU"
nvidia-smi || true

log "apt packages"
sudo apt-get update
sudo apt-get install -y git curl wget python3-venv python3-pip tmux

if ! command -v ollama >/dev/null 2>&1; then
  log "install ollama"
  curl -fsSL https://ollama.com/install.sh | sh
fi

log "start ollama"
if ! pgrep -x ollama >/dev/null 2>&1; then
  nohup ollama serve > "$HOME/ollama.log" 2>&1 &
fi
for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    break
  fi
  sleep 2
  if [ "$i" = 60 ]; then
    echo "ollama did not start; see $HOME/ollama.log" >&2
    exit 1
  fi
done

log "pull $MODEL"
ollama pull "$MODEL"

log "repo"
if [ -d "$WORKDIR/.git" ]; then
  git -C "$WORKDIR" fetch origin "$BRANCH"
  git -C "$WORKDIR" checkout "$BRANCH"
  git -C "$WORKDIR" pull --ff-only origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO_URL" "$WORKDIR"
fi
git -C "$WORKDIR" rev-parse HEAD

log "python env"
cd "$WORKDIR"
$PYTHON_BIN -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install torch transformers accelerate peft bitsandbytes datasets sentencepiece protobuf python-chess

log "cloudflared"
if [ ! -x "$HOME/cloudflared" ]; then
  wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O "$HOME/cloudflared"
  chmod +x "$HOME/cloudflared"
fi

cat > "$WORKDIR/run_fpt_server.sh" <<'RUNSERVER'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONPATH="$PWD/src/llm"
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.6:27b-q4_K_M}"
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export CHESS_HOST="${CHESS_HOST:-127.0.0.1}"
export CHESS_PORT="${CHESS_PORT:-7860}"
export CHESS_GGUF_PATH="${CHESS_GGUF_PATH:-/tmp/not-used.gguf}"
python -m backend.server
RUNSERVER
chmod +x "$WORKDIR/run_fpt_server.sh"

cat > "$WORKDIR/run_fpt_tunnel.sh" <<'RUNTUNNEL'
#!/usr/bin/env bash
set -euo pipefail
"$HOME/cloudflared" tunnel --url http://127.0.0.1:7860
RUNTUNNEL
chmod +x "$WORKDIR/run_fpt_tunnel.sh"

cat > "$WORKDIR/run_fpt_train.sh" <<'RUNTRAIN'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONPATH="$PWD/src/llm"
python -m llm_training.run_train --max-steps "${MAX_STEPS:-500}" --rank "${LORA_RANK:-4}" --targets "${LORA_TARGETS:-qv}" --grad-accum "${GRAD_ACCUM:-1}" --output "${TRAIN_OUTPUT:-gemma4_chess_fpt}"
RUNTRAIN
chmod +x "$WORKDIR/run_fpt_train.sh"

log "ready"
echo "Server: tmux new -s chess-server '$WORKDIR/run_fpt_server.sh'"
echo "Tunnel: tmux new -s chess-tunnel '$WORKDIR/run_fpt_tunnel.sh'"
echo "Train:  tmux new -s chess-train  '$WORKDIR/run_fpt_train.sh'"
echo "Watch:  tmux attach -t chess-server | chess-tunnel | chess-train"
