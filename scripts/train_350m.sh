#!/bin/bash
# Launch (or resume) the Elroy-350M pretraining run on both GPUs, detached, logging to logs/.
#   bash scripts/train_350m.sh            # start or resume
#   tail -f logs/train-350m.log           # watch
#   pkill -f 'elroy.train'                # stop; rerun this script to resume from the last checkpoint
#
# Anything else that holds GPU memory has to be out of the way first. On our box
# that is ComfyUI (pm2) and Ollama (moved to another host for the duration).
set -u
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"
mkdir -p logs checkpoints
if command -v pm2 >/dev/null && pm2 jlist 2>/dev/null | grep -q '"name":"comfyui"'; then
  pm2 stop comfyui >/dev/null 2>&1 && echo "stopped comfyui"
fi
if pgrep -f 'elroy.train' >/dev/null; then
  echo "a training process is already running"; exit 1
fi
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
setsid nohup torchrun --standalone --nproc_per_node=2 -m elroy.train --config configs/elroy-350m.yaml \
    >> logs/train-350m.log 2>&1 < /dev/null &
sleep 2
echo "started: pid $(pgrep -f 'torchrun.*elroy.train' | head -1), log logs/train-350m.log"
