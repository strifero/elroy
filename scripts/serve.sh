#!/bin/bash
# Start (or restart) the Elroy chat server on GPU 0, detached, port 8100.
#   bash scripts/serve.sh [checkpoint]
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"
CKPT=${1:-checkpoints/elroy-350m-chat/latest.pt}
pkill -f 'scripts/serve.py' 2>/dev/null
sleep 1
CUDA_VISIBLE_DEVICES=0 setsid nohup python scripts/serve.py --ckpt "$CKPT" --port 8100 > logs/serve.log 2>&1 < /dev/null &
sleep 6
tail -n 2 logs/serve.log
