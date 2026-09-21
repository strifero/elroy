#!/bin/bash
# Chapter 7: the three code benchmarks on the chat checkpoint, chat template, one GPU each.
#   bash scripts/eval_chat.sh [checkpoint]
set -u
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"
CKPT=${1:-checkpoints/elroy-350m-chat/chat-final.pt}
mkdir -p results/chat logs
CUDA_VISIBLE_DEVICES=0 nohup python scripts/eval_code.py --ckpt "$CKPT" --bench humaneval --chat \
    --out results/chat/humaneval.jsonl > logs/eval-chat-humaneval.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=1 nohup python scripts/eval_code.py --ckpt "$CKPT" --bench mbpp --chat \
    --out results/chat/mbpp.jsonl > logs/eval-chat-mbpp.log 2>&1 < /dev/null &
wait
CUDA_VISIBLE_DEVICES=1 nohup python scripts/eval_code.py --ckpt "$CKPT" --bench humanevalplus --chat \
    --out results/chat/humanevalplus.jsonl > logs/eval-chat-humanevalplus.log 2>&1 < /dev/null &
wait
echo "all chat evals done" >> logs/eval-chat-humaneval.log
