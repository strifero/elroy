#!/bin/bash
# Chapter 5: evaluate the base checkpoint on the three code benchmarks, one GPU each,
# detached. Results land in results/base/<bench>.jsonl plus a summary line in the log.
#   bash scripts/eval_base.sh [checkpoint]
set -u
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"
CKPT=${1:-checkpoints/elroy-350m/base-final.pt}
mkdir -p results/base logs
CUDA_VISIBLE_DEVICES=0 nohup python scripts/eval_code.py --ckpt "$CKPT" --bench humaneval \
    --out results/base/humaneval.jsonl > logs/eval-base-humaneval.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=1 nohup python scripts/eval_code.py --ckpt "$CKPT" --bench mbpp \
    --out results/base/mbpp.jsonl > logs/eval-base-mbpp.log 2>&1 < /dev/null &
wait
CUDA_VISIBLE_DEVICES=0 nohup python scripts/eval_code.py --ckpt "$CKPT" --bench humanevalplus \
    --out results/base/humanevalplus.jsonl > logs/eval-base-humanevalplus.log 2>&1 < /dev/null &
wait
echo "all evals done" >> logs/eval-base-humaneval.log
