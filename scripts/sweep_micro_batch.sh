#!/bin/bash
# Chapter 4: find the per-GPU micro-batch that gives the best throughput on THIS hardware.
# Runs a few optimizer steps per setting and prints tokens/s and MFU from the log.
#   bash scripts/sweep_micro_batch.sh configs/elroy-350m.yaml data/shards-dev "4 8 12 16"
set -u
CONFIG=${1:-configs/elroy-350m.yaml}
SHARDS=${2:-data/shards-dev}
SIZES=${3:-"4 8 12 16"}
NGPU=${NGPU:-2}
for mb in $SIZES; do
  rm -rf /tmp/sweep-ck
  echo "== micro_batch $mb"
  torchrun --standalone --nproc_per_node=$NGPU -m elroy.train --config "$CONFIG" --shards "$SHARDS" \
      --out /tmp/sweep-ck --max-steps 6 --micro-batch "$mb" --eval-windows 4 --no-resume 2>&1 \
    | grep -E "^step|out of memory|Error" | tail -3
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
done
