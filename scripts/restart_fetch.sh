#!/bin/bash
# Stop any running Stack-Edu fetcher and start N parallel copies (default 4).
# The first copy builds the ranked index file if it is missing; the others wait for it.
cd "$(dirname "$0")/.."
N=${1:-4}
pkill -f '[f]etch_stack_edu.py'
sleep 2
rm -f data/raw/stack-edu-text/*.tmp
echo "shards on disk: $(ls data/raw/stack-edu-text/shard-*.parquet 2>/dev/null | wc -l)"
start() {
  setsid nohup .venv/bin/python scripts/fetch_stack_edu.py --target-bytes 32e9 --workers 96 \
      --process-index "$1" --process-count "$N" > "logs/fetch-p$1.log" 2>&1 < /dev/null &
}
start 0
until ls data/raw/stack-edu-text/ranked-index-*.parquet >/dev/null 2>&1; do sleep 5; done
for i in $(seq 1 $((N - 1))); do start "$i"; done
sleep 3
echo "running: $(pgrep -fc '[f]etch_stack_edu')"
