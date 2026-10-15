#!/usr/bin/env bash
# Serial benchmark of the given seq/tag pairs (default: all exp_a100).
# Usage: bash bench_all.sh ["seq tag" ...]   e.g. bash bench_all.sh "256 exp_a100" "512 exp_a100"
set -u
K3DIR="$HOME/buddy-k3/buddy-mlir/examples/BuddyBgeM3"
source "$K3DIR/env.sh"
cd "$K3DIR"

SPECS=("$@")
if [ ${#SPECS[@]} -eq 0 ]; then
  SPECS=("128 exp_a100" "256 exp_a100" "512 exp_a100")
fi

for spec in "${SPECS[@]}"; do
  set -- $spec
  bash bench.sh "$1" "$2" 10 1 > "/tmp/bench_$1_$2.log" 2>&1
  echo "done: seq$1-$2"
done

echo "ALL_BENCH_DONE"
for spec in "${SPECS[@]}"; do
  set -- $spec
  echo "=== seq$1-$2 ==="
  cat "$RESULTS/seq$1-$2/summary.txt" 2>/dev/null
  echo
  cat "$RESULTS/seq$1-$2/latency_b.txt" 2>/dev/null
  echo
  cat "$RESULTS/seq$1-$2/latency_a.txt" 2>/dev/null
  echo
done
