#!/usr/bin/env bash
# ===- thread_scaling.sh - Thread scaling experiment (OMP 16/8/4/1) -------===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===----------------------------------------------------------------------===//
#
# Usage (run under nohup so disconnects are harmless):
#   nohup bash $APP/k3/thread_scaling.sh > $RESULTS/thread_scaling.txt 2>&1 &
#   tail -f $RESULTS/thread_scaling.txt
#
# Key fixes:
#   1) kill orphan buddy-server instances first (SSH disconnects used to
#      leave them around and exhaust memory);
#   2) readiness timeout 600s (warm-up inference at T=1 takes ~5 min);
#   3) one server at a time, killed right after 2 samples;
#   4) start from 16 threads (T=1 is slowest, runs last).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

RAX="$OUT/seq128-baseline/bge_m3.rax"
[ -f "$RAX" ] || {
  echo "not found: $RAX (run build_seq.sh 128 first)"
  exit 1
}

echo "== clean orphan servers =="
pkill -f 'buddy-server --model' 2>/dev/null || true
sleep 3
echo "== current memory =="
free -g | head -2
echo "== leftover processes (should be empty) =="
pgrep -af buddy-server || echo "(none)"

for T in 16 8 4 1; do
  PORT=$((8130 + T))
  echo
  echo "===== OMP_NUM_THREADS=$T (port $PORT) ====="
  OMP_NUM_THREADS=$T "$BUDDYBIN/buddy-server" --model "$RAX" \
    --host 127.0.0.1 --port "$PORT" >"/tmp/srv_$T.log" 2>&1 &
  SRV=$!
  cleanup() { kill "$SRV" 2>/dev/null || true; }
  trap cleanup EXIT

  ready=0
  for i in $(seq 1 600); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST \
      "127.0.0.1:$PORT/v1/embeddings" -H 'Content-Type: application/json' \
      -d '{"input":"warm"}' 2>/dev/null || true)
    [ "$code" = "200" ] && { ready=1; break; }
    sleep 1
  done
  if [ "$ready" != "1" ]; then
    echo "  [WARN] not ready in 600s, skipping. server.log tail:"
    tail -5 "/tmp/srv_$T.log"
    cleanup
    trap - EXIT
    continue
  fi
  echo "  ready (warm-up done). 2 samples, seconds:"
  for i in 1 2; do
    curl -s -o /dev/null -w '  %{time_total}\n' -X POST \
      "127.0.0.1:$PORT/v1/embeddings" -H 'Content-Type: application/json' \
      -d '{"input":"hello world"}'
  done
  cleanup
  trap - EXIT
  sleep 2
done

echo
echo "== done. final memory =="
free -g | head -2
echo "output -> $RESULTS/thread_scaling.txt"

