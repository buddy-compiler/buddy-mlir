#!/usr/bin/env bash
# ===- bench.sh - BGE-M3 RAX benchmark (mode A cold start + mode B steady) ===//
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
# Usage: bash bench.sh <seq> [tag] [repeatB] [repeatA]
#   seq     : 128 | 256 | 512
#   tag     : default baseline (matches build_seq.sh tag)
#   repeatB : steady-state requests, default 20 (drop first)
#   repeatA : cold-start runs, default 10 (drop first)
#
# Products: $RESULTS/seq<seq>-<tag>/
#   latency_b.txt   steady-state latencies (s)
#   latency_a.txt   cold-start runs (wall_s maxRSS_KB)
#   emb.txt         embedding output (for the cosine gate)
#   summary.txt     median / stdev summary
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

SEQ=${1:?Usage: bench.sh <seq> [tag] [repeatB] [repeatA]}
TAG=${2:-baseline}
RB=${3:-20}
RA=${4:-10}

RAX="$OUT/seq$SEQ-$TAG/bge_m3.rax"
[ -f "$RAX" ] || { echo "not found: $RAX (run build_seq.sh first)"; exit 1; }

D="$RESULTS/seq$SEQ-$TAG"
mkdir -p "$D"
PORT=${PORT:-8090}
PROMPT="The quick brown fox jumps over the lazy dog."

echo "[bench] rax=$RAX tag=$TAG threads=${OMP_NUM_THREADS:-default}"

# ── Mode B: buddy-server resident, steady-state single requests ──────────
echo "[bench] mode B: start buddy-server (port $PORT)..."
OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}" \
  "$BUDDYBIN/buddy-server" --model "$RAX" --host 127.0.0.1 \
  --port "$PORT" >"$D/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

# Readiness: require HTTP 200 (Content-Type header is mandatory, otherwise
# the server replies 4xx immediately).
ready=0
for i in $(seq 1 120); do
  code=$(curl -s -o "$D/warm_resp.json" -w '%{http_code}' -X POST \
    "http://127.0.0.1:$PORT/v1/embeddings" \
    -H 'Content-Type: application/json' \
    -d "{\"input\":\"$PROMPT\"}" 2>/dev/null || true)
  [ "$code" = "200" ] && { ready=1; break; }
  sleep 1
done
if [ "$ready" != "1" ]; then
  echo "[bench] server not ready (http=$code). server.log tail:"
  tail -20 "$D/server.log"
  echo "[bench] first 300 chars of the response body:"
  head -c 300 "$D/warm_resp.json"; echo
  exit 1
fi
echo "[bench] ready (first 200 doubles as warm-up; first 200 chars:"
head -c 200 "$D/warm_resp.json"; echo ")"

: > "$D/latency_b.txt"
for i in $(seq 1 "$RB"); do
  curl -s -o "$D/resp_$i.json" -w '%{time_total}\n' -X POST \
    "http://127.0.0.1:$PORT/v1/embeddings" \
    -H 'Content-Type: application/json' \
    -d "{\"input\":\"$PROMPT\"}" >> "$D/latency_b.txt"
done
kill $SRV 2>/dev/null || true
trap - EXIT
sleep 2

# ── Mode A: buddy-cli cold start (includes 2.27 GB weight load) ──────────
echo "[bench] mode A: buddy-cli x $RA"
if [ -x /usr/bin/time ]; then
  TIME_BIN=/usr/bin/time
  TIME_FMT='%e %M'
else
  echo "[bench] note: /usr/bin/time missing -> wall clock only (no peak RSS)"
  TIME_BIN=""
  TIME_FMT=""
fi
: > "$D/latency_a.txt"
for i in $(seq 1 "$RA"); do
  if [ -n "$TIME_BIN" ]; then
    "$TIME_BIN" -f "$TIME_FMT" -o "$D/.t" \
      "$BUDDYBIN/buddy-cli" --model "$RAX" --prompt "$PROMPT" --no-stats \
      > "$D/emb.txt" 2>"$D/cli_err.txt"
    cat "$D/.t" >> "$D/latency_a.txt"
  else
    T0=$(date +%s.%N)
    "$BUDDYBIN/buddy-cli" --model "$RAX" --prompt "$PROMPT" --no-stats \
      > "$D/emb.txt" 2>"$D/cli_err.txt" &
    PID=$!
    HWM=0
    while kill -0 $PID 2>/dev/null; do
      H="$(awk '/VmHWM/{print $2}' /proc/$PID/status 2>/dev/null)"
      [ -n "$H" ] && HWM=$H
      sleep 0.2
    done
    wait $PID || true
    T1=$(date +%s.%N)
    awk "BEGIN{printf \"%.3f %d\n\", $T1-$T0, $HWM}" >> "$D/latency_a.txt"
  fi
done
rm -f "$D/.t"
if [ -s "$D/emb.txt" ]; then
  echo "[bench] ok: emb.txt has output"
elif [ "$RA" -gt 0 ]; then
  echo "[bench] ERROR: emb.txt is empty! cli_err.txt tail:"
  tail -20 "$D/cli_err.txt" 2>/dev/null || true
fi

# ── Summary (pure stdlib) ────────────────────────────────────────────────
python3 - "$D" <<'PY' | tee "$D/summary.txt"
import statistics
import sys
import os

d = sys.argv[1]


def rows(f):
    p = os.path.join(d, f)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return []
    return [l.split() for l in open(p) if l.strip()]


def fmt(v):
    return (f"median={statistics.median(v):.2f} "
            f"stdev={statistics.pstdev(v):.2f} n={len(v)}")


b = [float(x[0]) for x in rows("latency_b.txt")]
b = b[1:] if len(b) > 1 else b                       # drop first
a = rows("latency_a.txt")
a = a[1:] if len(a) > 1 else a
aw = [float(x[0]) for x in a]
ar = [int(x[1]) / 1024 for x in a
      if len(x) > 1 and x[1].isdigit()]              # MB
print(f"[modeB steady] ms : {fmt([x * 1000 for x in b]) if b else 'no data'}")
if aw:
    print(f"[modeA cold]   s  : {fmt(aw)}")
if ar:
    print(f"[modeA peakRSS] MB : median={statistics.median(ar):.0f}")
PY

echo "[bench] raw data -> $D"
echo "Next: cosine gate -> python3 $HERE/cos.py <baseline>/emb.txt $D/emb.txt"

