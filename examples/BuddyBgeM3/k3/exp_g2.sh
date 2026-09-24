#!/usr/bin/env bash
# ===- exp_g2.sh - Experiment G2: +zvl256b -riscv-v-vector-bits-min=256 -===//
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
# Experiment #8 of the K3 study (see docs/K3-benchmark-report.md 6.5.5):
# keep the IR unchanged (baseline passes), only tell the llc backend that
# 256-bit vectors are guaranteed. Lowest correctness risk; if it passes the
# cosine gate and is faster, it becomes the Before/After pair.
#
# Result: cos = 0.340539329 FAIL, identical to the -mcpu=spacemit-x100
# experiment (#7) -> same upstream backend bug.
#
# Usage: bash exp_g2.sh [seq]    default 128
# Products: out/seq<seq>-exp4_g2/ and results/seq<seq>-exp4_g2/
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

SEQ=${1:-128}
TAG=exp4_g2

# Override LLC_ATTRS (env.sh uses default-value semantics).
export LLC_ATTRS="-march=riscv64 -mattr=+m,+d,+v,+zvl256b \
  -mtriple=riscv64-unknown-linux-gnu -riscv-v-vector-bits-min=256"

echo "[exp_g2] 1/4 build (baseline passes + zvl256b backend)"
bash "$HERE/build_seq.sh" "$SEQ" "$TAG"

echo "[exp_g2] 2/4 cosine gate"
"$BUDDYBIN/buddy-cli" --model "$OUT/seq$SEQ-$TAG/bge_m3.rax" \
  --prompt "hello world" --no-stats > /tmp/emb_$TAG.txt
python3 "$HERE/cos.py" "$RESULTS/seq$SEQ-baseline/emb.txt" /tmp/emb_$TAG.txt

echo "[exp_g2] 3/4 RVV instruction census"
bash "$HERE/profile.sh" "$SEQ" "$TAG"

echo "[exp_g2] 4/4 benchmark"
bash "$HERE/bench.sh" "$SEQ" "$TAG" 10 5

echo "[exp_g2] done. Compare: $RESULTS/seq$SEQ-baseline/summary.txt"
echo "  vs $RESULTS/seq$SEQ-$TAG/summary.txt"

