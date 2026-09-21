#!/usr/bin/env bash
# ===- profile.sh - RVV instruction census + VLEN -------------------------===//
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
# Usage: bash profile.sh <seq> [tag]
# Products:
#   $PROFILE/seq<seq>-<tag>-rvv.txt  instruction histogram
#   $PROFILE/vlen.txt                vector register width of the X100
#
# Interpretation:
#   many vfmacc / vle32.v -> bandwidth / thread bound
#   almost none           -> run the scalable vectorization experiments
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

SEQ=${1:?Usage: profile.sh <seq> [tag]}
TAG=${2:-baseline}
SO="$OUT/seq$SEQ-$TAG/bge_m3_model.so"
[ -f "$SO" ] || { echo "not found: $SO"; exit 1; }

OUTF="$PROFILE/seq$SEQ-$TAG-rvv.txt"
echo "[profile] objdump $SO"
RVV_PAT='vsetvli|vsetivli|vle32\.v|vse32\.v|vfmacc\.vf|vfmacc\.vv|'
RVV_PAT="${RVV_PAT}vfmul\.vf|vfredusum|vfredosum|vfredmax|vfredmin"
"$LLVMBIN/llvm-objdump" -d "$SO" \
  | grep -oE "$RVV_PAT" \
  | sort | uniq -c | sort -rn | tee "$OUTF"
echo "[profile] -> $OUTF"

# VLEN
if [ ! -f "$PROFILE/vlen.txt" ]; then
  cat > /tmp/vlen.c <<'EOF'
#include <stdio.h>
int main(void) {
  unsigned long v;
  __asm__ volatile("csrr %0, vlenb" : "=r"(v));
  printf("VLEN=%lu bits\n", v * 8);
  return 0;
}
EOF
  gcc /tmp/vlen.c -o /tmp/vlen && /tmp/vlen | tee "$PROFILE/vlen.txt"
fi
cat "$PROFILE/vlen.txt"

