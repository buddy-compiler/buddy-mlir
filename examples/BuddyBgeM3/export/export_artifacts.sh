#!/usr/bin/env bash
# ===- export_artifacts.sh - x86 one-time preparation (only torch step) ---===//
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
# Generates all architecture-independent artifacts that the K3 build needs
# for every seq variant, and packs them into dist/ for transfer.
#
# Usage:
#   bash export_artifacts.sh            # default 128 256 512
#   bash export_artifacts.sh 128        # a single variant
#   DOWNLOAD=1 bash export_artifacts.sh # re-download weights if missing
#
# Products (dist/):
#   arg0.data                    weights (shared by all seq, ~2.27 GB)
#   tokenizer.json               shared
#   seq<L>/forward.mlir          graph for this seq (shape varies with seq)
#   seq<L>/subgraph0.mlir
#   seq<L>/generated/bge_m3.mlir RHAL manifest (plain text)
#   seq<L>/seq<L>.json           spec (to regenerate the manifest on K3)
#
# Transfer:
#   scp -r dist user@k3-003:~/buddy-k3/buddy-mlir/examples/BuddyBgeM3/src
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$HERE/.."
REPO="$(cd "$APP/../.." && pwd)"
cd "$REPO"

# Optional local x86 environment (defines PY / LOCAL_BGE_M3).
[ -f "$APP/env.sh" ] && source "$APP/env.sh"
PY="${PY:-python3}"
LOCAL_BGE_M3="${LOCAL_BGE_M3:-$HOME/buddy-models/bge-m3}"

if [ $# -gt 0 ]; then
  SEQS=("$@")
else
  SEQS=(128 256 512)
fi

DIST="$APP/dist"
LOGS="$APP/logs"
mkdir -p "$DIST" "$LOGS"

# ── 0. Optional: re-download the model ───────────────────────────────────
if [ "${DOWNLOAD:-0}" = "1" ]; then
  echo "=== download BGE-M3 -> $LOCAL_BGE_M3 ==="
  "$PY" "$APP/export/download_bge_m3.py" --out-dir "$LOCAL_BGE_M3" || true
  ls -lh "$LOCAL_BGE_M3"/pytorch_model.bin
fi

# ── 0b. Make sure the per-seq specs exist ────────────────────────────────
if [ ! -f models/bge_m3/specs/seq128.json ]; then
  echo "=== generate seq{128,256,512} specs ==="
  "$PY" "$APP/export/gen_specs.py"
fi

for L in "${SEQS[@]}"; do
  echo
  echo "=========== seq$L ==========="
  BD="$REPO/build"
  STAMP="$BD/models/bge_m3/.buddy_import_done"
  rm -f "$STAMP"

  # Start the build; it imports (generates MLIR + weights) before the
  # expensive llc stage.
  "$PY" tools/buddy-codegen/build_model.py \
    --spec "models/bge_m3/specs/seq$L.json" \
    --build-dir build --local-model "$LOCAL_BGE_M3" \
    > "$LOGS/export-seq$L.log" 2>&1 &
  PID=$!

  # Wait for the import only: K3 compiles the rest natively.
  echo "[seq$L] waiting for import (see $LOGS/export-seq$L.log)..."
  ok=0
  for _ in $(seq 1 240); do
    if [ -f "$STAMP" ]; then
      ok=1
      break
    fi
    kill -0 "$PID" 2>/dev/null || break
    sleep 5
  done
  kill "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true

  if [ "$ok" != "1" ]; then
    echo "[seq$L] import did not finish -- see $LOGS/export-seq$L.log"
    echo "        if MLIR/weights were generated, ignore; else check torch."
  fi

  B="$BD/models/bge_m3"
  DST="$DIST/seq$L"
  mkdir -p "$DST/generated"

  for f in forward.mlir subgraph0.mlir; do
    cp -f "$B/$f" "$DST/" || { echo "[seq$L] missing $B/$f"; exit 1; }
  done

  # Manifest (pure stdlib, does not depend on the build).
  "$PY" models/bge_m3/codegen/gen_bge_m3_manifest.py \
    --spec "models/bge_m3/specs/seq$L.json" -o "$DST/generated/bge_m3.mlir" \
    --runner-library bge_m3_runner.so \
    --embedding-library bge_m3_embedding.so

  cp -f "models/bge_m3/specs/seq$L.json" "$DST/"

  # Shared big files: keep a single copy.
  [ -f "$DIST/arg0.data" ] || cp "$B/arg0.data" "$DIST/arg0.data"
  [ -f "$DIST/tokenizer.json" ] \
    || cp "$LOCAL_BGE_M3/tokenizer.json" "$DIST/tokenizer.json"

  echo "[seq$L] ok -> $DST"
  ls -lh "$DST"
done

echo
echo "=== done ==="
du -sh "$DIST"
echo
echo "Transfer to K3:"
echo "  scp -r $DIST \\"
echo "    user@k3-003:~/buddy-k3/buddy-mlir/examples/BuddyBgeM3/src"

