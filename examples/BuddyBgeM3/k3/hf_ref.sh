#!/usr/bin/env bash
# ===- hf_ref.sh - Same-hardware HF reference on K3 -----------------------===//
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
# Installs torch on K3 (user-mode uv + python3.12 + Ruyi index), downloads
# BGE-M3, and runs export/hf_bench.py with the exact RAX configuration
# (16 threads, f32, same text, same max_length) for seq 128/256/512.
#
# Usage: bash hf_ref.sh
# Idempotent: every step checks before redoing; safe to rerun.
# Budget: torch install 5-20 min; model download (2.2 GB) depends on network.
#
# Prerequisite (one-time, from x86, if hf_bench.py is missing):
#   scp examples/BuddyBgeM3/export/hf_bench.py \
#       examples/BuddyBgeM3/export/download_bge_m3.py \
#       user@k3-003:~/buddy-k3/buddy-mlir/examples/BuddyBgeM3/export/
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

VENV="$ROOT/venv312"
MODEL_DIR="${HF_MODEL_DIR:-$ROOT/models/bge-m3-hf}"
HFBENCH="$APP/export/hf_bench.py"
THREADS="${THREADS:-16}"
RES="$RESULTS/hf_ref_k3"

mkdir -p "$RES"

echo "=== 0/3 check hf_bench.py ==="
if [ ! -f "$HFBENCH" ]; then
  echo "missing $HFBENCH. On x86 run:"
  echo "  scp examples/BuddyBgeM3/export/hf_bench.py \\"
  echo "    user@k3-003:~/buddy-k3/buddy-mlir/examples/BuddyBgeM3/export/"
  exit 1
fi

echo "=== 1/3 install uv + python3.12 + torch (idempotent) ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [ ! -x "$VENV/bin/python" ]; then
  uv python install 3.12
  uv venv "$VENV" --python 3.12
fi
source "$VENV/bin/activate"

if ! python -c "import torch, transformers" 2>/dev/null; then
  echo "[hf_ref] installing deps (Ruyi torch, ~5-20 min)..."
  uv pip install -U pip setuptools wheel packaging
  uv pip install --index-strategy unsafe-best-match \
    -r "$REPO/requirements.txt" \
    --extra-index-url https://ruyirepo.ruyicommunity.cn/pypi/simple/
fi
python - <<'PY'
import torch
import transformers

print(f"[hf_ref] torch={torch.__version__} "
      f"transformers={transformers.__version__}")
PY

echo "=== 2/3 download BGE-M3 weights (2.2 GB, idempotent) ==="
if [ ! -f "$MODEL_DIR/pytorch_model.bin" ]; then
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" python - <<PY
from huggingface_hub import snapshot_download

snapshot_download("BAAI/bge-m3", local_dir="$MODEL_DIR")
PY
fi
ls -lh "$MODEL_DIR/pytorch_model.bin"

echo "=== 3/3 run reference (128/256/512, $THREADS threads) ==="
for L in 128 256 512; do
  echo "----- max_length=$L -----"
  python "$HFBENCH" --model-dir "$MODEL_DIR" \
    --max-length "$L" --threads "$THREADS" --repeat 10 \
    | tee "$RES/hf_ref_k3_${L}.txt"
done

echo
echo "done -> $RES/"
echo "fill the 'steady-state median' values into the K3 comparison table"
echo "in docs/K3-benchmark-report.md section 3.3."

