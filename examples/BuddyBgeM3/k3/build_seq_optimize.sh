#!/usr/bin/env bash
# ===- build_seq_optimize.sh - Fixed-16 matmul-vectorization experiment ---===//
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
# Experiment #6 of the K3 study (see docs/K3-benchmark-report.md 6.5.3):
# replace the baseline matmul pass with fixed-width vectorization
# -matmul-vectorization="vector-size=16". Result: zero/invalid CLS.
#
# Same structure as build_seq.sh; only SUB_PASSES differs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

SEQ=${1:?Usage: build_seq_optimize.sh <seq> [tag]}
TAG=${2:-baseline}
SRC="$APP/src/dist/seq$SEQ"
SHARED="$APP/src/dist/"
OUTD="$OUT/seq$SEQ-$TAG"
RT="$OUT/runtime"

res() {
  local p
  for p in "$SRC/$1" "$SHARED/$1"; do
    [ -f "$p" ] && { printf '%s' "$p"; return; }
  done
}

mkdir -p "$OUTD"
for f in forward.mlir subgraph0.mlir; do
  [ -f "$SRC/$f" ] || {
    echo "Missing $SRC/$f -- run export/export_artifacts.sh on x86 first"
    exit 1
  }
done
ARG0="$(res arg0.data)"
TOK="$(res tokenizer.json)"
[ -n "$ARG0" ] && [ -n "$TOK" ] || {
  echo "Missing arg0.data / tokenizer.json (put them in $SHARED/ or $SRC/)"
  exit 1
}
[ -f "$RT/bge_m3_runner.so" ] || {
  echo "Missing runtime -- run: bash $HERE/build_runtime.sh"
  exit 1
}

MOPT="$LLVMBIN/mlir-opt"
BOPT="$BUDDYBIN/buddy-opt"
MTR="$LLVMBIN/mlir-translate"
AS="$LLVMBIN/llvm-as"
LLC="$LLVMBIN/llc"
CXX="$LLVMBIN/clang++"

# Experiment knob: fixed-width matmul vectorization (experiment #6).
SUB_PASSES=(
  -arith-expand
  -eliminate-empty-tensors
  -convert-elementwise-to-linalg
  -empty-tensor-to-alloc-tensor
  -one-shot-bufferize=bufferize-function-boundaries
  -ownership-based-buffer-deallocation
  -buffer-deallocation-simplification
  -bufferization-lower-deallocations
  -matmul-vectorization="vector-size=16"
  -convert-linalg-to-affine-loops
  -affine-loop-fusion
  -affine-parallelize
  -lower-affine
  -convert-scf-to-openmp
  -convert-linalg-to-loops
  -convert-vector-to-scf
  -expand-strided-metadata
  -lower-affine
  -cse
  -convert-vector-to-llvm
  -memref-expand
  -convert-arith-to-llvm
  -finalize-memref-to-llvm
  -convert-scf-to-cf
  -convert-cf-to-llvm
  -llvm-request-c-wrappers
  -convert-openmp-to-llvm
  -convert-arith-to-llvm
  -convert-math-to-llvm
  -convert-math-to-libm
  -convert-func-to-llvm
  -reconcile-unrealized-casts
)

echo "[seq$SEQ/$TAG] 1/4 forward.mlir -> forward.o"
"$MOPT" "$SRC/forward.mlir" -pass-pipeline \
  "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, \
  tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, \
  convert-elementwise-to-linalg)" \
| "$BOPT" -pass-pipeline \
  "builtin.module(func.func(buffer-deallocation-simplification, \
  convert-linalg-to-loops), matmul-parallel-vectorization-optimize, \
  batchmatmul-optimize, eliminate-empty-tensors, \
  func.func(llvm-request-c-wrappers), convert-scf-to-openmp, \
  convert-openmp-to-llvm, convert-math-to-llvm, convert-math-to-libm, \
  convert-scf-to-cf, convert-arith-to-llvm, expand-strided-metadata, \
  finalize-memref-to-llvm, convert-func-to-llvm, reconcile-unrealized-casts)" \
| "$MTR" -mlir-to-llvmir | "$AS" \
| "$LLC" $LLC_ATTRS -filetype=obj -relocation-model=pic -O0 \
  -o "$OUTD/forward.o"

echo "[seq$SEQ/$TAG] 2/4 subgraph0.mlir -> subgraph0.o (slowest, be patient)"
"$MOPT" "$SRC/subgraph0.mlir" \
  -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, \
  tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" \
| "$MOPT" -test-linalg-transform-patterns=test-decompose-pad-tensor \
| "$BOPT" "${SUB_PASSES[@]}" \
| "$MTR" -mlir-to-llvmir | "$AS" \
| "$LLC" $LLC_ATTRS -filetype=obj -relocation-model=pic -O3 \
  -o "$OUTD/subgraph0.o"

echo "[seq$SEQ/$TAG] 3/4 link bge_m3_model.so"
OMP="$(find "$LLVM" -maxdepth 6 -name 'libomp.so' -print -quit)"
RUNNER="$(find "$LLVM" -maxdepth 6 \
  -name 'libmlir_c_runner_utils.so' -print -quit)"
[ -n "$OMP" ] || {
  echo "libomp.so not found (build LLVM with -DLLVM_ENABLE_RUNTIMES=openmp)"
  exit 1
}
[ -n "$RUNNER" ] || { echo "libmlir_c_runner_utils.so not found"; exit 1; }
OMP_DIR="$(dirname "$OMP")"
RUN_DIR="$(dirname "$RUNNER")"
"$CXX" -shared -fPIC -Wl,-soname,bge_m3_model.so \
  -Wl,--allow-multiple-definition \
  -o "$OUTD/bge_m3_model.so" "$OUTD/forward.o" "$OUTD/subgraph0.o" \
  "$OMP" "$RUNNER" \
  -Wl,-rpath,"$OMP_DIR" -Wl,-rpath,"$RUN_DIR" -Wl,-rpath,'$ORIGIN' -lm
cp -f "$OMP"    "$OUTD/libomp.so"
cp -f "$RUNNER" "$OUTD/libmlir_c_runner_utils.so"
if ldd "$OUTD/bge_m3_model.so" 2>/dev/null | grep -q 'not found'; then
  echo "warning: unresolved deps:"
  ldd "$OUTD/bge_m3_model.so" | grep 'not found'
else
  echo "        ok: runtime deps resolved (ldd)"
fi

echo "[seq$SEQ/$TAG] 4/4 pack bge_m3.rax"
cp -f "$RT/bge_m3_runner.so" "$RT/bge_m3_embedding.so" "$OUTD/"
cp -f "$ARG0" "$OUTD/arg0.data"
cp -f "$TOK"  "$OUTD/tokenizer.json"

SPEC="$SRC/seq$SEQ.json"
[ -f "$SPEC" ] || SPEC="$REPO/models/bge_m3/specs/seq$SEQ.json"
if [ -f "$SRC/generated/bge_m3.mlir" ]; then
  cp -f "$SRC/generated/bge_m3.mlir" "$OUTD/bge_m3.mlir"
else
  python3 "$REPO/models/bge_m3/codegen/gen_bge_m3_manifest.py" \
    --spec "$SPEC" -o "$OUTD/bge_m3.mlir" \
    --runner-library bge_m3_runner.so \
    --embedding-library bge_m3_embedding.so
fi

( cd "$OUTD" && "$BUDDYBIN/rax-pack" bge_m3.mlir -o bge_m3.rax \
  --embed-payload )

echo "[seq$SEQ/$TAG] done -> $OUTD/bge_m3.rax"
ls -lh "$OUTD/bge_m3.rax"
echo
echo "Next steps:"
echo "  correctness: python3 $HERE/cos.py <baseline-emb> <new-emb>"
echo "  benchmark  : bash $HERE/bench.sh $SEQ $TAG"
echo "  RVV profile: bash $HERE/profile.sh $SEQ $TAG"

