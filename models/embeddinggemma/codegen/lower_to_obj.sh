#!/usr/bin/env bash
# single_forward lowering with BLIS-style matmul and pre-OpenMP canonicalization.
# Remove one-trip parallel loops and combine directly nested parallel loops
# so a unit batch dimension does not hide the useful work from OpenMP.
# Keep thread selection at runtime, as in the default single_forward pipeline.
# As in the other BLIS model pipelines, omit affine-loop-fusion: combining it
# with the BLIS rewrite produces non-finite embeddings for this graph.
# Usage: lower_to_obj.sh <buddy-opt> <llvm-bin-dir> <in.mlir> <out.o> [threads] [llc-attrs...]
set -euo pipefail
BUDDY_OPT="$1"; LLVM_BIN="$2"; IN="$3"; OUT="$4"
shift 4
if [[ $# -gt 0 ]]; then
  shift
fi
LLC_ATTRS=("$@")
MATH_TRANSFORMS="$(dirname "$0")/tanh-approx.mlir"
KERNELS="$(dirname "$IN")/embeddinggemma_ime.mlir"
GRAPH_OUT="$OUT"
SOFTMAX_TRANSFORM=()
if [[ -f "$KERNELS" ]]; then
  SOFTMAX_TRANSFORM=(-transform-interpreter=entry-point=__vectorize_softmax_exp)
  : "${EMBEDDINGGEMMA_OBJECT_LINKER:?CMake must supply the target object linker}"
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  GRAPH_OUT="$TMP/graph.o"
fi

TOSA="builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"

"$LLVM_BIN/mlir-opt" "$IN" -pass-pipeline "$TOSA" \
  | "$LLVM_BIN/mlir-opt" -test-linalg-transform-patterns=test-decompose-pad-tensor \
  | "$BUDDY_OPT" \
      -arith-expand -eliminate-empty-tensors -convert-elementwise-to-linalg \
      -linalg-fold-into-elementwise \
      "-transform-preload-library=transform-library-paths=$MATH_TRANSFORMS" \
      -linalg-inline-scalar-operands \
      -transform-interpreter=entry-point=__flatten_scalar_muls \
      -empty-tensor-to-alloc-tensor \
      -one-shot-bufferize=bufferize-function-boundaries \
      -ownership-based-buffer-deallocation \
      -buffer-deallocation-simplification -bufferization-lower-deallocations \
      -matmul-vectorization-blis -batchmatmul-optimize \
      -transform-interpreter "${SOFTMAX_TRANSFORM[@]}" -cse \
      -convert-linalg-to-affine-loops \
      -affine-parallelize -lower-affine -canonicalize \
      -optimize-allocation-liveness \
      -convert-scf-to-openmp -convert-linalg-to-loops -convert-vector-to-scf \
      -expand-strided-metadata -lower-affine -cse -convert-vector-to-llvm \
      -memref-expand -convert-arith-to-llvm -finalize-memref-to-llvm \
      -convert-scf-to-cf -convert-cf-to-llvm -llvm-request-c-wrappers \
      -convert-openmp-to-llvm -convert-arith-to-llvm -convert-math-to-llvm \
      -convert-math-to-libm -convert-ub-to-llvm -convert-func-to-llvm \
      -reconcile-unrealized-casts \
  | "$LLVM_BIN/mlir-translate" -mlir-to-llvmir \
  | "$LLVM_BIN/llvm-as" \
  | "$LLVM_BIN/llc" "${LLC_ATTRS[@]}" \
      -filetype=obj -relocation-model=pic -O3 -o "$GRAPH_OUT"

if [[ -f "$KERNELS" ]]; then
  # Reuse the existing IME pipeline and A100/VLEN validation. Zero leaves
  # OpenMP thread selection to OMP_NUM_THREADS, as in the F32 graph.
  CODEGEN="$(dirname "$0")/../../../tools/buddy-codegen"
  "${PYTHON:-python3}" -c '
import shlex, sys
sys.path.insert(0, sys.argv[1])
from compile_pipeline import build_stages, run_pipeline
stages = build_stages("kernels_ime", 0, shlex.join(sys.argv[6:]))
run_pipeline(stages, sys.argv[4], sys.argv[5], sys.argv[2], sys.argv[3])
' "$CODEGEN" "$BUDDY_OPT" "$LLVM_BIN" "$KERNELS" "$TMP/kernels.o" "${LLC_ATTRS[@]}"
  "$EMBEDDINGGEMMA_OBJECT_LINKER" -r "$GRAPH_OUT" "$TMP/kernels.o" -o "$OUT"
fi
